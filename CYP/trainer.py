import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from model import CypGCN, mol_to_graph, save_checkpoint


def scaffold_split(smiles_list, frac_train=0.7, frac_val=0.2, seed=42):
    groups = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        groups[scaffold].append(idx)

    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)

    n = len(smiles_list)
    n_train, n_val = int(frac_train * n), int(frac_val * n)
    train, val, test = [], [], []
    for group in ordered:
        if len(train) + len(group) <= n_train:
            train += group
        elif len(val) + len(group) <= n_val:
            val += group
        else:
            test += group

    print(f"scaffold split: {len(groups)} scaffolds -> {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


def build_dataset(csv_path):
    df = pd.read_csv(csv_path)
    graphs, skipped = [], 0
    for smiles, y in zip(df["smiles"], df["y"]):
        g = mol_to_graph(smiles, y=y)
        if g is None:
            skipped += 1
            continue
        graphs.append(g)
    if skipped:
        print(f"skipped {skipped} molecules RDKit could not featurize")
    return graphs


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, logits, targets = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        losses.append(criterion(out, batch.y).item() * batch.num_graphs)
        logits.append(out.cpu())
        targets.append(batch.y.cpu())

    logits = torch.cat(logits).numpy()
    targets = torch.cat(targets).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return {
        "loss": sum(losses) / len(targets),
        "auroc": roc_auc_score(targets, probs),
        "auprc": average_precision_score(targets, probs),
        "acc": ((probs > 0.5) == targets).mean(),
    }


def build_optimizer(model, lr, weight_decay):
    decay = [p for p in model.parameters() if p.dim() > 1]
    no_decay = [p for p in model.parameters() if p.dim() <= 1]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def train_model(
    csv,
    out,
    hidden=128,
    layers=3,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    epochs=150,
    patience=25,
    frac_train=0.7,
    frac_val=0.2,
    seed=42,
    print_every=10,
    use_pos_weight=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = build_dataset(csv)
    train_idx, val_idx, test_idx = scaffold_split([g.smiles for g in graphs], frac_train, frac_val, seed)

    loaders = {
        "train": DataLoader([graphs[i] for i in train_idx], batch_size=batch_size, shuffle=True),
        "val": DataLoader([graphs[i] for i in val_idx], batch_size=batch_size),
        "test": DataLoader([graphs[i] for i in test_idx], batch_size=batch_size),
    }

    config = {"hidden": hidden, "layers": layers, "dropout": dropout}
    model = CypGCN(**config).to(device)
    optimizer = build_optimizer(model, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    if use_pos_weight:
        y_train = torch.cat([graphs[i].y for i in train_idx])
        pos_weight = ((y_train == 0).sum() / (y_train == 1).sum()).to(device)
        print(f"pos_weight = {pos_weight.item():.3f} (computed on the training split only)")
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | seed {seed} | {layers} GCN layers, hidden {hidden}, "
          f"dropout {dropout}, weight decay {weight_decay} | {n_params:,} params\n")

    best_auprc, best_epoch, best_state = 0.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs

        val = evaluate(model, loaders["val"], device, criterion)
        scheduler.step(val["auprc"])

        improved = val["auprc"] > best_auprc
        if improved:
            best_auprc, best_epoch = val["auprc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % print_every == 0 or improved:
            print(
                f"epoch {epoch:3d}  train loss {total / len(train_idx):.4f}  "
                f"val loss {val['loss']:.4f}  auroc {val['auroc']:.4f}  auprc {val['auprc']:.4f}"
                f"{'  <- best' if improved else ''}"
            )

        if epoch - best_epoch >= patience:
            print(f"\nno val improvement for {patience} epochs, stopping")
            break

    model.load_state_dict(best_state)
    print(f"\nbest epoch {best_epoch} (val auprc {best_auprc:.4f})")
    results = {}
    for split in ("train", "val", "test"):
        results[split] = evaluate(model, loaders[split], device, criterion)
        m = results[split]
        print(f"  {split:<5}  auroc {m['auroc']:.4f}  auprc {m['auprc']:.4f}  acc {m['acc']:.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        out,
        config,
        isoform=csv.stem,
        split="scaffold",
        seed=seed,
        weight_decay=weight_decay,
        pos_weight=pos_weight.item() if pos_weight is not None else None,
        metrics={
            "val_auprc": results["val"]["auprc"],
            "test_auprc": results["test"]["auprc"],
            "test_auroc": results["test"]["auroc"],
        },
    )
    print(f"\nsaved {out}")
    return results
