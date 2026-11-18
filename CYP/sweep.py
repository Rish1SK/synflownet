import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from model import CypGCN
from trainer import build_dataset, build_optimizer, evaluate, scaffold_split

HERE = Path(__file__).parent

WEIGHT_DECAYS = [0.0, 1e-3, 1e-2, 1e-1, 3e-1]
DROPOUTS = [0.1, 0.2, 0.3, 0.5]
HIDDENS = [64, 128]

LAYERS = 3
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 150
PATIENCE = 25
FRAC_TRAIN = 0.7
FRAC_VAL = 0.2
SEED = 42


def run_one(loaders, hidden, dropout, weight_decay, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = CypGCN(hidden=hidden, layers=LAYERS, dropout=dropout).to(device)
    optimizer = build_optimizer(model, LR, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val, best_epoch, best_state = 0.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch.x, batch.edge_index, batch.batch), batch.y)
            loss.backward()
            optimizer.step()

        val = evaluate(model, loaders["val"], device, criterion)
        scheduler.step(val["auprc"])
        if val["auprc"] > best_val:
            best_val, best_epoch = val["auprc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            break

    model.load_state_dict(best_state)
    out = {split: evaluate(model, loaders[split], device, criterion) for split in loaders}
    return {
        "val_auprc": out["val"]["auprc"],
        "val_auroc": out["val"]["auroc"],
        "test_auprc": out["test"]["auprc"],
        "test_auroc": out["test"]["auroc"],
        "test_acc": out["test"]["acc"],
        "train_auprc": out["train"]["auprc"],
        "gap": out["train"]["auprc"] - out["test"]["auprc"],
        "best_epoch": best_epoch,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def main():
    ap = argparse.ArgumentParser(description="Grid sweep for one CYP isoform")
    ap.add_argument("--isoform", required=True, help="e.g. cyp2d6")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or HERE / f"sweep_{args.isoform}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = build_dataset(HERE / "csv" / f"{args.isoform}_veith.csv")
    tr, va, te = scaffold_split([g.smiles for g in graphs], FRAC_TRAIN, FRAC_VAL, SEED)
    loaders = {
        "train": DataLoader([graphs[i] for i in tr], batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader([graphs[i] for i in va], batch_size=BATCH_SIZE),
        "test": DataLoader([graphs[i] for i in te], batch_size=BATCH_SIZE),
    }

    grid = [(h, d, w) for h in HIDDENS for d in DROPOUTS for w in WEIGHT_DECAYS]
    results, started = [], time.time()
    print(f"{args.isoform}: {len(grid)} configs, {len(tr)}/{len(va)}/{len(te)} split, device {device}",
          flush=True)

    for i, (hidden, dropout, wd) in enumerate(grid, 1):
        t0 = time.time()
        metrics = run_one(loaders, hidden, dropout, wd, device)
        row = {"hidden": hidden, "dropout": dropout, "weight_decay": wd,
               "secs": round(time.time() - t0, 1), **metrics}
        results.append(row)

        best = max(results, key=lambda r: r["val_auprc"])
        out_path.write_text(json.dumps({
            "isoform": args.isoform,
            "fixed": {"layers": LAYERS, "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                      "patience": PATIENCE, "seed": SEED, "optimizer": "AdamW",
                      "split": "scaffold", "frac": [FRAC_TRAIN, FRAC_VAL]},
            "grid": {"hidden": HIDDENS, "dropout": DROPOUTS, "weight_decay": WEIGHT_DECAYS},
            "n_done": len(results),
            "n_total": len(grid),
            "best": best,
            "experiments": results,
        }, indent=2))

        print(f"[{i:>2}/{len(grid)}] h={hidden:<4} drop={dropout:<4} wd={wd:<6} "
              f"val_auprc={row['val_auprc']:.4f} test_auprc={row['test_auprc']:.4f} "
              f"test_auroc={row['test_auroc']:.4f} gap={row['gap']:+.3f} "
              f"ep={row['best_epoch']:<3} {row['secs']:.0f}s", flush=True)

    best = max(results, key=lambda r: r["val_auprc"])
    print(f"\n{args.isoform} done in {(time.time()-started)/60:.1f} min")
    print(f"  best by val_auprc: hidden={best['hidden']} dropout={best['dropout']} "
          f"weight_decay={best['weight_decay']}")
    print(f"  -> val_auprc {best['val_auprc']:.4f} | test_auprc {best['test_auprc']:.4f} "
          f"| test_auroc {best['test_auroc']:.4f} | gap {best['gap']:+.3f}")
    print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
