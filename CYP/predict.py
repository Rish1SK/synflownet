import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from model import load_checkpoint, mol_to_graph

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
ISOFORMS = ["cyp1a2", "cyp2c19", "cyp2c9", "cyp2d6", "cyp3a4"]


class CypPredictor:

    def __init__(self, model_paths=None, isoforms=None, device="cpu", model_dir=MODEL_DIR):
        if model_paths is None:
            model_paths = {iso: Path(model_dir) / f"{iso}_gcn.pt" for iso in (isoforms or ISOFORMS)}
        missing = [str(p) for p in model_paths.values() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError("no checkpoint at: " + ", ".join(missing))

        self.device = torch.device(device)
        self.models, self.metrics = {}, {}
        for iso, path in model_paths.items():
            model, ckpt = load_checkpoint(path, map_location=self.device)
            self.models[iso] = model.to(self.device)  
            self.metrics[iso] = ckpt.get("metrics", {})

    @torch.no_grad()
    def predict(self, smiles, batch_size=256):
        if isinstance(smiles, str):
            smiles = [smiles]

        graphs, keep = [], []
        for i, smi in enumerate(smiles):
            graph = mol_to_graph(smi)
            if graph is not None:
                graphs.append(graph)
                keep.append(i)

        out = {iso: np.full(len(smiles), np.nan, dtype=np.float32) for iso in self.models}
        if not graphs:
            return out

        chunks = {iso: [] for iso in self.models}
        for batch in DataLoader(graphs, batch_size=batch_size):
            batch = batch.to(self.device)
            for iso, model in self.models.items():
                logits = model(batch.x, batch.edge_index, batch.batch)
                chunks[iso].append(torch.sigmoid(logits).cpu())

        for iso in self.models:
            out[iso][keep] = torch.cat(chunks[iso]).numpy()
        return out

    def predict_one(self, smiles):
        return {iso: float(v[0]) for iso, v in self.predict([smiles]).items()}


def main():
    ap = argparse.ArgumentParser(description="Predict CYP inhibition for SMILES")
    ap.add_argument("smiles", nargs="*", help="SMILES strings to score")
    ap.add_argument("--file", type=Path, help="file with one SMILES per line")
    ap.add_argument("--isoforms", nargs="+", default=ISOFORMS, choices=ISOFORMS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    smiles = list(args.smiles)
    if args.file:
        smiles += [ln.strip() for ln in args.file.read_text().splitlines() if ln.strip()]
    if not smiles:
        raise SystemExit("give SMILES as arguments or via --file")

    pred = CypPredictor(isoforms=args.isoforms, device=args.device)
    probs = pred.predict(smiles, batch_size=args.batch_size)

    header = "  ".join(f"{iso:>8}" for iso in args.isoforms)
    print(f"{'smiles':<48}  {header}")
    for i, smi in enumerate(smiles):
        row = "  ".join(
            "     n/a" if np.isnan(probs[iso][i]) else f"{probs[iso][i]:>8.3f}" for iso in args.isoforms
        )
        label = smi if len(smi) <= 46 else smi[:45] + "…"
        print(f"{label:<48}  {row}")


if __name__ == "__main__":
    main()
