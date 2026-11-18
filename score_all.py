"""Turn a SynFlowNet generation directory into one fully scored CSV.

    python score_all.py logs/<run>/final
    python score_all.py logs/<run>/final --out Generated_molecules/product_b16.csv

Output columns, ready for report.py:

    smi reward valid qed mw logp tpsa hbd hba rotb rings heavy_atoms lipinski
    sa scaffold                                            (RDKit properties)
    seh                                                    (sEH binding proxy)
    cyp1a2 cyp2c19 cyp2c9 cyp2d6 cyp3a4 n_inhibited        (CYP GCN ensemble)
"""
import argparse
import glob
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.data as gd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "Generated_molecules"
CYP_DIR = HERE / "CYP"
INHIBIT_THRESHOLD = 0.5  # matches report.py

# SA score lives in RDKit contrib
try:
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    HAS_SA = True
except Exception:
    HAS_SA = False
    print("WARNING: sascorer unavailable, 'sa' column will be empty")


def load_molecules(db_dir):
    """Read every generated_objs_*.db in a directory into one frame."""
    dbs = sorted(glob.glob(os.path.join(db_dir, "*.db")))
    if not dbs:
        raise SystemExit(f"no .db files in {db_dir}")

    frames = []
    for db in dbs:
        con = sqlite3.connect(db)
        try:
            frames.append(pd.read_sql_query("SELECT smi, r FROM results", con))
        finally:
            con.close()

    out = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(out)} rows from {len(dbs)} database(s)")
    return out


def compute_properties(df):
    """RDKit descriptors, Lipinski, SA score and Murcko scaffold per molecule."""
    rows = []
    for smi, reward in zip(df["smi"], df["r"]):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            rows.append({"smi": smi, "reward": reward, "valid": False})
            continue

        rec = {
            "smi": Chem.MolToSmiles(mol),  # canonical
            "reward": reward,
            "valid": True,
            "qed": QED.qed(mol),
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "rotb": Descriptors.NumRotatableBonds(mol),
            "rings": Descriptors.RingCount(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
        }
        rec["lipinski"] = (
            rec["mw"] < 500 and rec["logp"] < 5 and rec["hbd"] <= 5 and rec["hba"] <= 10
        )
        if HAS_SA:
            try:
                rec["sa"] = sascorer.calculateScore(mol)
            except Exception:
                rec["sa"] = np.nan
        try:
            rec["scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            rec["scaffold"] = None
        rows.append(rec)

    out = pd.DataFrame(rows)
    print(f"valid: {int(out['valid'].sum())}/{len(out)}")
    return out


def score_seh(smiles, device, batch_size):
    """sEH proxy scores, divided by 8 exactly as the task code does."""
    from synflownet.models import bengio2021flow

    print(f"scoring {len(smiles)} molecules with the sEH proxy on {device} ...")
    model = bengio2021flow.load_original_model().to(device).eval()

    out = np.full(len(smiles), np.nan, dtype=float)
    graphs, rows = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        try:
            g = bengio2021flow.mol2graph(mol)
        except Exception:
            g = None  # atom type outside the proxy's vocabulary
        if g is not None:
            graphs.append(g)
            rows.append(i)

    if len(graphs) < len(smiles):
        print(f"  {len(smiles) - len(graphs)} molecules unparseable or out of vocabulary -> seh ~ 0")

    with torch.no_grad():
        for s in range(0, len(graphs), batch_size):
            batch = gd.Batch.from_data_list(graphs[s : s + batch_size]).to(device)
            out[rows[s : s + batch_size]] = model(batch).reshape(-1).cpu().numpy() / 8.0

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.clip(np.nan_to_num(out, nan=0.0), 1e-4, 100)


def score_cyp(df, device, batch_size):
    """Add the five isoform probabilities and n_inhibited."""
    sys.path.insert(0, str(CYP_DIR))  # CYP/predict.py is not a package
    from predict import ISOFORMS, CypPredictor

    print(f"scoring {len(df)} molecules with the CYP ensemble on {device} ...")
    probs = CypPredictor(isoforms=ISOFORMS, device=device).predict(
        df["smi"].astype(str).tolist(), batch_size=batch_size
    )

    out = df.copy()
    for iso in ISOFORMS:
        out[iso] = probs[iso]

    stacked = np.vstack([probs[iso] for iso in ISOFORMS])
    n_inhibited = (stacked > INHIBIT_THRESHOLD).sum(axis=0).astype(float)
    n_inhibited[np.isnan(stacked).any(axis=0)] = np.nan  # unreadable: not "clean"
    out["n_inhibited"] = n_inhibited

    n_bad = int(np.isnan(n_inhibited).sum())
    if n_bad:
        print(f"  {n_bad} molecules unparseable -> NaN")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("db_dir", help="directory containing generated_objs_*.db")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output CSV (default: {OUT_DIR.name}/<run directory name>_scored.csv)",
    )
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    if not os.path.isdir(args.db_dir):
        raise SystemExit(f"not a directory: {args.db_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = os.path.basename(os.path.dirname(args.db_dir.rstrip("/\\")))
    out = args.out or OUT_DIR / f"{run_name}_scored.csv"

    df = compute_properties(load_molecules(args.db_dir))
    df["seh"] = score_seh(df["smi"].tolist(), device, args.batch_size)
    df = score_cyp(df, device, args.batch_size)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows x {len(df.columns)} columns to {out}")
    print(f"next: python report.py {out}")


if __name__ == "__main__":
    main()
