#!/usr/bin/env python
"""Summarise one scored CSV: print a report and save it as a one-row CSV.

    python report.py Generated_molecules/myrun_scored.csv
    python report.py myrun_scored.csv --out Generated_molecules/myrun_report.csv

Input is a score_all.py output. With no --out the report lands in
Generated_molecules/<input stem>_report.csv, next to this script.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "Generated_molecules"

SEH_THRESHOLD = 0.875  # a molecule counts as a binder above this
INHIBIT_THRESHOLD = 0.5  # proxy probability above this counts as inhibition
ISOFORMS = ["cyp1a2", "cyp2c19", "cyp2c9", "cyp2d6", "cyp3a4"]

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def diversity(smiles):
    """Mean pairwise 1 - Tanimoto over Morgan fingerprints (exact, no subsample)."""
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            fps.append(np.array(_MORGAN.GetFingerprint(mol), dtype=bool))
    if len(fps) < 2:
        return float("nan")
    return float(np.mean(pdist(np.array(fps), metric="jaccard")))


def scaffolds_of(uniq):
    """The scaffold column if present, otherwise computed from the SMILES."""
    if "scaffold" in uniq.columns:
        return uniq["scaffold"].astype(str)
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out = []
    for s in uniq["smi"]:
        mol = Chem.MolFromSmiles(str(s))
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else "")
        except Exception:
            out.append("")
    return pd.Series(out, index=uniq.index, dtype=str)


def summarise(path):
    """Return an ordered dict of metrics for one scored CSV."""
    df = pd.read_csv(path)
    if "smi" not in df.columns:
        raise SystemExit(f"{path}: no 'smi' column (found: {', '.join(df.columns)})")

    # plain-sEH runs have no separate column: the reward IS the sEH score
    if "seh" not in df.columns and "reward" in df.columns:
        df["seh"] = df["reward"]

    df = df.dropna(subset=["smi"])
    present = [i for i in ISOFORMS if i in df.columns]
    if present:
        df = df.dropna(subset=present)
    uniq = df.drop_duplicates("smi")
    if len(uniq) == 0:
        raise SystemExit(f"{path}: no usable rows after dropping blanks")

    scaf = scaffolds_of(uniq)
    nan = float("nan")

    has_seh = "seh" in uniq.columns
    hot = uniq["seh"] > SEH_THRESHOLD if has_seh else None

    if "n_inhibited" in uniq.columns:
        n_inh = uniq["n_inhibited"]
    elif present:
        n_inh = sum((uniq[i] > INHIBIT_THRESHOLD).astype(int) for i in present)
    else:
        n_inh = None
    clean = n_inh == 0 if n_inh is not None else None

    row = {
        "run": Path(path).stem.replace("_scored", ""),
        "samples": len(df),
        "unique": len(uniq),
        "scaffolds": scaf.nunique(),
        "largest_scaffold": scaf.value_counts().iloc[0] / len(uniq),
        "diversity": diversity(uniq["smi"]),
        "seh": uniq["seh"].mean() if has_seh else nan,
        f"seh_gt_{SEH_THRESHOLD}": hot.mean() if has_seh else nan,
    }
    for iso in ISOFORMS:
        row[f"inh_{iso}"] = (uniq[iso] > INHIBIT_THRESHOLD).mean() if iso in uniq.columns else nan
    row["mean_isoforms_inhibited"] = n_inh.mean() if n_inh is not None else nan
    row["clean"] = clean.mean() if clean is not None else nan
    row["useful"] = int((hot & clean).sum()) if (has_seh and clean is not None) else nan
    for c in ("sa", "qed", "logp", "lipinski"):
        row[c] = uniq[c].mean() if c in uniq.columns else nan
    return row


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def num(x, places=4):
    return "n/a" if pd.isna(x) else f"{x:.{places}f}"


def print_report(row, path):
    lines = [
        ("samples", f"{row['samples']}"),
        ("unique", f"{row['unique']}"),
        ("scaffolds", f"{row['scaffolds']}"),
        ("largest scaffold", pct(row["largest_scaffold"])),
        ("diversity (1-Tanimoto)", num(row["diversity"])),
        ("", ""),
        ("mean sEH", num(row["seh"])),
        (f"sEH > {SEH_THRESHOLD}", pct(row[f"seh_gt_{SEH_THRESHOLD}"])),
        ("", ""),
        ("inhibits CYP1A2", pct(row["inh_cyp1a2"])),
        ("inhibits CYP2C19", pct(row["inh_cyp2c19"])),
        ("inhibits CYP2C9", pct(row["inh_cyp2c9"])),
        ("inhibits CYP2D6", pct(row["inh_cyp2d6"])),
        ("inhibits CYP3A4", pct(row["inh_cyp3a4"])),
        ("mean isoforms inhibited", num(row["mean_isoforms_inhibited"], 2) + " / 5"),
        ("clean 0/5", pct(row["clean"])),
        ("useful (hot & clean)", "n/a" if pd.isna(row["useful"]) else f"{int(row['useful'])}"),
        ("", ""),
        ("SA", num(row["sa"], 3)),
        ("QED", num(row["qed"], 3)),
        ("logP", num(row["logp"], 3)),
        ("Lipinski pass", num(row["lipinski"], 3)),
    ]
    width = 62
    print("=" * width)
    print(f"  {row['run']}")
    print(f"  {path}")
    print("=" * width)
    print("  all metrics except `samples` are over unique canonical SMILES")
    print("-" * width)
    for label, value in lines:
        print("" if not label else f"  {label:<28} {value:>12}")
    print("=" * width)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", type=Path, help="one scored CSV to summarise")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"where to write the one-row report (default: {OUT_DIR.name}/<stem>_report.csv)",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"no such file: {args.csv}")

    row = summarise(args.csv)
    print_report(row, args.csv)

    out = args.out or OUT_DIR / f"{args.csv.stem.replace('_scored', '')}_report.csv"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out, index=False)
        print(f"\nwrote {out}")
    except OSError as e:
        print(f"\ncould not write {out}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
