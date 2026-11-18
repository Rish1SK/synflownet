import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from model import largest_fragment

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "csv"


def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = largest_fragment(mol)
    if mol.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(mol)  
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def load_tab(path):
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"Drug_ID": "drug_id", "Drug": "smiles", "Y": "y"})

    df["drug_id"] = df["drug_id"].astype("Int64")
    df["y"] = df["y"].astype(int)
    df["isoform"] = path.stem.split("_")[0].upper()

    n_raw = len(df)
    df = df.dropna(subset=["smiles", "y"])

    df["smiles"] = [standardize(s) for s in df["smiles"]]
    n_unparsed = df["smiles"].isna().sum()
    df = df.dropna(subset=["smiles"])
    conflicting = df.groupby("smiles")["y"].nunique() > 1
    conflicting = set(conflicting[conflicting].index)
    df = df[~df["smiles"].isin(conflicting)]

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="smiles", keep="first")
    n_collapsed = n_before_dedup - len(df)

    print(
        f"{path.name}: {n_raw} rows -> {len(df)} kept, {df['y'].mean():.1%} positives"
        f"  ({n_unparsed} unparseable, {len(conflicting)} conflicting molecules dropped,"
        f" {n_collapsed} duplicates collapsed)"
    )
    return df[["drug_id", "smiles", "y", "isoform"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the TDC CYP .tab files to csv")
    parser.add_argument("--data_dir", type=Path, help="Directory holding the .tab files", default=DATA_DIR)
    parser.add_argument("--out_dir", type=Path, help="Directory to write the csv files to", default=OUT_DIR)
    args = parser.parse_args()

    tab_paths = sorted(args.data_dir.glob("*.tab"))
    if not tab_paths:
        raise SystemExit(f"No .tab files found in {args.data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in tab_paths:
        df = load_tab(path)
        df.to_csv(args.out_dir / f"{path.stem}.csv", index=False)
        frames.append(df)

    all_cyp = pd.concat(frames, ignore_index=True)
    all_cyp.to_csv(args.out_dir / "all_cyp.csv", index=False)
    print(f"\nWrote {len(frames)} csv files and all_cyp.csv ({len(all_cyp)} rows) to {args.out_dir}")
