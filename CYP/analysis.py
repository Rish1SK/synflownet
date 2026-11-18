import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

CSV_DIR = Path(__file__).parent / "csv"


def load(path):
    df = pd.read_csv(path, sep="\t" if path.suffix == ".tab" else ",")
    return df.rename(columns={"Drug_ID": "drug_id", "Drug": "smiles", "Y": "y"})


def report_duplicates(df):
    n = len(df)
    dup_rows = n - len(df.drop_duplicates())
    dup_smiles = n - df["smiles"].nunique()
    dup_pairs = n - len(df.drop_duplicates(subset=["smiles", "y"]))
    conflicts = (df.groupby("smiles")["y"].nunique() > 1).sum()

    print("  duplicates")
    print(f"    identical rows            {dup_rows}")
    print(f"    duplicate smiles          {dup_smiles}")
    print(f"    duplicate (smiles, y)     {dup_pairs}")
    print(f"    smiles with 2 labels      {conflicts}  <- same molecule, contradictory y")
    if "drug_id" in df:
        print(f"    duplicate drug_id         {n - df['drug_id'].nunique()}")


def report_nulls(df):
    nulls = df.isna().sum()
    print("  nulls")
    if nulls.sum() == 0:
        print("    none in any column")
    else:
        for col, k in nulls[nulls > 0].items():
            print(f"    {col:<24}  {k} ({k / len(df):.2%})")
    blank = (df["smiles"].astype(str).str.strip() == "").sum()
    if blank:
        print(f"    empty smiles strings      {blank}")


def report_balance(df):
    counts = df["y"].value_counts().sort_index()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    print("  class balance")
    print(f"    negatives (y=0)           {neg} ({neg / len(df):.1%})")
    print(f"    positives (y=1)           {pos} ({pos / len(df):.1%})")
    if pos and neg:
        print(f"    majority:minority ratio   {max(pos, neg) / min(pos, neg):.2f} : 1")
    other = set(df["y"].unique()) - {0, 1}
    if other:
        print(f"    unexpected label values   {sorted(other)}")


def report_chemistry(df):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    invalid = sum(m is None for m in mols)
    sizes = pd.Series([m.GetNumHeavyAtoms() for m in mols if m is not None])
    mixtures = df["smiles"].str.contains(r"\.", regex=True).sum()
    canon = {Chem.MolToSmiles(m) for m in mols if m is not None}

    print("  chemistry")
    print(f"    unparseable by rdkit      {invalid}")
    print(f"    multi-component (salts)   {mixtures} ({mixtures / len(df):.1%})")
    print(f"    heavy atoms               min {sizes.min()}, median {int(sizes.median())}, max {sizes.max()}")
    print(f"    unique after canonical    {len(canon)} of {df['smiles'].nunique()} raw-unique")
    return canon


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic EDA on the CYP datasets")
    parser.add_argument("--data_dir", type=Path, help="Directory of .csv or .tab files", default=CSV_DIR)
    parser.add_argument("--no_rdkit", action="store_true", help="Skip the RDKit validity checks")
    args = parser.parse_args()

    paths = sorted(p for p in args.data_dir.iterdir() if p.suffix in {".csv", ".tab"} and p.stem != "all_cyp")
    if not paths:
        raise SystemExit(f"No .csv or .tab files found in {args.data_dir}")

    canon_sets = {}
    for path in paths:
        df = load(path)
        print(f"\n{path.name}  ({len(df)} rows, columns: {', '.join(df.columns)})")
        report_nulls(df)
        report_duplicates(df)
        report_balance(df)
        if not args.no_rdkit:
            canon_sets[path.stem] = report_chemistry(df)

    if len(canon_sets) > 1:
        print("\noverlap between datasets (shared canonical smiles)")
        for a, b in combinations(canon_sets, 2):
            shared = len(canon_sets[a] & canon_sets[b])
            print(f"  {a:<16} vs {b:<16} {shared} ({shared / min(len(canon_sets[a]), len(canon_sets[b])):.1%})")
        union = set().union(*canon_sets.values())
        print(f"  unique molecules across all datasets: {len(union)}")
