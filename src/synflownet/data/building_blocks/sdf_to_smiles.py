import argparse
from pathlib import Path

from rdkit import Chem, RDLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an Enamine building blocks SDF file to one SMILES per line")
    parser.add_argument("--sdf_path", type=str, help="Path to Enamine building blocks SDF file", required=True)
    parser.add_argument(
        "--output_filename", type=str, help="Path to output building blocks", default="building_blocks.txt"
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    n_ok, n_failed = 0, 0
    with open(args.sdf_path, "rb") as sdf, open(Path(__file__).parent / args.output_filename, "w") as file:
        for mol in Chem.ForwardSDMolSupplier(sdf):
            if mol is None:
                n_failed += 1
                continue
            file.write(Chem.MolToSmiles(mol) + "\n")
            n_ok += 1
            if n_ok % 100000 == 0:
                print(f"{n_ok} molecules converted")

    print(f"Done: {n_ok} molecules written, {n_failed} could not be parsed")
