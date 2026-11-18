import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

RDLogger.DisableLog("rdApp.*")

FEATURE_VERSION = "cyp-atom-v1"


ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Se", "Na"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

NUM_FEATURES = (
    len(ATOM_SYMBOLS) + 1 + len(DEGREES) + len(CHARGES) + len(NUM_HS) + len(HYBRIDIZATIONS) + 1 + 2
)


def _one_hot(value, choices, allow_other=True):
    vec = [0.0] * (len(choices) + (1 if allow_other else 0))
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        if allow_other:
            vec[-1] = 1.0
    return vec


def atom_features(atom):
    return (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        + _one_hot(atom.GetDegree(), DEGREES, allow_other=False)
        + _one_hot(atom.GetFormalCharge(), CHARGES, allow_other=False)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS, allow_other=False)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def largest_fragment(mol):
    """Strip counterions: ~4% of the Veith molecules are salts."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda m: m.GetNumHeavyAtoms()) if len(frags) > 1 else mol


def mol_to_graph(smiles, y=None, strip_salts=True):
    """SMILES -> PyG Data, or None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if strip_salts:
        mol = largest_fragment(mol)
    if mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, smiles=smiles)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float)
    return data


class CypGCN(nn.Module):
    """GCN -> BatchNorm -> ReLU -> Dropout, repeated, then mean pooling."""

    def __init__(self, in_dim=NUM_FEATURES, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(torch.relu(norm(conv(x, edge_index))))
        return self.head(global_mean_pool(x, batch)).squeeze(-1)  # logits


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "ndim", 0) == 0:
        return obj.item()
    if obj is None or type(obj) in (str, bool, int, float):
        return obj
    for base in (bool, int, float, str):
        if isinstance(obj, base):
            return base(obj)
    return str(obj)


def save_checkpoint(model, path, config, **meta):
    torch.save(
        {
            "state_dict": {k: v.detach().cpu().float() for k, v in model.state_dict().items()},
            "config": _plain(config),
            "featurizer": FEATURE_VERSION,
            "torch_version": str(torch.__version__),
            **_plain(meta),
        },
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if ckpt.get("featurizer") != FEATURE_VERSION:
        raise ValueError(
            f"Checkpoint was trained with featurizer {ckpt.get('featurizer')!r} "
            f"but this module provides {FEATURE_VERSION!r}. Scores would be meaningless."
        )
    model = CypGCN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
