import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles, frac_train=0.8, frac_valid=0.1, seed=42):
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[get_scaffold(s)].append(i)

    rng = np.random.RandomState(seed)
    sets = list(groups.values())
    big = [g for g in sets if len(g) > len(smiles) * frac_valid / 2]
    small = [g for g in sets if len(g) <= len(smiles) * frac_valid / 2]
    rng.shuffle(small)
    ordered = sorted(big, key=len, reverse=True) + small

    n = len(smiles)
    train, valid, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= frac_train * n:
            train += g
        elif len(valid) + len(g) <= frac_valid * n:
            valid += g
        else:
            test += g
    return np.array(train), np.array(valid), np.array(test)


def assert_no_scaffold_leak(smiles, tr, va, te):
    sc = lambda idx: {get_scaffold(smiles[i]) for i in idx}
    assert not (sc(tr) & sc(va)) and not (sc(tr) & sc(te)) and not (sc(va) & sc(te)), "Scaffold leakage!"
    print("✓ No scaffold overlap between train/valid/test")
