import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator

N_BITS = 1024
RADIUS = 2
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

_BASIC_N = Chem.MolFromSmarts(
    "[NX3;!$(N-C=[O,S,N]);!$(N-S(=O)=O);!$(N-a);!$(N-[#7,#8]);!$(N-C#N)]"
)

DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "RotBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "AromRings": rdMolDescriptors.CalcNumAromaticRings,
    "FracCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
    "BasicN": lambda m: len(m.GetSubstructMatches(_BASIC_N)),
}
DESC_NAMES = list(DESCRIPTORS)
FEATURE_NAMES = DESC_NAMES + [f"bit_{i}" for i in range(N_BITS)]


def featurize(mol, return_bitinfo: bool = False):
    desc = np.array([f(mol) for f in DESCRIPTORS.values()], dtype=np.float32)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = _gen.GetFingerprintAsNumPy(mol, additionalOutput=ao).astype(np.float32)
    x = np.concatenate([desc, fp])
    return (x, ao.GetBitInfoMap()) if return_bitinfo else x


def featurize_smiles(smiles_list):
    rows, valid = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            valid.append(False)
            continue
        rows.append(featurize(mol))
        valid.append(True)
    return np.vstack(rows), np.array(valid)
