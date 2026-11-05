import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = _chooser.choose(mol)
        mol = _uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_raw() -> pd.DataFrame:
    from tdc.single_pred import Tox
    df = Tox(name="hERG_Karim").get_data()
    return df.rename(columns={"Drug": "smiles", "Y": "label"})[["smiles", "label"]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.copy()
    df["smiles"] = df["smiles"].map(standardize_smiles)
    df = df.dropna(subset=["smiles"])
    conflict = df.groupby("smiles")["label"].nunique()
    conflict = conflict[conflict > 1].index
    df = df[~df["smiles"].isin(conflict)].drop_duplicates("smiles").reset_index(drop=True)
    print(f"Cleaned: {n0} -> {len(df)} molecules ({len(conflict)} label-conflicts removed)")
    return df


if __name__ == "__main__":
    df = clean(load_raw())
    print(df["label"].value_counts())
    df.to_csv("data/herg_clean.csv", index=False)
