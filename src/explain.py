import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from src.features import featurize, DESC_NAMES

N_DESC = len(DESC_NAMES)


def explain_molecule(bundle, smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    model, thr, names = bundle["model"], bundle["threshold"], bundle["feature_names"]
    x, bitinfo = featurize(mol, return_bitinfo=True)
    prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(x.reshape(1, -1))[0]
    return {"mol": mol, "prob": prob, "pred": int(prob >= thr), "shap": sv,
            "x": x, "bitinfo": bitinfo, "names": names, "threshold": thr}


def top_features(res, k=10):
    idx = np.argsort(-np.abs(res["shap"]))[:k]
    return [(res["names"][i], float(res["x"][i]), float(res["shap"][i])) for i in idx]


def highlight_image(res, top_bits=5, size=(450, 350)):
    shap_bits = res["shap"][N_DESC:]
    present = list(res["bitinfo"])
    ranked = sorted(present, key=lambda b: -shap_bits[b])[:top_bits]
    ranked = [b for b in ranked if shap_bits[b] > 0]

    mol = res["mol"]
    atoms, bonds = set(), set()
    for b in ranked:
        for atom_idx, radius in res["bitinfo"][b]:
            if radius == 0:
                atoms.add(atom_idx)
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            amap = {}
            Chem.PathToSubmol(mol, env, atomMap=amap)
            atoms.update(amap.keys())
            bonds.update(env)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(atoms),
                                       highlightBonds=list(bonds))
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def global_shap_plots(bundle, X_test, out_dir="reports/figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    explainer = shap.TreeExplainer(bundle["model"])
    sv = explainer.shap_values(X_test)
    shap.summary_plot(sv, X_test, feature_names=bundle["feature_names"], max_display=20, show=False)
    plt.savefig(f"{out_dir}/shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
    shap.summary_plot(sv, X_test, feature_names=bundle["feature_names"], plot_type="bar",
                      max_display=20, show=False)
    plt.savefig(f"{out_dir}/shap_bar.png", dpi=200, bbox_inches="tight"); plt.close()
    return sv
