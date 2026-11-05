import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.explain import explain_molecule, top_features, highlight_image

st.set_page_config(page_title="CardioGuard", page_icon="🫀", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("models/cardioguard_xgb.joblib")


bundle = load_bundle()

st.title("🫀 CardioGuard: hERG Cardiotoxicity Predictor")
st.markdown(
    "Predicts whether a small molecule is likely to **block the hERG potassium channel**, "
    "a major cause of drug-induced arrhythmia (QT prolongation). "
    "Model: XGBoost on Morgan fingerprints and physicochemical descriptors, validated with a "
    "**scaffold split**. Explanations come from **SHAP**."
)

examples = {
    "Terfenadine (withdrawn, hERG blocker)": "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
    "Fexofenadine (safer metabolite)": "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}
choice = st.selectbox("Load an example", ["(custom)"] + list(examples))
default = examples.get(choice, "")
smiles = st.text_input("Enter a SMILES string", value=default)

if smiles:
    res = explain_molecule(bundle, smiles.strip())
    if res is None:
        st.error("Invalid SMILES. Please check the input.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Prediction")
            st.metric("hERG blocker probability", f"{res['prob']:.1%}")
            if res["pred"]:
                st.error("⚠️ Predicted hERG BLOCKER: potential cardiotoxicity risk")
            else:
                st.success("✅ Predicted NON-blocker")
            st.caption(f"Decision threshold: {res['threshold']:.2f} (optimized on validation MCC)")
            st.image(highlight_image(res), caption="Highlighted: substructures pushing toward 'blocker'")
        with c2:
            st.subheader("Why? Top SHAP features")
            tf = top_features(res, 10)
            df = pd.DataFrame(tf, columns=["Feature", "Value", "SHAP"])
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["SHAP"]]
            ax.barh(df["Feature"][::-1], df["SHAP"][::-1], color=colors[::-1])
            ax.set_xlabel("SHAP value (red = toward blocker)")
            st.pyplot(fig)
            st.dataframe(df.round(3), use_container_width=True)

st.divider()
st.caption("⚠️ Research tool only. Not for clinical or regulatory decisions. "
           "Predictions are limited by the training data's chemical space.")
