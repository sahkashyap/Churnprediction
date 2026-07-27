"""
Customer Churn Prediction Dashboard
Run with: streamlit run app.py
Expects the following files (all produced by Churn_Analysis.ipynb) in the same folder:
  model.pkl, rf_model.pkl, lr_model.pkl, scaler.pkl, label_encoders.pkl,
  feature_columns.pkl, metadata.pkl, test_results.csv,
  confusion_matrix.png, roc_pr_curves.png, feature_importance.png, shap_summary.png
"""

import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------------------------------
# Load artifacts (cached so they load once, not on every interaction)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        best_model = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    test_results = pd.read_csv("test_results.csv")
    return best_model, rf_model, lr_model, scaler, label_encoders, feature_columns, metadata, test_results


try:
    (best_model, rf_model, lr_model, scaler, label_encoders,
     feature_columns, metadata, test_results) = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. Run every cell in **Churn_Analysis.ipynb** first — "
        "it generates all the files this app needs."
    )
    st.stop()

best_model_name = metadata["best_model_name"]
best_uses_scaling = metadata["best_uses_scaling"]
comparison_df = metadata["comparison_table"]

CATEGORICAL_COLS = list(label_encoders.keys())


def safe_label_transform(le, value):
    """Transform a single value, falling back to the first known class if unseen."""
    try:
        return int(le.transform([value])[0])
    except ValueError:
        return int(le.transform([le.classes_[0]])[0])


def safe_series_transform(le, series):
    known = set(le.classes_)
    fallback = le.classes_[0]
    cleaned = series.apply(lambda v: v if v in known else fallback)
    return le.transform(cleaned)


def build_input_row(raw_values: dict) -> pd.DataFrame:
    """raw_values: {column: value as entered in the UI (human-readable)}"""
    row = {}
    for col in feature_columns:
        val = raw_values[col]
        if col in label_encoders:
            row[col] = safe_label_transform(label_encoders[col], val)
        else:
            row[col] = val
    return pd.DataFrame([row])[feature_columns]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("📊 Customer Churn Predictor")
st.sidebar.markdown(f"""
### About
Predicts whether a customer is likely to churn.

**Best model:** {best_model_name}
**Dataset:** Telco Customer Churn

Models are compared automatically — see the *Model Performance* tab.
""")
st.sidebar.markdown("---")
st.sidebar.dataframe(comparison_df.style.format("{:.3f}"))

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Enter customer information, upload a batch file, or review model performance below.")

tab_predict, tab_batch, tab_perf, tab_about = st.tabs(
    ["🔮 Prediction", "📁 Batch Prediction", "📈 Model Performance", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# TAB 1 — Single Prediction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Customer Information")
    st.caption("All fields below map to the exact features the model was trained on.")

    raw_values = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_columns):
        target_col = cols[i % 3]
        with target_col:
            if col in label_encoders:
                options = list(label_encoders[col].classes_)
                raw_values[col] = st.selectbox(col, options, key=f"in_{col}")
            elif col == "SeniorCitizen":
                choice = st.selectbox("SeniorCitizen", ["No", "Yes"], key="in_senior")
                raw_values[col] = 1 if choice == "Yes" else 0
            elif col == "tenure":
                raw_values[col] = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, key="in_tenure")
            elif col == "MonthlyCharges":
                raw_values[col] = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0, key="in_mc")
            elif col == "TotalCharges":
                raw_values[col] = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0, key="in_tc")
            else:
                raw_values[col] = st.number_input(col, value=0.0, key=f"in_num_{col}")

    st.divider()

    if st.button("🔮 Predict Customer Churn", use_container_width=True, type="primary"):
        input_df = build_input_row(raw_values)

        if best_uses_scaling:
            model_input = scaler.transform(input_df)
        else:
            model_input = input_df

        prediction = best_model.predict(model_input)[0]
        probability = best_model.predict_proba(model_input)[0][1]

        c1, c2 = st.columns(2)
        with c1:
            if prediction == 1:
                st.error("⚠️ High Risk of Churn")
            else:
                st.success("✅ Low Risk of Churn")
        with c2:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
        st.progress(min(max(probability, 0.0), 1.0))

        st.divider()
        st.subheader("Recommendation")
        if prediction == 1:
            st.warning(
                "- Offer loyalty discounts\n"
                "- Suggest upgrading to a longer contract\n"
                "- Proactively reach out via support\n"
                "- Flag account for retention team follow-up"
            )
        else:
            st.success("Customer is likely to remain. Continue providing good service.")

        # ---- SHAP explanation for this single prediction ----
        st.divider()
        st.subheader("🔍 Why did the model predict this?")
        try:
            import shap
            explainer = shap.TreeExplainer(rf_model)
            rf_input = input_df  # SHAP explanation uses the tree model regardless of which model is "best"
            explanation = explainer(rf_input)
            # explanation.values shape: (1, n_features, 2) for binary RF — take churn class (index 1)
            vals = explanation.values[0]
            if vals.ndim == 2:
                vals = vals[:, 1]
            contrib = pd.Series(vals, index=feature_columns).sort_values()

            fig, ax = plt.subplots(figsize=(7, 6))
            colors = ["#EF4444" if v > 0 else "#3B82F6" for v in contrib.values]
            ax.barh(contrib.index, contrib.values, color=colors)
            ax.set_xlabel("Impact on churn prediction (red = pushes toward churn)")
            ax.set_title("SHAP contribution — this customer")
            plt.tight_layout()
            st.pyplot(fig)
        except ImportError:
            st.info("Install `shap` (`pip install shap`) to see per-prediction explanations here.")
        except Exception as e:
            st.info(f"SHAP explanation unavailable for this input: {e}")

# --------------------------------------------------------------------------
# TAB 2 — Batch Prediction
# --------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV for bulk predictions")
    st.caption(
        "CSV should have the same raw columns as the training data "
        "(customerID and Churn columns, if present, are ignored automatically)."
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        original = batch_df.copy()

        for drop_col in ["customerID", "Churn"]:
            if drop_col in batch_df.columns:
                batch_df.drop(columns=[drop_col], inplace=True)

        if "TotalCharges" in batch_df.columns:
            batch_df["TotalCharges"] = pd.to_numeric(batch_df["TotalCharges"], errors="coerce")
            batch_df["TotalCharges"].fillna(batch_df["TotalCharges"].median(), inplace=True)

        missing_cols = [c for c in feature_columns if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {missing_cols}")
        else:
            for col in CATEGORICAL_COLS:
                if col in batch_df.columns:
                    batch_df[col] = safe_series_transform(label_encoders[col], batch_df[col].astype(str))

            batch_df = batch_df[feature_columns]
            model_input = scaler.transform(batch_df) if best_uses_scaling else batch_df

            preds = best_model.predict(model_input)
            probas = best_model.predict_proba(model_input)[:, 1]

            result = original.copy()
            result["Churn_Prediction"] = np.where(preds == 1, "Yes", "No")
            result["Churn_Probability"] = (probas * 100).round(2)

            st.success(f"Predicted {len(result)} rows.")
            st.dataframe(result, use_container_width=True)

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=csv_bytes,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

# --------------------------------------------------------------------------
# TAB 3 — Model Performance
# --------------------------------------------------------------------------
with tab_perf:
    st.subheader("📈 Model Comparison")
    st.dataframe(comparison_df.style.format("{:.3f}").highlight_max(axis=0, color="#DCFCE7"))

    st.divider()
    st.subheader("Confusion Matrices")
    if Path("confusion_matrix.png").exists():
        st.image("confusion_matrix.png", use_container_width=True)

    st.subheader("ROC & Precision-Recall Curves")
    if Path("roc_pr_curves.png").exists():
        st.image("roc_pr_curves.png", use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    if Path("feature_importance.png").exists():
        st.image("feature_importance.png", use_container_width=True)

    st.subheader("Global SHAP Feature Impact")
    if Path("shap_summary.png").exists():
        st.image("shap_summary.png", use_container_width=True)

# --------------------------------------------------------------------------
# TAB 4 — About
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
This dashboard predicts customer churn for a telecom provider using the **Telco Customer Churn** dataset.

**Pipeline:**
1. Data cleaning (missing values, duplicates, type conversion)
2. Label encoding — encoders saved and reused here for consistency with training
3. Two models trained and compared: Random Forest and Logistic Regression
4. Best model selected automatically by ROC-AUC: **{best_model_name}**
5. Evaluated with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, PR curve
6. Explained with feature importance and SHAP values

**Stack:** Python, pandas, scikit-learn, SHAP, Streamlit
""")

st.divider()
st.caption("Customer Churn Prediction System | Data Science Project")
