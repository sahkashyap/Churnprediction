# Customer Churn Prediction

Predicts whether a telecom customer is likely to churn, using the Telco Customer
Churn dataset. Includes a full analysis notebook and an interactive Streamlit
dashboard for single predictions, batch predictions, and model explainability.

## Problem Statement

Customer churn (customers leaving a subscription service) directly hits revenue.
This project builds a classifier that flags at-risk customers early enough for a
retention team to act, and explains *why* each customer is flagged — not just a
yes/no.

## What's inside

| File | Purpose |
|---|---|
| `Churn_Analysis.ipynb` | Data cleaning, EDA, encoding, training, evaluation, SHAP, saves all model artifacts |
| `app.py` | Streamlit dashboard — single prediction, batch CSV prediction, model comparison, SHAP explanations |
| `requirements.txt` | Python dependencies |
| `customer-churn.csv` | Dataset (not included here — see below) |

## Approach

1. **Cleaning** — fixed `TotalCharges` type issue, removed duplicates, dropped the
   `customerID` identifier column (it carries no predictive signal).
2. **Encoding** — categorical columns label-encoded; encoders are saved to disk so
   the app transforms user input identically to how the model was trained
   (earlier drafts hardcoded these mappings, which silently produced wrong
   predictions when the hardcoded order didn't match the encoder's actual order).
3. **Modeling** — trained both Random Forest and Logistic Regression, compared on
   Accuracy, Precision, Recall, F1, and ROC-AUC. The notebook picks the stronger
   model automatically and the app always tells you which one is active.
4. **Evaluation** — confusion matrix, ROC curve, precision-recall curve (all
   important here since churn is an imbalanced class).
5. **Explainability** — global feature importance (Random Forest) plus SHAP values
   for both the overall model and each individual prediction the app makes.

## Results

See the *Model Performance* tab in the app, or the comparison table at the end of
the notebook, for exact numbers on your data run.

## Running locally

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Run every cell in Churn_Analysis.ipynb (Jupyter or Jupyter Lab)
#    — this trains the models and saves model.pkl, scaler.pkl, etc.

# 3. Launch the dashboard
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud: **[add your live link here]**

## Tech stack

Python · pandas · scikit-learn · SHAP · Streamlit · seaborn/matplotlib
