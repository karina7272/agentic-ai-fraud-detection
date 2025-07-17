
# Agentic AI Fraud Detection Application (Revised with Rule-Based Explanations)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

# Streamlit UI config
st.set_page_config(page_title="Agentic AI Fraud Detection", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stApp {background-color: #eaf4fb; padding-top:70px;}
.header-logo {position: fixed; top:10px; left:10px; z-index:999;}
.css-18e3th9 {background-color: rgba(255,255,255,0.95); padding:1rem; border-radius:0.5rem;}
h1, h2, h3 {color:#2c3e50;}
</style>
<div class="header-logo">
    <img src="https://cdn-icons-png.flaticon.com/512/3379/3379050.png" width="50"/>
</div>
""", unsafe_allow_html=True)

st.title("Agentic AI Fraud Detection Application")
st.markdown("**Empowering Accountants with Explainable AI for Fraud Detection**")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Sample")
    st.dataframe(df.head())

    # Step 1: Preprocessing
    st.subheader("Step 1: Perception Module - Data Preprocessing")
    df["AmountBucket"] = pd.cut(df["Amount"], bins=[0,5000,20000,np.inf], labels=["Low (<$5k)","Medium ($5k–$20k)","High (>$20k)"])
    df["VendorCode"] = df["VendorCategory"].astype("category").cat.codes
    df["Prior_Vendor"] = df["PriorFlag"] * df["VendorCode"]
    df = pd.get_dummies(df, columns=["AmountBucket"], drop_first=False)

    X = df[["Amount","DayOfMonth","PriorFlag","VendorCode","Prior_Vendor"] + [c for c in df.columns if "AmountBucket_" in c]]
    y = df["IsFraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 2: Multi-Model Training
    st.subheader("Step 2: Policy Learning - Multi-Model Training")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=100)
    }

    models["Logistic Regression"].fit(X_train_scaled, y_train)
    models["Random Forest"].fit(X_train, y_train)
    models["XGBoost"].fit(X_train.values, y_train)
    models["Neural Network"].fit(X_train_scaled, y_train)

    # SHAP Explanation for Random Forest
    st.subheader("Step 3: Execution Module - Interpretability")
    st.write("Random Forest Feature Importances")
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots()
    importances.plot(kind="barh", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.write("SHAP Summary Plot")
    explainer = shap.TreeExplainer(rf)
    sample_X = shap.utils.sample(X_test, 100)  # reduce to 100 rows to prevent crash
    shap_values = explainer.shap_values(sample_X)
    fig_shap = plt.figure()
    shap.summary_plot(shap_values[1], sample_X, show=False)
    st.pyplot(fig_shap)
    plt.close(fig_shap)

    # Step 4: Agentic AI Policy Table
    st.subheader("Step 4: Planning Module - Q-Learning Policy Table")
    policy_table = pd.DataFrame([
        [0, "Low (<$5k)", +0.65, -0.15, -0.45],
        [0, "Medium ($5k–$20k)", +0.30, +0.10, -0.10],
        [0, "High (>$20k)", -0.20, +0.45, +0.80],
        [1, "Low (<$5k)", -0.10, +0.40, +0.50],
        [1, "Medium ($5k–$20k)", -0.30, +0.55, +0.70],
        [1, "High (>$20k)", -0.60, +0.70, +1.00]
    ], columns=["PriorFlag","AmountBucket","Approve","Flag","Reject"])
    st.dataframe(policy_table)

    # Step 5: Rule-Based Interpretation
    st.subheader("Step 5: Human-Readable Explanations")
    def interpret_policy(prior_flag, amount):
        if prior_flag == 0:
            if amount < 5000:
                return "No prior flags and low amount → Approve likely."
            elif amount < 20000:
                return "No prior flags, moderate amount → Mixed signals."
            else:
                return "No prior flags, high amount → Consider Flag or Reject."
        else:
            if amount < 5000:
                return "Prior fraud history and low amount → Flag advised."
            elif amount < 20000:
                return "Prior fraud history and moderate amount → Likely Flag or Reject."
            else:
                return "Prior fraud and high amount → Strongly recommend Reject."

    sample_cases = df.sample(5, random_state=0)[["Amount","PriorFlag"]]
    sample_cases["Explanation"] = sample_cases.apply(lambda row: interpret_policy(row["PriorFlag"], row["Amount"]), axis=1)
    st.dataframe(sample_cases)

else:
    st.info("Please upload a CSV file with the required format.")

st.markdown("---")
st.caption("© 2024 Agentic AI Research | Enhanced Interpretability & Decision Support")
