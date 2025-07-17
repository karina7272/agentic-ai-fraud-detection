
# Agentic AI Fraud Detection Application

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

# Page config
st.set_page_config(
    page_title="Agentic AI Fraud Detection",
    layout="wide"
)

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

uploaded_file = st.file_uploader(
    "Upload your CSV file (Columns: EntryID, Amount, VendorCategory, DayOfMonth, PriorFlag, IsFraud)",
    type="csv"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Sample")
    st.dataframe(df.head())

    # ----------------------------
    st.subheader("Step 1: Perception Module - Data Preprocessing")
    try:
        fig, ax = plt.subplots()
        df["IsFraud"].value_counts().plot(kind="bar", color=["#3498db","#e74c3c"], ax=ax)
        ax.set_title("Fraud vs Non-Fraud Counts")
        ax.set_ylabel("Count")
        ax.set_xticklabels(["Non-Fraud","Fraud"], rotation=0)
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        for label in [0,1]:
            subset = df[df["IsFraud"]==label]["Amount"]
            ax.hist(subset, bins=50, alpha=0.5, label="Fraud" if label==1 else "Non-Fraud")
        ax.set_xlabel("Transaction Amount ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        bins = [0,1000,5000,10000,20000,50000]
        labels_bins = ["<$1k","$1k-5k","$5k-10k","$10k-20k","$20k-50k"]
        df["AmountBucket"] = pd.cut(df["Amount"], bins=bins, labels=labels_bins)
        df["VendorCode"] = df["VendorCategory"].astype("category").cat.codes
        df = pd.get_dummies(df, columns=["AmountBucket"], drop_first=False)
        df["Prior_Vendor"] = df["PriorFlag"] * df["VendorCode"]
        feature_cols = ["Amount","DayOfMonth","PriorFlag","VendorCode","Prior_Vendor"] +             [c for c in df.columns if c.startswith("AmountBucket_")]
        X = df[feature_cols]
        y = df["IsFraud"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.success("Step 1 completed.")
    except Exception as e:
        st.error(f"Step 1 error: {e}")

    # ----------------------------
    st.subheader("Step 2: Policy Learning - Multi-Model Training")
    try:
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_train_scaled, y_train)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        X_train_np = X_train.values
        X_test_np = X_test.values
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb_model.fit(X_train_np, y_train)
        nn = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=100)
        nn.fit(X_train_scaled, y_train)

        lr_probs = lr.predict_proba(X_test_scaled)[:,1]
        rf_probs = rf.predict_proba(X_test)[:,1]
        xgb_probs = xgb_model.predict_proba(X_test_np)[:,1]
        nn_probs = nn.predict_proba(X_test_scaled)[:,1]

        fig, ax = plt.subplots(figsize=(8,6))
        for name, probs in zip(
            ["Logistic Regression","Random Forest","XGBoost","Neural Network"],
            [lr_probs, rf_probs, xgb_probs, nn_probs]
        ):
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0,1],[0,1],"k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.set_title("ROC Curves")
        st.pyplot(fig)
        plt.close(fig)
        st.success("Step 2 completed.")
    except Exception as e:
        st.error(f"Step 2 error: {e}")

    # ----------------------------
    st.subheader("Step 3: Execution Module - Interpretability")
    try:
        st.write("Random Forest Feature Importance")
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
        fig, ax = plt.subplots()
        importances.plot(kind="barh", ax=ax)
        ax.set_title("Feature Importances")
        st.pyplot(fig)
        plt.close(fig)

        st.write("SHAP Summary Plot (Random Forest)")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        fig_shap = plt.figure()
        shap.summary_plot(shap_values[1], X_test, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)
        st.success("Step 3 completed.")
    except Exception as e:
        st.error(f"Step 3 error: {e}")

else:
    st.info("Awaiting CSV file upload.")

st.markdown("---")
st.caption("© 2024 Agentic AI Research | Complete Multi-Model Fraud Detection Pipeline")
