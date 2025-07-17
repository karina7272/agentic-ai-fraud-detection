
# Optimized Agentic AI Fraud Detection Application

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

st.set_page_config(page_title="Optimized Agentic AI Fraud Detection", layout="wide")

st.title("🚀 Optimized Agentic AI Fraud Detection")
st.markdown("Improved for Speed and Scalability")

uploaded_file = st.file_uploader("Upload your CSV (EntryID, Amount, VendorCategory, DayOfMonth, PriorFlag, IsFraud)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.sample(n=min(10000, len(df)), random_state=42)  # Speed fix: limit rows
    st.subheader("Data Sample")
    st.dataframe(df.head())

    bins = [0, 1000, 5000, 10000, 20000, 50000]
    labels_bins = ["<$1k", "$1k-5k", "$5k-10k", "$10k-20k", "$20k-50k"]
    df["AmountBucket"] = pd.cut(df["Amount"], bins=bins, labels=labels_bins)
    df["VendorCode"] = df["VendorCategory"].astype("category").cat.codes
    df = pd.get_dummies(df, columns=["AmountBucket"], drop_first=False)
    df["Prior_Vendor"] = df["PriorFlag"] * df["VendorCode"]

    feature_cols = ["Amount", "DayOfMonth", "PriorFlag", "VendorCode", "Prior_Vendor"] +         [c for c in df.columns if c.startswith("AmountBucket_")]
    X = df[feature_cols]
    y = df["IsFraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models with speed improvements
    st.subheader("Model Training")
    st.write("Training Logistic Regression, Random Forest, XGBoost, Neural Net (max_iter=50)")

    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    nn = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=50, random_state=42)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)
    nn.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train.values, y_train)

    models = {
        "Logistic Regression": (lr, X_test_scaled),
        "Random Forest": (rf, X_test),
        "Neural Network": (nn, X_test_scaled),
        "XGBoost": (xgb_model, X_test.values)
    }

    fig, ax = plt.subplots()
    for name, (model, X_eval) in models.items():
        probs = model.predict_proba(X_eval)[:,1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--")
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Random Forest Feature Importances")
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots()
    importances.plot(kind="barh", ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)

    st.subheader("SHAP Values (Random Forest)")
    sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(sample_X)
    fig_shap = plt.figure()
    shap.summary_plot(shap_values[1], sample_X, show=False)
    st.pyplot(fig_shap)
else:
    st.info("Please upload a CSV file to begin.")
