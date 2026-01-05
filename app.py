import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px

st.set_page_config(page_title="Earnings Manipulator Detection", layout="wide")

st.title("📊 Earnings Manipulator Detection System")
st.write("Upload your financial ratio dataset to detect earnings manipulation")

uploaded_file = st.file_uploader("Upload Dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    if "Manipulator" not in df.columns:
        st.error("Dataset must contain 'Manipulator' column")
        st.stop()

    X = df.drop("Manipulator", axis=1)
    y = df["Manipulator"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob)
        })

    result_df = pd.DataFrame(results)

    st.subheader("📈 Model Performance Comparison")
    st.dataframe(result_df)

    fig = px.bar(result_df, x="Model", y="Accuracy", title="Accuracy Comparison")
    st.plotly_chart(fig, use_container_width=True)
