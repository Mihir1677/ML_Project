import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Added for better visualization
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Streamlit Page Configuration
st.set_page_config(page_title="ML Model Evaluation", layout="wide")

st.title("📊 Machine Learning Model Evaluation")
st.sidebar.header("ℹ️ About this Tool")
st.sidebar.write(
    "This application evaluates classification models using various metrics. "
    "Upload a CSV file with labeled data to analyze model performance."
)


# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Preprocessing Function
def preprocess_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna('', inplace=True)
    
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, df

# Model Evaluation Function
def evaluate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores, prec_scores, rec_scores, f1_scores, conf_matrices = [], [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        conf_matrices.append(confusion_matrix(y_test, y_pred).tolist())

    return {
        'accuracy': round(np.mean(acc_scores) * 100, 2),
        'precision': round(np.mean(prec_scores) * 100, 2),
        'recall': round(np.mean(rec_scores) * 100, 2),
        'f1_score': round(np.mean(f1_scores) * 100, 2),
        'confusion_matrix': conf_matrices,
        'acc_scores': acc_scores
    }

# Process File if Uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Uploaded Dataset Preview")
    st.write(df.head())

    # Preprocess Data
    X, y, processed_df = preprocess_data(df)

    # Define Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    # Evaluate Models
    results, feature_importance, boxplot_data = [], {}, {}

    for model_name, model in models.items():
        metrics = evaluate_model(model, X, y)
        results.append({'Model': model_name, **metrics})

        if hasattr(model, 'feature_importances_'):
            model.fit(X, y)
            feature_importance[model_name] = sorted(
                zip(processed_df.columns[:-1], model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )

        # Boxplot Data for Accuracy Scores
        boxplot_data[model_name] = [
            np.min(metrics['acc_scores']) * 100,
            np.percentile(metrics['acc_scores'], 25) * 100,
            np.median(metrics['acc_scores']) * 100,
            np.percentile(metrics['acc_scores'], 75) * 100,
            np.max(metrics['acc_scores']) * 100
        ]

    # Display Model Results
    st.subheader("🔍 Model Performance Summary")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Best Model
    best_model = max(results, key=lambda x: x['accuracy']) if results else None
    if best_model:
        st.success(f"🏆 Best Performing Model: **{best_model['Model']}** with **{best_model['accuracy']}% Accuracy**")

    # Accuracy Bar Chart (Using Plotly for Better Scaling)
    st.subheader("📊 Accuracy Comparison")
    
    fig = px.bar(results_df, x="Model", y="accuracy", title="Model Accuracy Comparison", text="accuracy")
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=500)  # Adjust height for better view
    st.plotly_chart(fig, use_container_width=True)  # Auto-resizes

    # Feature Importance Visualization
    if feature_importance:
        st.subheader("🔬 Feature Importance (For Tree-Based Models)")
        for model, importance in feature_importance.items():
            st.write(f"**{model}**")
            importance_df = pd.DataFrame(importance, columns=["Feature", "Importance"])

            fig = px.bar(importance_df, x="Importance", y="Feature", title=f"Feature Importance - {model}", orientation='h')
            st.plotly_chart(fig, use_container_width=True)  # Auto-scaling

    # Boxplot for Model Performance
    st.subheader("📈 Model Accuracy Distribution (Boxplot)")
    
    fig, ax = plt.subplots(figsize=(10, 5))  # Wider figure for better view
    sns.boxplot(data=pd.DataFrame(boxplot_data, index=["Min", "Q1", "Median", "Q3", "Max"]).T, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Distribution")

    st.pyplot(fig)

    # Dataset Information
    with st.expander("📌 Dataset Overview (Click to Expand)"):
        st.write(f"📏 Shape: {processed_df.shape}")
        st.write(f"❌ Missing Values: {processed_df.isnull().sum().sum()}")
        st.write(f"📌 Data Types: {processed_df.dtypes.to_dict()}")
