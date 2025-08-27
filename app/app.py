# app/app.py
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# --- Page config ---
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# --- Load model & metadata ---
MODEL_PATH = Path("models/student_model.pkl")
FEATURES_PATH = Path("models/features.pkl")
DEFAULTS_PATH = Path("models/defaults.json")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
with open(DEFAULTS_PATH, "r") as f:
    defaults = json.load(f)

# --- App title ---
st.title("Student Performance Predictor")
st.write("Predict if a student will pass (Final Marks ≥ 10).")

# --- Sidebar: Mode selection ---
mode = st.sidebar.radio("Mode", ["Single input", "Batch upload (CSV)"])

# --- Batch CSV upload mode ---
if mode == "Batch upload (CSV)":
    uploaded_file = st.file_uploader(
        "Upload CSV with the same columns as training data",
        type=["csv"]
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Fill missing columns with defaults
        for col in features:
            if col not in df.columns:
                df[col] = defaults[col]
            df[col].fillna(defaults[col], inplace=True)

        # Make predictions
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]
        df_result = df.copy()
        df_result["Predicted_Pass"] = ["Pass" if p==1 else "Fail" for p in preds]
        df_result["Pass_Probability"] = probs.round(2)

        st.success("Predictions completed!")
        st.dataframe(df_result)

        # Download results
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("Download results CSV", csv, file_name="predictions.csv", mime="text/csv")

# --- Single student input mode ---
else:
    st.subheader("Enter student data")
    input_data = {}
    cols = st.columns(2)
    for i, col in enumerate(features):
        default_val = defaults.get(col, "")
        # numeric input if default is number, else text
        if isinstance(default_val, (int, float)):
            val = cols[i % 2].number_input(col, value=float(default_val))
        else:
            val = cols[i % 2].text_input(col, value=str(default_val))
        input_data[col] = val

    if st.button("Predict"):
        df_input = pd.DataFrame([input_data])

        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]

        if pred == 1:
            st.success(f"Predicted: PASS — probability {prob:.2f}")
        else:
            st.error(f"Predicted: FAIL — probability {prob:.2f}")

        # Feature importances
        try:
            classifier = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']

            # Attempt to get processed feature names
            try:
                feat_names = preprocessor.get_feature_names_out()
            except:
                feat_names = features

            importances = classifier.feature_importances_
            fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10)

            st.subheader("Top 10 Feature Importances")
            st.bar_chart(fi)
        except Exception:
            st.write("Feature importance not available for this model.")
