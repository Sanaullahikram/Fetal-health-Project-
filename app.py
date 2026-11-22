import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fetal Health Classifier", layout="wide")
st.title("Fetal Health Classifier (Random Forest)")

model = joblib.load("fetal_rf_model.pkl")
feature_cols = joblib.load("feature_columns (1).pkl")
df_ref = pd.read_csv("fetal health.csv.zip").drop(columns=["fetal_health"])

st.subheader("Enter Feature Values")
input_data = {}

cols = st.columns(3)
for idx, col in enumerate(feature_cols):
  default_val = float(df_ref[col].median())
  min_val = float(df_ref[col].min())
  max_val = float(df_ref[col].max())
  with cols[idx % 3]:
    input_data[col] = st.number_input(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
    )

if st.button("Predict Fetal Health"):
  input_df = pd.DataFrame([input_data])[feature_cols]
  pred = int(model.predict(input_df)[0])
  probs = model.predict_proba(input_df)[0]

  label_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}
  st.success(f"Predicted Class: {pred} ({label_map.get(pred, 'Unknown')})")

  prob_df = pd.DataFrame(
      {"Class": [label_map.get(int(c), str(c)) for c in model.classes_],
       "Probability": np.round(probs, 4)}
  )
  st.dataframe(prob_df, use_container_width=True)
