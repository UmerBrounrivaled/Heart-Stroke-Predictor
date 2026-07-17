import streamlit as st
import joblib
import pandas as pd

model=joblib.load('KNN_heart.pkl')
scalar=joblib.load('scalar.pkl')
columns=joblib.load('columns.pkl')
st.title("Heart Stroke Prediciton by UmerBro🌟")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("predict"):
  input_data = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "RestingECG": resting_ecg,
    "MaxHR": max_hr,
    "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope
}
  input_df = pd.DataFrame([input_data])
  # Handle categorical features by creating dummy variables, ensuring all original columns are present
  # and aligning the order with the columns used during training.
  for col in columns:
    if col not in input_df.columns:
      input_df[col] = 0

  # Map input categorical features to dummy columns. This assumes that the `columns` list
  # contains the expected order of features including dummy variables.
  # For 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'

  # Convert 'Sex'
  input_df['Sex_M'] = 1 if input_df['Sex'].iloc[0] == 'M' else 0
  input_df = input_df.drop('Sex', axis=1)

  # Convert 'ChestPainType'
  input_df['ChestPainType_ATA'] = 1 if input_df['ChestPainType'].iloc[0] == 'ATA' else 0
  input_df['ChestPainType_NAP'] = 1 if input_df['ChestPainType'].iloc[0] == 'NAP' else 0
  input_df['ChestPainType_TA'] = 1 if input_df['ChestPainType'].iloc[0] == 'TA' else 0
  input_df = input_df.drop('ChestPainType', axis=1)

  # Convert 'RestingECG'
  input_df['RestingECG_Normal'] = 1 if input_df['RestingECG'].iloc[0] == 'Normal' else 0
  input_df['RestingECG_ST'] = 1 if input_df['RestingECG'].iloc[0] == 'ST' else 0
  input_df = input_df.drop('RestingECG', axis=1)

  # Convert 'ExerciseAngina'
  input_df['ExerciseAngina_Y'] = 1 if input_df['ExerciseAngina'].iloc[0] == 'Y' else 0
  input_df = input_df.drop('ExerciseAngina', axis=1)

  # Convert 'ST_Slope'
  input_df['ST_Slope_Flat'] = 1 if input_df['ST_Slope'].iloc[0] == 'Flat' else 0
  input_df['ST_Slope_Up'] = 1 if input_df['ST_Slope'].iloc[0] == 'Up' else 0
  input_df = input_df.drop('ST_Slope', axis=1)

  # Ensure all expected columns are present, fill missing with 0
  # and reorder columns to match the training data
  final_input_df = pd.DataFrame(0, index=[0], columns=columns)
  for col in input_df.columns:
      if col in final_input_df.columns:
          final_input_df[col] = input_df[col].iloc[0]

  scaled_input=scalar.transform(final_input_df)
  prediction = model.predict(scaled_input)[0]
  if prediction == 1:
    st.error("⚠️ High Risk of Heart Disease")
  else:
    st.success("✅ Low Risk of Heart Disease")
