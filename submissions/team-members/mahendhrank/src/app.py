import streamlit as st
import pandas as pd
import pickle

# --- Load model and preprocessors ---
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("src/preprocessors.pkl", "rb") as f:
    preprocessors = pickle.load(f)

le_dict = preprocessors["label_encoders"]
scaler = preprocessors["scaler"]
feature_names = preprocessors["features"]

st.title("💰 Income Prediction App")
st.write("Predict if a person earns >50K based on their attributes")

inputs = {}

st.sidebar.subheader("⚙️ Show/Hide Input Fields")
# Checkboxes to toggle visibility
show_age = st.sidebar.checkbox("Age", True)
show_education_num = st.sidebar.checkbox("Education Number", True)
show_capital_gain = st.sidebar.checkbox("Capital Gain", True)
show_capital_loss = st.sidebar.checkbox("Capital Loss", True)
show_hours_per_week = st.sidebar.checkbox("Hours per Week", True)
show_workclass = st.sidebar.checkbox("Workclass", True)
show_education = st.sidebar.checkbox("Education", True)
show_marital_status = st.sidebar.checkbox("Marital Status", True)
show_occupation = st.sidebar.checkbox("Occupation", True)
show_relationship = st.sidebar.checkbox("Relationship", True)
show_race = st.sidebar.checkbox("Race", True)
show_sex = st.sidebar.checkbox("Sex", True)
show_native_country = st.sidebar.checkbox("Native Country", True)

# --- Numeric inputs ---
if show_age:
    inputs["age"] = st.number_input("Age", min_value=18, max_value=100, value=30)
if show_education_num:
    inputs["education-num"] = st.number_input("Education Number", min_value=1, max_value=16, value=10)
if show_capital_gain:
    inputs["capital-gain"] = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
if show_capital_loss:
    inputs["capital-loss"] = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
if show_hours_per_week:
    inputs["hours-per-week"] = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

# --- Categorical inputs ---
if show_workclass:
    inputs["workclass"] = le_dict["workclass"].transform(
        [st.selectbox("Workclass", le_dict["workclass"].classes_)]
    )[0]
if show_education:
    inputs["education"] = le_dict["education"].transform(
        [st.selectbox("Education", le_dict["education"].classes_)]
    )[0]
if show_marital_status:
    inputs["marital-status"] = le_dict["marital-status"].transform(
        [st.selectbox("Marital Status", le_dict["marital-status"].classes_)]
    )[0]
if show_occupation:
    inputs["occupation"] = le_dict["occupation"].transform(
        [st.selectbox("Occupation", le_dict["occupation"].classes_)]
    )[0]
if show_relationship:
    inputs["relationship"] = le_dict["relationship"].transform(
        [st.selectbox("Relationship", le_dict["relationship"].classes_)]
    )[0]
if show_race:
    inputs["race"] = le_dict["race"].transform(
        [st.selectbox("Race", le_dict["race"].classes_)]
    )[0]
if show_sex:
    inputs["sex"] = le_dict["sex"].transform(
        [st.selectbox("Sex", le_dict["sex"].classes_)]
    )[0]
if show_native_country:
    inputs["native-country"] = le_dict["native-country"].transform(
        [st.selectbox("Native Country", le_dict["native-country"].classes_)]
    )[0]

# --- Build dataframe in correct order ---
input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = [0] * len(feature_names)

for key, val in inputs.items():
    input_data[key] = val

# Scale numeric features
numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

st.write("### Input Preview")
st.dataframe(input_data)

# --- Predict ---
if st.button("Predict Income"):
    pred = model.predict(input_data)
    st.success(f"Predicted Income: {'>50K' if pred[0]==1 else '<=50K'}")
