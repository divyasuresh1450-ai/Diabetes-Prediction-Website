import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction System")
st.write("Enter your details to get a prediction from our trained ML model.")

st.warning("This project is for educational purposes only. It is not a medical diagnosis.")

@st.cache_resource
def load_model():
    return joblib.load("diabetes_prediction_pipeline_compressed.pkl")

try:
    model = load_model()

    st.subheader("Enter Patient Information")

    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)

    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "No Info", "current", "former", "ever", "not current"]
    )

    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, value=5.5)

    glucose = st.number_input(
        "Blood Glucose Level",
        min_value=40,
        max_value=500,
        value=100
    )

    if st.button("Predict Diabetes"):
        input_data = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose
        }])

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("The model predicts diabetes.")
        else:
            st.success("The model does not predict diabetes.")

        st.caption("This prediction is not a medical diagnosis.")

except Exception as e:
    st.error("There was a problem loading the model or making a prediction.")
    st.exception(e)