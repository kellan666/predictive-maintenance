import streamlit as st
import pandas as pd
import joblib

# Set up page config
st.set_page_config(page_title="AI Predictive Maintenance", page_icon="🤖")

st.title("🤖 AI Predictive Maintenance")
st.write("Enter machine operating parameters to predict failure risk.")

# Load your trained model
@st.cache_resource
def load_model():
    # Make sure 'model.joblib' or 'model.pkl' is uploaded to your GitHub repository
    return joblib.load("model.joblib")

# User Input Fields
air_temperature = st.number_input("Air Temperature (K)", value=300)
process_temperature = st.number_input("Process Temperature (K)", value=310)
rotational_speed = st.number_input("Rotational Speed (RPM)", value=1500)
torque = st.number_input("Torque (Nm)", value=40)
tool_wear = st.number_input("Tool Wear (min)", value=100)

# Prediction Logic
if st.button("Predict"):
    try:
        model = load_model()
        
        # Prepare data frame for prediction
        input_data = pd.DataFrame([{
            "Air_temperature_K": air_temperature,
            "Process_temperature_K": process_temperature,
            "Rotational_speed_RPM": rotational_speed,
            "Torque_Nm": torque,
            "Tool_wear_min": tool_wear
        }])

        prediction = model.predict(input_data)[0]

        st.subheader("Machine Condition")
        if prediction == 1:
            st.error("🔴 HIGH FAILURE RISK\n\n⚠️ Preventive maintenance is recommended.")
        else:
            st.success("🟢 MACHINE HEALTHY\n\n✅ No immediate maintenance required.")
            
    except Exception as e:
        st.error(f"Error loading model or generating prediction: {e}")
