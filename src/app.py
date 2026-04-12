import streamlit as st
import pandas as pd
import mlflow.pyfunc

st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥", layout="wide")

st.title("🏥 Hospital Readmission ML Predictor")
st.markdown("This live dashboard runs our **Champion XGBoost Model** from the MLflow Registry. It predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days based on clinical data.")

mlflow.set_tracking_uri("mlruns/")

@st.cache_resource
def load_champion_model():
    # Load the champion model directly from MLflow
    model_uri = "models:/HospitalReadmissionChampion/latest"
    return mlflow.pyfunc.load_model(model_uri)

@st.cache_data
def load_sample_data():
    return pd.read_csv("data/processed/diabetic_data_clean.csv")

try:
    with st.spinner("Connecting to Model Registry..."):
        model = load_champion_model()
        df = load_sample_data()
        
    st.success("Champion XGBoost model successfully loaded from MLflow!")
    
    st.subheader("Interactive Prediction Demo")
    st.write("Since creating a patient from scratch requires exactly 45 clinical features, you can click the button below to randomly sample a real patient from the dataset and run them through our trained model.")
    
    if st.button("Sample Random Patient & Run Prediction", type="primary"):
        sample = df.sample(1)
        X_sample = sample.drop(columns=["readmitted"])
        y_true = sample["readmitted"].values[0]
        
        st.markdown("### Patient Clinical Data:")
        st.dataframe(X_sample)
        
        with st.spinner("Running inference..."):
            pred = model.predict(X_sample)[0]
        
        st.markdown("### Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Model Prediction", value="Early Readmission (<30 days)" if pred == 1 else "No Early Readmission")
            
        with col2:
            st.metric(label="Actual Ground Truth", value="Readmitted" if y_true == 1 else "Not Readmitted")

        if pred == y_true:
            st.balloons()
            st.info("✅ The model predicted accurately!")
        else:
            st.warning("❌ The model was incorrect this time.")
            
except Exception as e:
    st.error(f"Error loading model or data. Ensure MLFlow tracking is generated. Details: {e}")
