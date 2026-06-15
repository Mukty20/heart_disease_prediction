import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title='Heart Disease Prediction', page_icon='❤️', layout='wide')

@st.cache_resource
def load_model():
    with open('heart-disease-model.sav', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'heart-disease-model.sav' not found. Please ensure it is in the same folder as app.py.")
    st.stop()

st.title('❤️ Heart Disease Prediction')
st.markdown('Enter the patient clinical details below and click **Predict** to get the result.')
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    age      = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex      = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
    cp       = st.selectbox('Chest Pain Type', [0, 1, 2, 3],
                            format_func=lambda x: ['Typical Angina (0)','Atypical Angina (1)','Non-anginal (2)','Asymptomatic (3)'][x])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=250, value=120)
    chol     = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)

with col2:
    fbs      = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
    restecg  = st.selectbox('Resting ECG Results', [0, 1, 2],
                            format_func=lambda x: ['Normal (0)', 'ST-T Abnormality (1)', 'LV Hypertrophy (2)'][x])
    thalach  = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=250, value=150)
    exang    = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

with col3:
    oldpeak  = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope    = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2],
                            format_func=lambda x: ['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'][x])
    ca       = st.selectbox('Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])
    thal     = st.selectbox('Thalassemia', [0, 1, 2],
                            format_func=lambda x: ['Normal (0)', 'Fixed Defect (1)', 'Reversible Defect (2)'][x])

st.divider()

if st.button('🔍 Predict', use_container_width=True, type='primary'):
    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 1:
        st.error(f'⚠️ The person is likely to have Heart Disease (confidence: {probability[1]*100:.1f}%)')
    else:
        st.success(f'✅ The person does NOT have Heart Disease (confidence: {probability[0]*100:.1f}%)')
