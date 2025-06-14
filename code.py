import streamlit as st
import joblib
import pandas as pd
import pickle

# Load your trained model
model = joblib.load("meilleur_modele6.joblib")

# Load the list of model feature columns saved from training
with open('model_features2.joblib', 'rb') as f:
    model_features = pickle.load(f)

# Streamlit UI
st.title("Cancer DSS Status Prediction🎗️")

# Input widgets
subtype = st.selectbox("SUBTYPE", ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Basal', 'BRCA_Her2', 'Other'])
cancer_type = st.selectbox("CANCER_TYPE_ACRONYM", ['BRCA'])
age = st.number_input("AGE", min_value=0, max_value=120, value=50)
sex = st.selectbox("SEX", ['Female', 'Male'])
tumor_stage = st.selectbox("AJCC_PATHOLOGIC_TUMOR_STAGE", ['STAGE I', 'STAGE II', 'STAGE III', 'STAGE IV'])
staging_edition = st.selectbox("AJCC_STAGING_EDITION", ['6TH', '7TH', '8TH'])
days_followup = st.number_input("DAYS_LAST_FOLLOWUP", value=1000.0)
dAYS_TO_BIRTH = st.number_input("DAYS_TO_BIRTH", value=-20211)
days_to_diag = st.number_input("DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS", value=300)
ethnicity = st.selectbox("ETHNICITY", ['Hispanic Or Latino', 'Not Hispanic Or Latino', 'Unknown'])
history_neidjuvant_trtyn = st.selectbox("HISTORY_NEOADJUVANT_TRTYN", ['Yes', 'No'])
icd10 = st.text_input('ICD_10', value='C50.9')
iCD_O_3_HISTOLOGY = st.text_input('ICD_O_3_HISTOLOGY', value='8500/3')
iCD_O_3_CLINICAL = st.text_input('ICD_O_3_SITE', value='C50.9')
nEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT = st.text_input('NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT', value='No')
m_stage = st.text_input("PATH_M_STAGE", value='MX')
n_stage = st.text_input("PATH_N_STAGE", value='NX')
t_stage = st.text_input("PATH_T_STAGE", value='TX')
cancer_status = st.selectbox("PERSON_NEOPLASM_CANCER_STATUS", ['With Tumor', 'Tumor Free'])
pRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT = st.selectbox('PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT', ['Yes', 'No'])
pRIOR_DX = st.selectbox("PRIOR_DX", ['Yes', 'No'])
rACE = st.text_input('RACE', value='White')
radiation = st.selectbox("RADIATION_THERAPY", ['Yes', 'No'])
iN_PANCANPATHWAYS_FREEZE = st.selectbox("IN_PANCANPATHWAYS_FREEZE", ['Yes', 'No'])

# Create DataFrame with user input
input_data = pd.DataFrame([[
    subtype, cancer_type, age, sex, tumor_stage, staging_edition,
    days_followup, dAYS_TO_BIRTH, days_to_diag, ethnicity,
    history_neidjuvant_trtyn, icd10, iCD_O_3_HISTOLOGY, iCD_O_3_CLINICAL,
    nEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT, m_stage, n_stage, t_stage,
    cancer_status, pRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT,
    pRIOR_DX, rACE, radiation, iN_PANCANPATHWAYS_FREEZE
]], columns=[
    'SUBTYPE', 'CANCER_TYPE_ACRONYM', 'AGE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE',
    'AJCC_STAGING_EDITION', 'DAYS_LAST_FOLLOWUP', 'DAYS_TO_BIRTH', 'DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS',
    'ETHNICITY', 'HISTORY_NEOADJUVANT_TRTYN', 'ICD_10', 'ICD_O_3_HISTOLOGY', 'ICD_O_3_CLINICAL',
    'NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT', 'PATH_M_STAGE', 'PATH_N_STAGE', 'PATH_T_STAGE',
    'PERSON_NEOPLASM_CANCER_STATUS', 'PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT',
    'PRIOR_DX', 'RACE', 'RADIATION_THERAPY', 'IN_PANCANPATHWAYS_FREEZE'
])

# Encoding categorical columns the same way you did during training
categorical_cols = input_data.select_dtypes(include=['object']).columns

# Label encoding for columns with <= 5 unique values; else one-hot encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_df = input_data.copy()

for col in categorical_cols:
    if input_data[col].nunique() <= 5:
        encoded_df[col] = label_encoder.fit_transform(input_data[col])
    else:
        dummies = pd.get_dummies(input_data[col], prefix=col)
        encoded_df = pd.concat([encoded_df.drop(col, axis=1), dummies], axis=1)

# Reindex columns to match model features, fill missing with 0
encoded_df = encoded_df.reindex(columns=model_features, fill_value=0)

# Prediction on button press
if st.button("Predict DSS Status"):
    try:
        prediction = model.predict(encoded_df)[0]
        st.success(f"Predicted DSS Status: {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")





