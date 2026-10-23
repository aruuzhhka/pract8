import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "student_model.joblib"
    return joblib.load(path)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]  # Kaggle-де сақталған бағандар тізімі

st.title("2-ші зерттеу сұрағы: Келесі тапсырманың кеш тапсырылуын болжау")
st.write(f"Модель күтетін белгілер (бағандар): `{features}`")
st.write("Студенттің көрсеткіштерін енгізіңіз:")

# Динамикалық түрде бағандар санына қарай енгізу өрістерін жасау
input_vals = []
for i, feature_name in enumerate(features):
    val = st.number_input(f"{feature_name}", value=0.0)
    input_vals.append(val)

if st.button("Болжам жасау"):
    # Модель үйренген дәл сол баған аттарымен DataFrame құру
    input_df = pd.DataFrame([input_vals], columns=features)
    
    # Болжам жасау
    prediction = int(model.predict(input_df)[0])
    
    if prediction == 1:
        st.error("⚠️ Болжам: Тапсырма мерзімінен кеш тапсырылуы мүмкін (1)")
    else:
        st.success("✅ Болжам: Тапсырма уақытында тапсырылады (0)")
