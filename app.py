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
features = bundle["features"]  # Kaggle-де сақталған баған аттары тізімі

st.title("2-ші зерттеу сұрағы: Келесі тапсырманың кеш тапсырылуын болжау")
st.write("Студенттің көрсеткіштерін енгізіңіз:")

# Пайдаланушыдан мәндерді алу
val1 = st.number_input("Алдыңғы жұмыстардың кешігу үлесі (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.1)
val2 = st.number_input("Осыған дейін орындалған жұмыстар саны", min_value=0, max_value=100, value=5)
val3 = st.number_input("Соңғы екі аптадағы белсенді күндер саны", min_value=0, max_value=14, value=7)

if st.button("Болжам жасау"):
    # Модель үйренген дәл сол баған аттарымен DataFrame құру
    input_df = pd.DataFrame([[val1, val2, val3]], columns=features)
    
    # Болжам жасау
    prediction = int(model.predict(input_df)[0])
    
    if prediction == 1:
        st.error("⚠️ Болжам: Тапсырма мерзімінен кеш тапсырылуы мүмкін (1)")
    else:
        st.success("✅ Болжам: Тапсырма уақытында тапсырылады (0)")
