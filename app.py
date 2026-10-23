import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Тапсырманың кеш қалуын болжау", page_icon="📚", layout="centered")

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "student_model.joblib"
    return joblib.load(path)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]  # Kaggle-де сақталған бағандар тізімі

st.title("🎓 Педагогикалық болжау жүйесі")
st.subheader("2-ші зерттеу сұрағы: Келесі тапсырманың кеш тапсырылуын болжау")
st.markdown("Студенттің орташа көрсеткіштерін төмендегі тізімдер арқылы таңдаңыз:")

st.markdown("---")

# Қолданушыға түсінікті ашылмалы тізімдер
attendance_choice = st.selectbox(
    "Студенттің сабаққа қатысу деңгейі:",
    ["60%-дан төмен", "70% - 80%", "80% - 90%", "90% - 100%"]
)

study_time_choice = st.selectbox(
    "Күндізгі орташа оқу уақыты:",
    ["1 сағаттан аз", "3-4 сағат", "5-6 сағат", "6 сағаттан көп"]
)

st.markdown("---")

if st.button("🚀 Болжам жасау", type="primary"):
    # Барлық бағандарды 0 деп бастаймыз
    input_data = {feat: 0.0 for feat in features}
    
    # Таңдалған мәндерге сәйкес бағанды 1-ге теңейміз
    if attendance_choice == "60%-дан төмен" and "Avg_Attendance_Below 60%" in features:
        input_data["Avg_Attendance_Below 60%"] = 1.0
    elif attendance_choice == "70% - 80%" and "Avg_Attendance_70%-80%" in features:
        input_data["Avg_Attendance_70%-80%"] = 1.0
    elif attendance_choice == "80% - 90%" and "Avg_Attendance_80%-90%" in features:
        input_data["Avg_Attendance_80%-90%"] = 1.0
    elif attendance_choice == "90% - 100%" and "Avg_Attendance_90% - 100%" in features:
        input_data["Avg_Attendance_90% - 100%"] = 1.0
        
    if study_time_choice == "1 сағаттан аз" and "Avg_Study_time_per_day_Less than 1 hour" in features:
        input_data["Avg_Study_time_per_day_Less than 1 hour"] = 1.0
    elif study_time_choice == "3-4 сағат" and "Avg_Study_time_per_day_3-4 hours" in features:
        input_data["Avg_Study_time_per_day_3-4 hours"] = 1.0
    elif study_time_choice == "5-6 сағат" and "Avg_Study_time_per_day_5-6 hours" in features:
        input_data["Avg_Study_time_per_day_5-6 hours"] = 1.0
    elif study_time_choice == "6 сағаттан көп" and "Avg_Study_time_per_day_More than 6 hours" in features:
        input_data["Avg_Study_time_per_day_More than 6 hours"] = 1.0

    # DataFrame құру
    input_df = pd.DataFrame([input_data])
    
    # Болжам жасау және ықтималдықты (пайызды) есептеу
    prediction = int(model.predict(input_df)[0])
    
    # model.predict_proba арқылы пайыздарды аламыз
    probabilities = model.predict_proba(input_df)[0]
    late_prob = probabilities[1] * 100  # Кеш қалу ықтималдығы (%)
    ontime_prob = probabilities[0] * 100 # Уақытында тапсыру ықтималдығы (%)
    
    st.markdown("### 📊 Болжам нәтижесі пен ықтималдықтар:")
    
    # Прогресс бар арқылы пайызды визуализация жасау
    st.write(f"**Тапсырманың кеш қалу ықтималдығы:** `{late_prob:.1f}%`")
    st.progress(int(late_prob))
    
    if prediction == 1:
        st.error(f"⚠️ **Болжам:** Тапсырма мерзімінен кеш тапсырылуы мүмкін (Ықтималдығы: {late_prob:.1f}%)")
        st.warning("""
        **💡 Педагогикалық ұсыныстар мен қолдау шаралары:**
        * Студентке мерзім аяқталмай тұрып еске салу хабарламасын жіберу[cite: 1].
        * Уақытты жоспарлау бойынша кеңес беру[cite: 1].
        """)
    else:
        st.success(f"✅ **Болжам:** Тапсырма уақытында тапсырылады (Уақытында болу ықтималдығы: {ontime_prob:.1f}%)")
        st.info("""
        **💡 Педагогикалық ұсыныстар:**
        * Студенттің оқу қарқыны жақсы, оң пікір білдіріп қолдау көрсету.
        """)

st.markdown("---")
st.caption("Мектепте информатиканы оқытуда бұлтты технологиялар мен ЖИ қолдану зерттеу жұмысы | 2026")
