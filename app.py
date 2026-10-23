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

# Қолданушыға модель көретін нақты бағандарды тексеру үшін шығарып көтейік (көмекші жол)
with st.expander("🔍 Техникалық ақпарат (Бағандарды тексеру)"):
    st.write("Модельде бар бағандар:", features)

st.markdown("Студенттің орташа көрсеткіштерін төмендегі тізімдер арқылы таңдаңыз:")
st.markdown("---")

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
    
    # Дәл сәйкестендіру үшін features ішінен іздейміз
    for feat in features:
        if attendance_choice == "60%-дан төмен" and "Below 60" in feat:
            input_data[feat] = 1.0
        elif attendance_choice == "70% - 80%" and "70%" in feat:
            input_data[feat] = 1.0
        elif attendance_choice == "80% - 90%" and "80%" in feat:
            input_data[feat] = 1.0
        elif attendance_choice == "90% - 100%" and "90%" in feat:
            input_data[feat] = 1.0
            
        if study_time_choice == "1 сағаттан аз" and "Less than 1" in feat:
            input_data[feat] = 1.0
        elif study_time_choice == "3-4 сағат" and "3-4" in feat:
            input_data[feat] = 1.0
        elif study_time_choice == "5-6 сағат" and "5-6" in feat:
            input_data[feat] = 1.0
        elif study_time_choice == "6 сағаттан көп" and "More than 6" in feat:
            input_data[feat] = 1.0

    # DataFrame құру
    input_df = pd.DataFrame([input_data])
    
    # Болжам жасау және ықтималдықты есептеу
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    
    # Егер модель классарды басқаша түсінсе немесе керісінше болса:
    late_prob = probabilities[1] * 100  
    ontime_prob = probabilities[0] * 100 
    
    st.markdown("### 📊 Болжам нәтижесі пен ықтималдықтар:")
    st.write(f"**Тапсырманың кеш қалу ықтималдығы:** `{late_prob:.1f}%`")
    st.progress(int(late_prob))
    
    if prediction == 1:
        st.error(f"⚠️ **Болжам:** Тапсырма мерзімінен кеш тапсырылуы мүмкін (Ықтималдығы: {late_prob:.1f}%)")
        st.warning("""
        **💡 Педагогикалық ұсыныстар мен қолдау шаралары:**
        * Студентке мерзім аяқталмай тұрып еске салу хабарламасын жіберу.
        * Уақытты жоспарлау бойынша кеңес беру.
        """)
    else:
        st.success(f"✅ **Болжам:** Тапсырма уақытында тапсырылады (Уақытында болу ықтималдығы: {ontime_prob:.1f}%)")
        st.info("""
        **💡 Педагогикалық ұсыныстар:**
        * Студенттің оқу қарқыны жақсы, оң пікір білдіріп қолдау көрсету.
        """)

st.markdown("---")
st.caption("Мектепте информатиканы оқытуда бұлтты технологиялар мен ЖИ қолдану зерттеу жұмысы | 2026")
