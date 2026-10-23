import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Беттің конфигурациясы
st.set_page_config(
    page_title="Тапсырманың кеш қалуын болжау",
    page_icon="📚",
    layout="centered"
)

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "student_model.joblib"
    return joblib.load(path)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]  # Kaggle-де сақталған бағандар тізімі

# Баған атауларын қазақшаға аударуға арналған сөздік (қажет болса толықтыра аласыз)
kazakh_labels = {
    "Avg_Attendance_70%-80%": "Орташа қатысу: 70% - 80%",
    "Avg_Attendance_80%-90%": "Орташа қатысу: 80% - 90%",
    "Avg_Attendance_90% - 100%": "Орташа қатысу: 90% - 100%",
    "Avg_Attendance_Below 60%": "Орташа қатысу: 60%-дан төмен",
    "Avg_Study_time_per_day_3-4 hours": "Күндізгі оқу уақыты: 3-4 сағат",
    "Avg_Study_time_per_day_5-6 hours": "Күндізгі оқу уақыты: 5-6 сағат",
    "Avg_Study_time_per_day_Less than 1 hour": "Күндізгі оқу уақыты: 1 сағаттан аз",
    "Avg_Study_time_per_day_More than 6 hours": "Күндізгі оқу уақыты: 6 сағаттан көп"
}

# Интерфейс тақырыбы
st.title("🎓 Педагогикалық болжау жүйесі")
st.subheader("2-ші зерттеу сұрағы: Келесі тапсырманың кеш тапсырылуын болжау")
st.markdown("Бұл веб-қосымша студенттің оқу көрсеткіштері негізінде келесі тапсырманың мерзімінде немесе кеш орындалатынын болжайды.")

with st.expander("ℹ️ Модель туралы ақпарат"):
    st.write(f"**Модель күтетін белгілер саны:** {len(features)} баған")

st.markdown("---")
st.write("### 📝 Студенттің көрсеткіштерін енгізіңіз:")

# Динамикалық түрде енгізу өрістерін жасау (қазақша атаумен)
input_vals = []
for feature_name in features:
    # Егер сөздікте қазақшасы болса соны аламыз, болмаса өзін шығарамыз
    label = kazakh_labels.get(feature_name, feature_name)
    val = st.number_input(f"{label}", value=0.0, step=0.01, format="%.2f")
    input_vals.append(val)

st.markdown("---")

if st.button("🚀 Болжам жасау", type="primary"):
    # DataFrame құру (модель дәл сол ағылшынша баған аттарын күтетіндіктен, columns=features қолданылады)
    input_df = pd.DataFrame([input_vals], columns=features)
    
    # Болжам жасау
    prediction = int(model.predict(input_df)[0])
    
    st.markdown("### 📊 Болжам нәтижесі:")
    
    if prediction == 1:
        st.error("⚠️ **Болжам:** Тапсырма мерзімінен кеш тапсырылуы мүмкін (1)")
        
        st.warning("""
        **💡 Педагогикалық ұсыныстар мен қолдау шаралары:**
        * Студентке мерзім аяқталмай тұрып еске салу хабарламасын жіберу[cite: 1].
        * Уақытты жоспарлау және тапсырманы бөліктерге бөліп орындау бойынша кеңес беру[cite: 1].
        * Қосымша түсіндіру сабақтарын ұйымдастыру[cite: 1].
        """)
    else:
        st.success("✅ **Болжам:** Тапсырма уақытында тапсырылады (0)")
        
        st.info("""
        **💡 Педагогикалық ұсыныстар:**
        * Студенттің қазіргі оқу қарқыны жақсы, ынталандыру мақсатында қолдау көрсету.
        """)

st.markdown("---")
st.caption("Мектепте информатиканы оқытуда бұлтты технологиялар мен ЖИ қолдану зерттеу жұмысы | 2026")
