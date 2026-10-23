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

# Интерфейс тақырыбы
st.title("🎓 Педагогикалық болжау жүйесі")
st.subheader("2-ші зерттеу сұрағы: Келесі тапсырманың кеш тапсырылуын болжау")
st.markdown("Бұл веб-қосымша студенттің оқу көрсеткіштері негізінде келесі тапсырманың уақытында немесе кеш орындалатынын болжайды.")

with st.expander("ℹ️ Модель туралы ақпарат"):
    st.write(f"**Модель күтетін белгілер саны:** {len(features)} баған")
    st.write("Бұл көрсеткіштер студенттің қатысу белсенділігі мен оқу уақытына негізделген.")

st.markdown("---")
st.write("### 📝 Студенттің көрсеткіштерін енгізіңіз:")

# Динамикалық түрде бағандарға қарай енгізу өрістерін жасау
input_vals = []
for feature_name in features:
    val = st.number_input(f"{feature_name}", value=0.0, step=0.01, format="%.2f")
    input_vals.append(val)

st.markdown("---")

if st.button("🚀 Болжам жасау", type="primary"):
    # DataFrame құру
    input_df = pd.DataFrame([input_vals], columns=features)
    
    # Болжам жасау
    prediction = int(model.predict(input_df)[0])
    
    st.markdown("### 📊 Болжам нәтижесі:")
    
    if prediction == 1:
        st.error("⚠️ **Болжам:** Тапсырма мерзімінен кеш тапсырылуы мүмкін (1)")
        
        # Қосымша педагогикалық кеңестер (педагогикалық қолдану)
        st.warning("""
        **💡 Педагогикалық ұсыныстар мен қолдау шаралары:**
        * Студентке мерзім аяқталмай тұрып еске салу хабарламасын жіберу.
        * Уақытты жоспарлау және тапсырманы бөліктерге бөліп орындау бойынша кеңес беру.
        * Қосымша түсіндіру сабақтарын немесе консультация ұсыну.
        """)
    else:
        st.success("✅ **Болжам:** Тапсырма уақытында тапсырылады (0)")
        
        st.info("""
        **💡 Педагогикалық ұсыныстар:**
        * Студенттің қазіргі оқу қарқыны жақсы, ынталандыру мақсатында оң пікір білдіруге болады.
        """)

# Қосымша ақпарат (Footer)
st.markdown("---")
st.caption("Мектепте информатиканы оқытуда бұлтты технологиялар мен ЖИ қолдану зерттеу жұмысы | 2026")
