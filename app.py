import streamlit as st
import os
import tempfile

# Importing our new OOP Architectures
from components.main import PipelineSettings, ProcessingPipeline
from components.parser import LLMParserClassifier, RegexParserClassifier, SpacyParserClassifier
from components.serialize import ResultSerializer

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="OOP Data Extractor", page_icon="📄", layout="wide")
st.title("📄 OOP-Driven PDF Information Extractor")
st.markdown("Showcasing Factory, Decorator, and Encapsulation Patterns.")

@st.cache_resource
def load_classifiers():
    return {
        "Regex Engine (Fast)": RegexParserClassifier(),
        "spaCy NLP Engine": SpacyParserClassifier(),
        "LLM Engine (Smart)": LLMParserClassifier()
    }

classifiers = load_classifiers()

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Pipeline Settings")

# Toggle for the Encapsulated Classifier
selected_engine = st.sidebar.radio(
    "Select Classification Engine:",
    options=["Regex Engine (Fast)", "spaCy NLP Engine", "LLM Engine (Smart)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Normalization Decorators")
st.sidebar.caption("Dynamically stack text processing behaviors.")

# Toggles for the Decorator Pattern
use_whitespace_remover = st.sidebar.checkbox("Remove Extra Whitespace", value=True)
use_word_converter = st.sidebar.checkbox("Convert Words to Digits", value=True)
use_punctuation_stripper = st.sidebar.checkbox("Strip Punctuation", value=False)

# ==========================================
# 3. MAIN APPLICATION LOGIC
# ==========================================
uploaded_file = st.file_uploader("Upload your scanned or printed PDF here", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_file.name
    
    st.info("File uploaded successfully. Initializing Pipeline...")

    col1, col2 = st.columns(2)

    with st.spinner('Running OOP Extraction Pipeline...'):
        try:
            active_classifier = classifiers[selected_engine]
            settings = PipelineSettings(
                use_whitespace_remover=use_whitespace_remover,
                use_word_converter=use_word_converter,
                use_punctuation_stripper=use_punctuation_stripper,
            )
            pipeline = ProcessingPipeline(active_classifier, settings)
            output = pipeline.run(temp_pdf_path)
            final_data = ResultSerializer.to_records(output.classified_results)

            # --- Render Results in UI ---
            with col1:
                st.subheader("Normalized Text Output")
                st.caption(f"Processed by: {output.processor_name}")
                st.text_area("Pipeline Text", output.normalized_text, height=400)

            with col2:
                st.subheader("Extracted JSON Output")
                if not final_data:
                    st.warning("No numerical entities were found in this document.")
                else:
                    st.json(final_data)
                    
                st.download_button(
                    label="⬇️ Download JSON File",
                    file_name=f"extracted_{uploaded_file.name}.json",
                    mime="application/json",
                    data=ResultSerializer.to_json(output.classified_results),
                    use_container_width=True
                )

        except RuntimeError as e:
            st.error(f"Pipeline Error: {str(e)}")
            st.stop()
            
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
