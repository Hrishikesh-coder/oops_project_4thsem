import streamlit as st
import json
import os
import tempfile

# Importing our new OOP Architectures
from extractor import DocumentProcessorFactory
from normalization import (
    BaseTextProcessor, 
    WhitespaceRemover, 
    PunctuationStripper, 
    WordToDigitConverter
)
from parser import RegexParserClassifier, LLMParserClassifier

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
    options=["Regex Engine (Fast)", "LLM Engine (Smart)"]
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
            # ---------------------------------------------------------
            # STEP A: THE FACTORY PATTERN
            # ---------------------------------------------------------
            # We don't instantiate the processor directly. The factory decides.
            processor = DocumentProcessorFactory.get_processor(temp_pdf_path)
            raw_text = processor.extract_text(temp_pdf_path)

            # ---------------------------------------------------------
            # STEP B: THE DECORATOR PATTERN
            # ---------------------------------------------------------
            # Start with the base component
            text_pipeline = BaseTextProcessor()
            
            # Dynamically wrap it in decorators based on UI checkboxes!
            if use_whitespace_remover:
                text_pipeline = WhitespaceRemover(text_pipeline)
            if use_word_converter:
                text_pipeline = WordToDigitConverter(text_pipeline)
            if use_punctuation_stripper:
                text_pipeline = PunctuationStripper(text_pipeline)
                
            # Execute the decorator chain
            normalized_text = text_pipeline.process(raw_text)

            # ---------------------------------------------------------
            # STEP C: ENCAPSULATED CLASSIFICATION
            # ---------------------------------------------------------
            active_classifier = classifiers[selected_engine]
            final_data = active_classifier.process(normalized_text)

            # --- Render Results in UI ---
            with col1:
                st.subheader("Normalized Text Output")
                st.caption(f"Processed by: {processor.__class__.__name__}")
                st.text_area("Pipeline Text", normalized_text, height=400)

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
                    data=json.dumps(final_data, indent=4),
                    use_container_width=True
                )

        except RuntimeError as e:
            st.error(f"Pipeline Error: {str(e)}")
            st.stop()
            
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
