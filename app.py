import streamlit as st
import json
import os
import tempfile
from extractor import DocumentExtractor
from parser import RegexParserClassifier, LLMParserClassifier

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="Numerical Data Extractor", 
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Numerical Information Extractor")
st.markdown("Upload a PDF to extract, normalize, and classify numerical data.")

# ==========================================
# 2. MODEL INITIALIZATION (CACHED)
# ==========================================
# We cache the models so they don't reload on every UI interaction
@st.cache_resource
def load_extractor():
    return DocumentExtractor()

@st.cache_resource
def load_parsers():
    return {
        "Regex Engine (Fast)": RegexParserClassifier(),
        "LLM Engine (Smart)": LLMParserClassifier()
    }

extractor = load_extractor()
parsers = load_parsers()

# Sidebar for configuration
st.sidebar.header("⚙️ Pipeline Settings")
selected_engine = st.sidebar.radio(
    "Select Extraction Engine:",
    options=["Regex Engine (Fast)", "LLM Engine (Smart)"],
    help="Regex is faster and uses standard patterns. LLM handles complex, messy natural language better."
)

# Set the active parser based on user selection
active_parser = parsers[selected_engine]

# ==========================================
# 3. MAIN APPLICATION LOGIC
# ==========================================
uploaded_file = st.file_uploader("Upload your scanned or printed PDF here", type=["pdf"])

if uploaded_file is not None:
    # FIX 1: Concurrency-Safe Temporary File Handling
    # This ensures multiple users uploading simultaneously won't overwrite each other's files.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_file.name
    
    st.info(f"File uploaded successfully. Processing with **{selected_engine}**...")

    # Set up UI columns for side-by-side display
    col1, col2 = st.columns(2)

    with st.spinner('Extracting text and classifying entities...'):
        try:
            # Step 1: Extract Text (Throws RuntimeError if OCR fails)
            raw_text = extractor.process_pdf(temp_pdf_path)
            
            # Step 2: Parse and Classify using the selected encapsulated engine
            final_data = active_parser.process(raw_text)

            # Step 3: Render Results in UI
            with col1:
                st.subheader("Raw Extracted Text")
                st.text_area("OCR / PDF Text Output", raw_text, height=400)

            with col2:
                st.subheader("Extracted JSON Output")
                if not final_data:
                    st.warning("No numerical entities were found in this document.")
                else:
                    st.json(final_data)
                    
                # Download button for the JSON results
                json_string = json.dumps(final_data, indent=4)
                st.download_button(
                    label="⬇️ Download JSON File",
                    file_name=f"extracted_{uploaded_file.name}.json",
                    mime="application/json",
                    data=json_string,
                    use_container_width=True
                )

        # If Poppler or Tesseract crashes in the extractor, it displays cleanly here.
        except RuntimeError as e:
            st.error(f"Pipeline Error: {str(e)}")
            st.stop()
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.stop()
            
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
