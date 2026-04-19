import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from components.main import PipelineSettings, ProcessingPipeline
from components.serialize import ResultSerializer

# Load environment variables (Required for the LLM API Key)
load_dotenv()

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="OOP Data Extractor", page_icon="📄", layout="wide")
st.title("📄 OOP-Driven PDF Information Extractor")
st.markdown("Features dynamic PDF Factory, Cascading Classifiers (Regex -> NLP -> LLM), and Normalization Decorators.")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Pipeline Settings")

st.sidebar.subheader("🤖 LLM Fallback (Llama 3.3)")
st.sidebar.caption("Use LLM to reconstruct messy OCR/PDF extractions.")
use_llm_extraction = st.sidebar.checkbox("Enable Extraction LLM Fallback", value=True)

st.sidebar.subheader("🤖 AI Classification")
st.sidebar.caption("Use LLM to refine entity classification.")
auto_llm_fallback = st.sidebar.checkbox("Auto-AI Fallback", value=True, help="Automatically trigger AI if Regex/NER results look incomplete.")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Normalization Decorators")
st.sidebar.caption("Dynamically stack text processing behaviors.")
use_whitespace_remover = st.sidebar.checkbox("Remove Extra Whitespace", value=True)
use_word_converter = st.sidebar.checkbox("Convert Words to Digits", value=True)
use_punctuation_stripper = st.sidebar.checkbox("Strip Punctuation", value=False)

# ==========================================
# 3. MAIN APPLICATION LOGIC
# ==========================================
if "force_ai" not in st.session_state:
    st.session_state.force_ai = False

uploaded_file = st.file_uploader("Upload your scanned or printed PDF here", type=["pdf"])

if uploaded_file is not None:
    # Reset force_ai if a new file is uploaded
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.force_ai = False
        st.session_state.last_uploaded = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_file.name
    
    st.info("File uploaded successfully. Initializing Cascade Pipeline...")

    col1, col2 = st.columns(2)

    with st.spinner('Running Extraction & Classification...'):
        try:
            settings = PipelineSettings(
                use_whitespace_remover=use_whitespace_remover,
                use_word_converter=use_word_converter,
                use_punctuation_stripper=use_punctuation_stripper,
                use_llm_extraction=use_llm_extraction,
                auto_llm_fallback=auto_llm_fallback,
                force_llm_classification=st.session_state.force_ai
            )
            # Pipeline now manages its own cascading engines
            pipeline = ProcessingPipeline(settings)
            output = pipeline.run(temp_pdf_path)
            
            final_data = ResultSerializer.to_records(output.classified_results)

            # --- Render Results in UI ---
            with col1:
                st.subheader("Normalized Text Output")
                st.caption(f"Extraction Engine Used: {output.processor_name}")
                st.text_area("Pipeline Text", output.normalized_text, height=400)

            with col2:
                st.subheader("Extracted JSON Output")
                
                # Manual AI Refinement Button
                if not st.session_state.force_ai:
                    if st.button("✨ Run AI Classification Refinement", help="Manually trigger Llama 3.3 to find missed entities."):
                        st.session_state.force_ai = True
                        st.rerun()
                else:
                    st.success("AI Refinement Applied!")

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