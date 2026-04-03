import streamlit as st
import json
import os
from extractor import DocumentExtractor
from parser import DataParser

# Page Config
st.set_page_config(page_title="Numerical Data Extractor", layout="wide")

st.title("📄 PDF Numerical Information Extractor")
st.markdown("Upload a PDF (scanned or printed) to extract, normalize, and classify numerical data.")

# Initialize backend classes
@st.cache_resource
def load_models():
    return DocumentExtractor(), DataParser()

extractor, parser = load_models()

uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("File uploaded successfully. Processing...")

    col1, col2 = st.columns(2)

    with st.spinner('Extracting text and parsing entities...'):
        # Run Pipeline
        raw_text = extractor.process_pdf(temp_pdf_path)
        final_data = parser.extract_and_classify(raw_text)

        with col1:
            st.subheader("Raw Extracted Text")
            st.text_area("OCR / PDF Text", raw_text, height=400)

        with col2:
            st.subheader("Extracted JSON Output")
            st.json(final_data)
            
            # Download button for the JSON
            json_string = json.dumps(final_data, indent=4)
            st.download_button(
                label="Download JSON File",
                file_name=f"{uploaded_file.name}_output.json",
                mime="application/json",
                data=json_string
            )

    # Cleanup temp file
    os.remove(temp_pdf_path)