import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import os

class DocumentExtractor:
    def __init__(self, tesseract_cmd_path=None):
        # Update this path if you are on Windows (e.g., r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

    def process_pdf(self, file_path: str) -> str:
        """Extracts text, automatically choosing between digital and scanned methods."""
        print(f"Extracting text from: {file_path}")
        digital_text = self._extract_via_lib(file_path)
        
        # If the PDF yields very few characters, it's likely a scanned image
        if len(digital_text.strip()) < 50:
            print("Detected scanned document. Falling back to OCR...")
            return self._extract_via_ocr(file_path)
        
        print("Detected printed/digital document.")
        return digital_text

    def _extract_via_lib(self, file_path: str) -> str:
        """Extracts text directly from digital PDFs."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text

    def _extract_via_ocr(self, file_path: str) -> str:
        """Converts PDF pages to images and runs Tesseract OCR."""
        text = ""
        try:
            images = convert_from_path(file_path)
            for i, image in enumerate(images):
                text += pytesseract.image_to_string(image) + "\n"
        except Exception as e:
            print(f"OCR Error (Is Poppler installed?): {e}")
        return text