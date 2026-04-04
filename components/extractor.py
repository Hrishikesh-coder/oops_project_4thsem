from abc import ABC, abstractmethod
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path

# ==========================================
# INTERFACE (The Contract)
# ==========================================
class IDocumentProcessor(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extracts text from the given file path."""
        pass

# ==========================================
# CONCRETE IMPLEMENTATIONS
# ==========================================
class DigitalPDFProcessor(IDocumentProcessor):
    def extract_text(self, file_path: str) -> str:
        print("Using Digital PDF Engine...")
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
        except Exception as e:
            raise RuntimeError(f"Failed to read digital PDF: {e}")
        return text

class ScannedPDFProcessor(IDocumentProcessor):
    def extract_text(self, file_path: str) -> str:
        print("Using Scanned OCR Engine...")
        text = ""
        try:
            images = convert_from_path(file_path)
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
        except Exception as e:
            raise RuntimeError(f"OCR Error (Is Poppler/Tesseract installed?): {e}")
        return text

# ==========================================
# THE FACTORY
# ==========================================
class DocumentProcessorFactory:
    @staticmethod
    def get_processor(file_path: str) -> IDocumentProcessor:
        """
        Inspects the file and returns the appropriate processor object dynamically.
        """
        # Basic inspection to determine if scanned or digital
        try:
            doc = fitz.open(file_path)
            # Sample the first page to check text density
            sample_text = doc[0].get_text() if len(doc) > 0 else ""
            
            # If text is extremely sparse, it's likely a scanned image
            if len(sample_text.strip()) < 50:
                return ScannedPDFProcessor()
            else:
                return DigitalPDFProcessor()
                
        except Exception:
            # Fallback to OCR if PyMuPDF completely fails to read the metadata
            return ScannedPDFProcessor()
