from abc import ABC, abstractmethod
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from shutil import which
from pytesseract.pytesseract import TesseractNotFoundError

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
                page_text = str(page.get_text("text"))
                text += page_text + "\n"
        except Exception as e:
            raise RuntimeError(f"Failed to read digital PDF: {e}")
        return text

class ScannedPDFProcessor(IDocumentProcessor):
    @staticmethod
    def _validate_ocr_dependencies() -> None:
        missing_tools = []
        if which("tesseract") is None:
            missing_tools.append("tesseract")
        if which("pdftoppm") is None:
            missing_tools.append("pdftoppm (poppler-utils)")

        if missing_tools:
            joined = ", ".join(missing_tools)
            raise RuntimeError(
                "OCR dependencies are missing: "
                f"{joined}. Install required system packages before running scanned-PDF OCR."
            )

    def extract_text(self, file_path: str) -> str:
        print("Using Scanned OCR Engine...")
        text = ""
        self._validate_ocr_dependencies()
        try:
            images = convert_from_path(file_path)
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
        except TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract binary not found in PATH. "
                "Install tesseract-ocr and ensure the 'tesseract' command is available."
            ) from e
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
            sample_text = str(doc[0].get_text("text")) if len(doc) > 0 else ""
            
            # If text is extremely sparse, it's likely a scanned image
            if len(sample_text.strip()) < 50:
                return ScannedPDFProcessor()
            else:
                return DigitalPDFProcessor()
                
        except Exception:
            # Fallback to OCR if PyMuPDF completely fails to read the metadata
            return ScannedPDFProcessor()
