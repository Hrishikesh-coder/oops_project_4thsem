from abc import ABC, abstractmethod
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from shutil import which
from pytesseract.pytesseract import TesseractNotFoundError
from concurrent.futures import ProcessPoolExecutor

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
                text += str(page.get_text("text")) + "\n"
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
            raise RuntimeError(f"OCR dependencies are missing: {joined}. Install required system packages.")

    def extract_text(self, file_path: str) -> str:
        print("Using Scanned OCR Engine with Multiprocessing...")
        self._validate_ocr_dependencies()
        try:
            images = convert_from_path(file_path)
            # Parallel processing for OCR pages
            with ProcessPoolExecutor() as executor:
                extracted_pages = list(executor.map(pytesseract.image_to_string, images))
            return "\n".join(extracted_pages)
            
        except TesseractNotFoundError as e:
            raise RuntimeError("Tesseract binary not found in PATH.") from e
        except Exception as e:
            raise RuntimeError(f"OCR Error (Is Poppler/Tesseract installed?): {e}")

# ==========================================
# THE FACTORY
# ==========================================
class DocumentProcessorFactory:
    @staticmethod
    def get_processor(file_path: str) -> IDocumentProcessor:
        try:
            doc = fitz.open(file_path)
            sample_text = str(doc[0].get_text("text")) if len(doc) > 0 else ""
            if len(sample_text.strip()) < 50:
                return ScannedPDFProcessor()
            else:
                return DigitalPDFProcessor()
        except Exception:
            return ScannedPDFProcessor()
