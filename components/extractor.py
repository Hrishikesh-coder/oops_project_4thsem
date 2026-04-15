import os
from abc import ABC, abstractmethod
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from shutil import which
from pytesseract.pytesseract import TesseractNotFoundError
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv

# Import the LLM fallback logic we created earlier
from .llm_fallback import LLMExtractionFallback

# Load environment variables for the API Key
load_dotenv()

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
# DECORATOR (The LLM Fallback)
# ==========================================
class LLMFallbackDecorator(IDocumentProcessor):
    """Wraps any IDocumentProcessor with a Llama 3.3 evaluation step."""
    
    def __init__(self, base_processor: IDocumentProcessor):
        self.base_processor = base_processor
        
        # Initialize the LLM Agent
        api_key = os.getenv("LLAMA_API_KEY")
        base_url = os.getenv("LLAMA_BASE_URL", "https://api.groq.com/openai/v1")
        self.llm_agent = LLMExtractionFallback(api_key=api_key, base_url=base_url)

    def _is_extraction_poor(self, text: str) -> bool:
        """Heuristic to determine if the extracted text needs LLM cleanup."""
        if not text or len(text.strip()) == 0:
            return True
            
        # Check for high density of non-alphanumeric characters (common in bad OCR/PDF extraction)
        text_len = max(1, len(text))
        garbage_ratio = sum(1 for char in text if not char.isalnum() and not char.isspace()) / text_len
        
        return garbage_ratio > 0.25 # Trigger LLM if > 25% of text is special characters

    def extract_text(self, file_path: str) -> str:
        # 1. Run the standard extraction (Digital or Scanned)
        raw_text = self.base_processor.extract_text(file_path)

        # 2. Evaluate the output
        if self._is_extraction_poor(raw_text):
            print("[LLM Fallback] Poor extraction quality detected. Triggering Llama 3.3...")
            cleaned_text = self.llm_agent.extract_and_clean(raw_text)
            
            if cleaned_text:
                return cleaned_text
                
        return raw_text

# ==========================================
# THE FACTORY
# ==========================================
class DocumentProcessorFactory:
    @staticmethod
    def get_processor(file_path: str, use_llm: bool = True) -> IDocumentProcessor:
        """
        Determines the best extraction engine.
        Now includes an option to wrap the engine in the LLM fallback decorator.
        """
        try:
            doc = fitz.open(file_path)
            sample_text = str(doc[0].get_text("text")) if len(doc) > 0 else ""
            
            if len(sample_text.strip()) < 50:
                base_processor = ScannedPDFProcessor()
            else:
                base_processor = DigitalPDFProcessor()
        except Exception:
            base_processor = ScannedPDFProcessor()

        # Wrap the chosen processor with the LLM Fallback if requested
        if use_llm:
            return LLMFallbackDecorator(base_processor)
            
        return base_processor