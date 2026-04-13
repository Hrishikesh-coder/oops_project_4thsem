from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from typing import List

# ==========================================
# 1. THE ENCAPSULATED INTERFACE
# ==========================================
@dataclass
class ClassificationResult:
    original_text: str
    normalized_value: str
    category: str


class BaseParserClassifier(ABC):
    """Classifier contract. Implementations should only classify text."""
    @abstractmethod
    def classify(self, text: str) -> List[ClassificationResult]:
        pass


# ==========================================
# 2. PATH A: PURE REGEX IMPLEMENTATION
# ==========================================
class RegexParserClassifier(BaseParserClassifier):
    def __init__(self):
        # Pure Regex Dictionary for classification
        self.patterns = {
            # Matches standard dates and common text dates (e.g., 12/04/2026 or Jan 12th 2026)
            "DATE": r'\b(?:\d{1,2}[-/thstnd\s]+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/ \.,]+\d{1,2}[-/ \.,]+\d{2,4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            # Matches currency values with common symbols and optional magnitude words
            "MONEY": r'\b(?:USD|INR|EUR|GBP|Rs\.?|\$)\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:thousand|million|billion|lakh|crore))?\b|\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:USD|INR|EUR|GBP|dollars?|rupees?|euros?|pounds?)\b',
            # Matches percentage values
            "PERCENT": r'\b[-+]?\d+(?:\.\d+)?\s?%\b|\b[-+]?\d+(?:\.\d+)?\s?(?:percent|percentage)\b',
            # Matches common phone formats
            "PHONE_NUMBER": r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            # Matches standard Indian/International license plate formats (e.g., WB-02-AD-1234)
            "LICENSE_PLATE": r'\b[A-Z]{2}[-.\s]?\d{1,2}[-.\s]?[A-Z]{1,2}[-.\s]?\d{3,4}\b',
            # Matches standard time
            "TIME": r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b',
            # Matches age expressions
            "AGE": r'\b(?:age\s*[:=-]?\s*)?\d{1,3}\s?(?:years?\s?old|yrs?\.?\b)\b',
            # Matches temperature values
            "TEMPERATURE": r'\b[-+]?\d+(?:\.\d+)?\s?°?\s?(?:C|F|K|celsius|fahrenheit|kelvin)\b',
            # Matches distances and lengths
            "DISTANCE": r'\b\d+(?:\.\d+)?\s?(?:km|kilometers?|m|meters?|mi|miles?|ft|feet|in|inches|cm|mm)\b',
            # Matches weights/mass
            "WEIGHT": r'\b\d+(?:\.\d+)?\s?(?:kg|kilograms?|g|grams?|mg|lb|lbs|pounds?)\b',
            # Matches human height formats
            "HEIGHT": r'\b\d\s?\'\s?\d{1,2}\s?\"\b|\b\d+(?:\.\d+)?\s?(?:cm|m|ft|feet|in|inches)\s?(?:tall)?\b',
            # Matches speed values
            "SPEED": r'\b\d+(?:\.\d+)?\s?(?:km/h|kph|mph|m/s)\b',
            # Matches area units
            "AREA": r'\b\d+(?:\.\d+)?\s?(?:sq\.?\s?(?:ft|feet|m|km)|square\s?(?:feet|meters?|kilometers?)|sq\s?m|sq\s?km|sq\s?ft|acres?|hectares?)\b',
            # Matches volume/capacity values
            "VOLUME": r'\b\d+(?:\.\d+)?\s?(?:l|liters?|litres?|ml|milliliters?|millilitres?|gallons?)\b',
            # Matches directional percentage change phrases
            "PERCENTAGE_CHANGE": r'\b(?:up|down|increased\sby|decreased\sby|rise\sof|drop\sof)\s+\d+(?:\.\d+)?\s?%\b|\b[-+]\d+(?:\.\d+)?\s?%\b',
            # Matches ratio forms
            "RATIO": r'\b\d+(?:\.\d+)?\s?:\s?\d+(?:\.\d+)?\b',
            # Matches numeric ranges
            "RANGE": r'\b\d+(?:\.\d+)?\s?(?:-|to)\s?\d+(?:\.\d+)?\b',
            # Matches ordinal numbers
            "ORDINAL": r'\b\d{1,3}(?:st|nd|rd|th)\b',
            # Matches Indian PIN / common postal code formats
            "PIN_CODE": r'\b\d{6}\b|\b\d{5}(?:-\d{4})?\b',
            # Matches account-like identifiers
            "ACCOUNT_NUMBER": r'\b(?:A/C|AC|Account\s?(?:No\.?|Number)?)\s*[:#-]?\s*\d{8,18}\b',
            # Matches common invoice/order IDs
            "INVOICE_ID": r'\b(?:INV|INVOICE|ORD|ORDER)[-_/]?[A-Z0-9]{3,}\b',
            # Matches PAN and GSTIN style tax identifiers
            "TAX_ID": r'\b[A-Z]{5}\d{4}[A-Z]\b|\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Zz][A-Z0-9]\b',
            # Matches standalone quantity-like numerals with optional thousand separators
            "QUANTITY": r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            # Matches plain cardinal integers and decimals
            "CARDINAL": r'\b[-+]?\d+(?:\.\d+)?\b',
        }
        

    def classify(self, text: str) -> List[ClassificationResult]:
        results: List[ClassificationResult] = []
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                original = match.group().strip()
                results.append(
                    ClassificationResult(
                        original_text=original,
                        normalized_value=re.sub(r"[^\w\s]", "", original),
                        category=label,
                    )
                )
        return results


class SpacyParserClassifier(BaseParserClassifier):
    """spaCy-backed classifier using model entities and matcher patterns."""

    def __init__(self):
        try:
            import spacy  # type: ignore[import-not-found]
            from spacy.matcher import Matcher  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is not installed. Install requirements.txt to use the spaCy classifier."
            ) from exc

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is missing. "
                "Install requirements.txt or run: python -m spacy download en_core_web_sm"
            ) from exc

        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add(
            "PHONE_NUMBER",
            [[{"TEXT": {"REGEX": r"^\+?\d[\d\-\.\s\(\)]{7,}$"}}]],
        )
        self.matcher.add(
            "LICENSE_PLATE",
            [[{"TEXT": {"REGEX": r"^[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,2}[-\s]?\d{3,4}$"}}]],
        )

        self.entity_label_map = {
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT",
            "QUANTITY": "QUANTITY",
            "CARDINAL": "CARDINAL",
            "ORDINAL": "ORDINAL",
            # Keep direct mappings for custom labels if they are provided by
            # custom pipelines or EntityRuler configurations.
            "AGE": "AGE",
            "TEMPERATURE": "TEMPERATURE",
            "DISTANCE": "DISTANCE",
            "WEIGHT": "WEIGHT",
            "HEIGHT": "HEIGHT",
            "SPEED": "SPEED",
            "AREA": "AREA",
            "VOLUME": "VOLUME",
            "PERCENTAGE_CHANGE": "PERCENTAGE_CHANGE",
            "RATIO": "RATIO",
            "RANGE": "RANGE",
            "PIN_CODE": "PIN_CODE",
            "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
            "INVOICE_ID": "INVOICE_ID",
            "TAX_ID": "TAX_ID",
            "PHONE_NUMBER": "PHONE_NUMBER",
            "LICENSE_PLATE": "LICENSE_PLATE",
        }

        # Reuse full regex category set so spaCy path covers all project labels.
        self.regex_patterns = RegexParserClassifier().patterns

    def classify(self, text: str) -> List[ClassificationResult]:
        doc = self.nlp(text)
        results: List[ClassificationResult] = []

        for ent in doc.ents:
            mapped_category = self.entity_label_map.get(ent.label_)
            if not mapped_category:
                continue
            value = ent.text.strip()
            results.append(
                ClassificationResult(
                    original_text=value,
                    normalized_value=re.sub(r"\s+", " ", value),
                    category=mapped_category,
                )
            )

        for match_id, start, end in self.matcher(doc):
            span = doc[start:end]
            value = span.text.strip()
            category = self.nlp.vocab.strings[match_id]

            results.append(
                ClassificationResult(
                    original_text=value,
                    normalized_value=re.sub(r"[^\w\s]", "", value),
                    category=category,
                )
            )

        # Ensure the spaCy parser can emit the full regex-based label space.
        for label, pattern in self.regex_patterns.items():
            for match in re.finditer(pattern, text):
                value = match.group().strip()
                if any(r.original_text == value and r.category == label for r in results):
                    continue
                results.append(
                    ClassificationResult(
                        original_text=value,
                        normalized_value=re.sub(r"[^\w\s]", "", value),
                        category=label,
                    )
                )

        # Include standalone numerals that may not be tagged as entities.
        for token in doc:
            if token.like_num:
                value = token.text.strip()
                if any(r.original_text == value and r.category == "CARDINAL" for r in results):
                    continue
                results.append(
                    ClassificationResult(
                        original_text=value,
                        normalized_value=value,
                        category="CARDINAL",
                    )
                )

        return results


# ==========================================
# 3. PATH B: LLM IMPLEMENTATION
# ==========================================
class LLMParserClassifier(BaseParserClassifier):
    def classify(self, text: str) -> List[ClassificationResult]:
        prompt = f"""
        Task: Extract, convert, and classify numerical data from the following text.
        
        Rules:
        1. Parse: Convert any numbers written in natural language words to standard digits.
        2. Classify: Identify the type of data (e.g., DATE, PHONE_NUMBER, LICENSE_PLATE, TIME, or QUANTITY).
        
        Return ONLY a valid JSON array of dictionaries with the exact keys: 
        "original_text", "normalized_value", "category".
        
        Text to process:
        {text}
        """
        
        print("Sending classification task to LLM...")
        
        # [!] In production, replace this string with your actual API call 
        # (e.g., openai.ChatCompletion.create or a local PyTorch model call)
        mock_api_response = '''
        [
            {
                "original_text": "twenty two", 
                "normalized_value": "22", 
                "category": "QUANTITY"
            },
            {
                "original_text": "WB-24-X-9999", 
                "normalized_value": "WB24X9999", 
                "category": "LICENSE_PLATE"
            }
        ]
        '''
        
        try:
            loaded = json.loads(mock_api_response)
            return [
                ClassificationResult(
                    original_text=item.get("original_text", ""),
                    normalized_value=item.get("normalized_value", ""),
                    category=item.get("category", "UNKNOWN"),
                )
                for item in loaded
            ]
        except json.JSONDecodeError:
            print("LLM returned invalid JSON.")
            return []
