from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import json
import re
from typing import List

# ==========================================
# 1. THE ENCAPSULATED INTERFACE
# ==========================================
class ClassificationResult(BaseModel):
    original_text: str
    normalized_value: str
    category: str = Field(default="UNKNOWN", description="The identified entity type")


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
        # ORDER MATTERS: Most specific categories MUST be at the top.
        self.patterns = {
            # --- IDENTIFIERS & REGIONAL CODES ---
            "TAX_ID": r'\b[A-Z]{5}\d{4}[A-Z]\b|\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Zz][A-Z0-9]\b',
            "AADHAAR_NUMBER": r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            "LICENSE_PLATE": r'\b[A-Z]{2}[-.\s]?\d{1,2}[-.\s]?[A-Z]{1,2}[-.\s]?\d{3,4}\b',
            "PIN_CODE": r'\b\d{6}\b|\b\d{5}(?:-\d{4})?\b',
            "ACADEMIC_ROLL_NUMBER": r'\b\d{10,12}\b',
            
            # --- FINANCIAL & TRADING ---
            "IFSC_CODE": r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            "ACCOUNT_NUMBER": r'\b(?:A/C|AC|Account\s?(?:No\.?|Number)?)\s*[:#-]?\s*\d{8,18}\b',
            "CREDIT_CARD": r'\b(?:\d[ -]*?){13,16}\b|\b\*{4}[\s-]?\*{4}[\s-]?\*{4}[\s-]?\d{4}\b',
            "STOCK_TICKER": r'\b(?:NYSE|NASDAQ|NSE|BSE):\s?[A-Z]{1,10}\b|\$[A-Z]{1,5}\b',
            "MONEY": r'\b(?:USD|INR|EUR|GBP|Rs\.?|₹|\$|€|£)\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:k|m|b|thousand|million|billion|lakh|crore))?\b|\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:USD|INR|EUR|GBP|dollars?|rupees?|euros?|pounds?)\b',
            "INVOICE_ID": r'\b(?:INV|INVOICE|ORD|ORDER)[-_/]?[A-Z0-9]{3,}\b',

            # --- TECHNICAL & NETWORK ---
            "IPV4_ADDRESS": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "MAC_ADDRESS": r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
            "UUID": r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',

            # --- DATETIME & CONTACT ---
            "DATE": r'\b(?:\d{1,2}[-/thstnd\s]+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/ \.,]+\d{1,2}[-/ \.,]+\d{2,4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            "TIME": r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b',
            "PHONE_NUMBER": r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',

            # --- MEASUREMENTS & PHYSICAL DATA ---
            "SPEED": r'\b\d+(?:\.\d+)?\s?(?:km/h|kph|mph|m/s)\b',
            "AREA": r'\b\d+(?:\.\d+)?\s?(?:sq\.?\s?(?:ft|feet|m|km)|square\s?(?:feet|meters?|kilometers?)|sq\s?m|sq\s?km|sq\s?ft|acres?|hectares?)\b',
            "VOLUME": r'\b\d+(?:\.\d+)?\s?(?:l|liters?|litres?|ml|milliliters?|millilitres?|gallons?)\b',
            "WEIGHT": r'\b\d+(?:\.\d+)?\s?(?:kg|kilograms?|g|grams?|mg|lb|lbs|pounds?)\b',
            "DISTANCE": r'\b\d+(?:\.\d+)?\s?(?:km|kilometers?|m|meters?|mi|miles?|ft|feet|in|inches|cm|mm)\b',
            "TEMPERATURE": r'\b[-+]?\d+(?:\.\d+)?\s?°?\s?(?:C|F|K|celsius|fahrenheit|kelvin)\b',
            "HEIGHT": r'\b\d\s?\'\s?\d{1,2}\s?\"\b|\b\d+(?:\.\d+)?\s?(?:cm|m|ft|feet|in|inches)\s?(?:tall)?\b',
            "AGE": r'\b(?:age\s*[:=-]?\s*)?\d{1,3}\s?(?:years?\s?old|yrs?\.?\b)\b',

            # --- MATHEMATICAL & RELATIONAL ---
            "PERCENTAGE_CHANGE": r'\b(?:up|down|increased\sby|decreased\sby|rise\sof|drop\sof)\s+\d+(?:\.\d+)?\s?%\b|\b[-+]\d+(?:\.\d+)?\s?%\b',
            "PERCENT": r'\b[-+]?\d+(?:\.\d+)?\s?%\b|\b[-+]?\d+(?:\.\d+)?\s?(?:percent|percentage)\b',
            "RATIO": r'\b\d+(?:\.\d+)?\s?:\s?\d+(?:\.\d+)?\b',
            "RANGE": r'\b\d+(?:\.\d+)?\s?(?:-|to)\s?\d+(?:\.\d+)?\b',
            "ORDINAL": r'\b\d{1,3}(?:st|nd|rd|th)\b',

            # --- CATCH-ALL NUMERICS ---
            "CARDINAL": r'\b[-+]?\d+(?:,\d{3})*(?:\.\d+)?\b',
        }

    def classify(self, text: str) -> List[ClassificationResult]:
        results: List[ClassificationResult] = []
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                original = match.group().strip()
                # Use a specific clean-up based on category (preserve dots for IPs, colons for MACs)
                if label in ["IPV4_ADDRESS", "MAC_ADDRESS", "TIME"]:
                    normalized = re.sub(r"[^\w\s\-\:\.]", "", original)
                else:
                    normalized = re.sub(r"[^\w\s]", "", original)
                    
                results.append(
                    ClassificationResult(
                        original_text=original,
                        normalized_value=normalized,
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
                "Run: python -m spacy download en_core_web_sm"
            ) from exc

        # Initialize Matcher for specific, strict formatting
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("PHONE_NUMBER", [[{"TEXT": {"REGEX": r"^\+?\d[\d\-\.\s\(\)]{7,}$"}}]])
        self.matcher.add("LICENSE_PLATE", [[{"TEXT": {"REGEX": r"^[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,2}[-\s]?\d{3,4}$"}}]])
        self.matcher.add("IPV4_ADDRESS", [[{"TEXT": {"REGEX": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"}}]])
        self.matcher.add("MAC_ADDRESS", [[{"TEXT": {"REGEX": r"^(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})$"}}]])
        self.matcher.add("PAN_CARD", [[{"TEXT": {"REGEX": r"^[A-Z]{5}[0-9]{4}[A-Z]$"}}]])
        self.matcher.add("IFSC_CODE", [[{"TEXT": {"REGEX": r"^[A-Z]{4}0[A-Z0-9]{6}$"}}]])

        # Map spaCy's internal tags to our standard JSON output
        self.entity_label_map = {
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "CURRENCY",
            "PERCENT": "PERCENTAGE",
            "QUANTITY": "MEASUREMENT",
            "CARDINAL": "CARDINAL",
            "ORG": "ORGANIZATION_ID",
            "GPE": "LOCATION_DATA",
            "LOC": "LOCATION_DATA",
            "FAC": "FACILITY_NAME",
            "LAW": "LEGAL_REFERENCE"
        }

    def classify(self, text: str) -> List[ClassificationResult]:
        doc = self.nlp(text)
        results: List[ClassificationResult] = []

        # 1. Check Standard NLP Entities
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

        # 2. Check Strict Regex Matchers overlayed on spaCy tokens
        for match_id, start, end in self.matcher(doc):
            span = doc[start:end]
            value = span.text.strip()
            category = self.nlp.vocab.strings[match_id]
            
            results.append(
                ClassificationResult(
                    original_text=value,
                    normalized_value=re.sub(r"[^\w\s\-\:\.]", "", value),
                    category=category,
                )
            )

        # 3. Include standalone numerals that may not be tagged as entities
        for token in doc:
            if token.like_num:
                value = token.text.strip()
                # Avoid duplicating items already found by the entity or matcher step
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
# 3. PATH B: LLM IMPLEMENTATION (Fallback)
# ==========================================
class LLMParserClassifier(BaseParserClassifier):
    def classify(self, text: str) -> List[ClassificationResult]:
        # [!] In production, hook this up to the OpenAI / Anthropic / Gemini SDK
        print("Low confidence detected. Sending task to LLM API...")
        
        # Mock response to simulate an LLM reading the text
        mock_api_response = '''
        [
            {
                "original_text": "twenty two", 
                "normalized_value": "22", 
                "category": "QUANTITY"
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
