from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import json
import re
import os
from typing import List
from openai import OpenAI

SUPPORTED_CATEGORY_LABELS = [
    "DATE",
    "YEAR",
    "MONEY",
    "PERCENT",
    "PHONE_NUMBER",
    "EMAIL",
    "URL",
    "LICENSE_PLATE",
    "TIME",
    "AGE",
    "TEMPERATURE",
    "DISTANCE",
    "WEIGHT",
    "HEIGHT",
    "SPEED",
    "AREA",
    "VOLUME",
    "PERCENTAGE_CHANGE",
    "RATIO",
    "RANGE",
    "ORDINAL",
    "PIN_CODE",
    "ACCOUNT_NUMBER",
    "INVOICE_ID",
    "TAX_ID",
    "LOCATION_DATA",
    "QUANTITY",
    "CARDINAL",
    "ACADEMIC_ROLL_NUMBER",
    "AADHAAR_NUMBER",
    "IFSC_CODE",
    "CREDIT_CARD",
    "STOCK_TICKER",
    "IPV4_ADDRESS",
    "MAC_ADDRESS",
    "UUID"
]

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

    @abstractmethod
    def supported_categories(self) -> List[str]:
        """Returns the category labels this classifier can emit."""
        pass


# ==========================================
# 2. PATH A: PURE REGEX IMPLEMENTATION
# ==========================================
class RegexParserClassifier(BaseParserClassifier):
    def __init__(self):
        # Pure Regex Dictionary for classification
        # ORDER MATTERS: Most specific/priority categories MUST be at the top.
        self.patterns = {
            # --- CONTACT & WEB ---
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "URL": r'\bhttps?://[^\s/$.?#].[^\s]*\b',
            "PHONE_NUMBER": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',

            # --- IDENTIFIERS & REGIONAL CODES ---
            "TAX_ID": r'\b[A-Z]{5}\d{4}[A-Z]\b|\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Zz][A-Z0-9]\b',
            "LICENSE_PLATE": r'\b[A-Z]{2}[-.\s]?\d{1,2}[-.\s]?[A-Z]{1,2}[-.\s]?\d{3,4}\b',
            "IFSC_CODE": r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            "AADHAAR_NUMBER": r'\b\d{4}[-\s]\d{4}[-\s]\d{4}\b',
            
            # --- FINANCIAL & TRADING ---
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
            "YEAR": r'\b(19|20)\d{2}\b',
            "TIME": r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b',
            "AGE": r'\b(?:age\s*[:=-]?\s*)?\d{1,3}\s?(?:years?\s?old|yrs?\.?\b)\b',
            "TEMPERATURE": r'\b[-+]?\d+(?:\.\d+)?\s?°?\s?(?:C|F|K|celsius|fahrenheit|kelvin)\b',
            "PERCENT": r'\b[-+]?\d+(?:\.\d+)?\s?%\b|\b[-+]?\d+(?:\.\d+)?\s?(?:percent|percentage)\b',
            "PERCENTAGE_CHANGE": r'\b(?:up|down|increased\sby|decreased\sby|rise\sof|drop\sof)\s+\d+(?:\.\d+)?\s?%\b|\b[-+]\d+(?:\.\d+)?\s?%\b',
            
            # --- MEASUREMENTS ---
            "DISTANCE": r'\b\d+(?:\.\d+)?\s?(?:km|kilometers?|m|meters?|mi|miles?|ft|feet|in|inches|cm|mm)\b',
            "WEIGHT": r'\b\d+(?:\.\d+)?\s?(?:kg|kilograms?|g|grams?|mg|lb|lbs|pounds?)\b',
            "HEIGHT": r'\b\d\s?\'\s?\d{1,2}\s?\"\b|\b\d+(?:\.\d+)?\s?(?:cm|m|ft|feet|in|inches)\s?(?:tall)?\b',
            "SPEED": r'\b\d+(?:\.\d+)?\s?(?:km/h|kph|mph|m/s)\b',
            "AREA": r'\b\d+(?:\.\d+)?\s?(?:sq\.?\s?(?:ft|feet|m|km)|square\s?(?:feet|meters?|kilometers?)|sq\s?m|sq\s?km|sq\s?ft|acres?|hectares?)\b',
            "VOLUME": r'\b\d+(?:\.\d+)?\s?(?:l|liters?|litres?|ml|milliliters?|millilitres?|gallons?)\b',
            
            # --- MISC ---
            "RATIO": r'\b\d+(?:\.\d+)?\s?:\s?\d+(?:\.\d+)?\b',
            "RANGE": r'\b\d+(?:\.\d+)?\s?(?:-|to)\s?\d+(?:\.\d+)?\b',
            "ORDINAL": r'\b\d{1,3}(?:st|nd|rd|th)\b',
            "LOCATION_DATA": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+(?:City|District|State|County|Province|Village|Town)\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+(?:City|District|State|County|Province|Village|Town)\b',
            
            # --- FALLBACKS (Lowest Priority) ---
            "ACADEMIC_ROLL_NUMBER": r'\b(?:Roll\s?(?:No\.?|Number)?|UID|ID)\s*[:#-]?\s*\d{8,12}\b',
            "QUANTITY": r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            "PIN_CODE": r'\b\d{6}\b',
            "CARDINAL": r'\b[-+]?\d+(?:\.\d+)?\b',
        }

    def classify(self, text: str) -> List[ClassificationResult]:
        results: List[ClassificationResult] = []
        matched_indices = set()  # Fixed: Tracks character indices that have already been matched

        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                start, end = match.start(), match.end()
                
                # Check if this substring is already part of a higher-priority match
                if any(i in matched_indices for i in range(start, end)):
                    continue
                
                original = match.group().strip()
                
                # Clean up logic based on category
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
                
                # Mark these character indices as consumed
                matched_indices.update(range(start, end))
                
        return results

    def supported_categories(self) -> List[str]:
        return list(self.patterns.keys())


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

        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("PHONE_NUMBER", [[{"TEXT": {"REGEX": r"^\+?\d[\d\-\.\s\(\)]{7,}$"}}]])
        self.matcher.add("LICENSE_PLATE", [[{"TEXT": {"REGEX": r"^[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,2}[-\s]?\d{3,4}$"}}]])
        self.matcher.add("IPV4_ADDRESS", [[{"TEXT": {"REGEX": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"}}]])
        self.matcher.add("MAC_ADDRESS", [[{"TEXT": {"REGEX": r"^(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})$"}}]])
        self.matcher.add("PAN_CARD", [[{"TEXT": {"REGEX": r"^[A-Z]{5}[0-9]{4}[A-Z]$"}}]])
        self.matcher.add("IFSC_CODE", [[{"TEXT": {"REGEX": r"^[A-Z]{4}0[A-Z0-9]{6}$"}}]])

        self.entity_label_map = {
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT",
            "QUANTITY": "QUANTITY",
            "CARDINAL": "CARDINAL",
            "GPE": "LOCATION_DATA",
            "LOC": "LOCATION_DATA",
        }

        self.regex_patterns = RegexParserClassifier().patterns
        # Common tech terms misclassified by spaCy as GPE/LOC
        self.tech_whitelist = {"Nextjs", "React", "FastAPI", "Python", "Java", "TypeScript", "JavaScript", "GeoJSON", "GitHub", "WebGIS", "Leaflet"}

    def classify(self, text: str) -> List[ClassificationResult]:
        doc = self.nlp(text)
        results: List[ClassificationResult] = []
        matched_indices = set() # Optional: also apply index tracking here if needed

        # 1. Check Standard NLP Entities
        for ent in doc.ents:
            if ent.text.strip() in self.tech_whitelist:
                continue

            mapped_category = self.entity_label_map.get(ent.label_)
            if not mapped_category:
                continue
            
            value = ent.text.strip()
            # Basic noise filter for location data
            if mapped_category == "LOCATION_DATA" and (len(value) < 3 or any(c.islower() for c in value) and any(c.isupper() for c in value[1:])):
                # If it looks like CamelCase or is too short, it's likely not a location
                continue

            results.append(
                ClassificationResult(
                    original_text=value,
                    normalized_value=re.sub(r"\s+", " ", value),
                    category=mapped_category,
                )
            )
            matched_indices.update(range(ent.start_char, ent.end_char))

        # 2. Check Strict Regex Matchers overlayed on spaCy tokens
        for match_id, start, end in self.matcher(doc):
            span = doc[start:end]
            
            # Skip if already captured by standard entities
            if any(i in matched_indices for i in range(span.start_char, span.end_char)):
                continue
                
            value = span.text.strip()
            category = self.nlp.vocab.strings[match_id]
            
            results.append(
                ClassificationResult(
                    original_text=value,
                    normalized_value=re.sub(r"[^\w\s\-\:\.]", "", value),
                    category=category,
                )
            )
            matched_indices.update(range(span.start_char, span.end_char))

        # 3. Include standalone numerals that may not be tagged as entities
        for token in doc:
            if token.like_num:
                # Avoid duplicating items already found by the entity or matcher step
                if any(i in matched_indices for i in range(token.idx, token.idx + len(token.text))):
                    continue
                    
                value = token.text.strip()
                results.append(
                    ClassificationResult(
                        original_text=value,
                        normalized_value=value,
                        category="CARDINAL",
                    )
                )

        return results

    def supported_categories(self) -> List[str]:
        return list(self.regex_patterns.keys())


# ==========================================
# 3. PATH B: LLM IMPLEMENTATION (Fallback)
# ==========================================
class LLMParserClassifier(BaseParserClassifier):
    def __init__(self):
        self.api_key = os.getenv("LLAMA_API_KEY")
        self.base_url = os.getenv("LLAMA_BASE_URL", "https://api.groq.com/openai/v1")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = "llama-3.3-70b-versatile"

    def classify(self, text: str) -> List[ClassificationResult]:
        if not self.client:
            print("[LLM Classifier] No API Key found. Skipping...")
            return []

        # Fixed: Completely overhauled prompt to force NER extraction rather than text summarization/cleaning.
        prompt = f"""
        You are a strict Named Entity Recognition (NER) system.
        Task: Extract specific entities from the text below and classify them.
        
        Rules:
        1. DO NOT modify, summarize, clean, or translate the original text.
        2. Extract the EXACT substring exactly as it appears in the text provided.
        3. Classify each extracted string into one of these strict categories: {SUPPORTED_CATEGORY_LABELS}
        
        Return ONLY a valid JSON array of dictionaries with the exact keys: 
        "original_text", "normalized_value", "category". Do not include markdown code blocks or any other text.
        
        Text to process:
        {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional data extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"} if "openai" in self.base_url.lower() else None
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown blocks if LLM ignored instructions
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            # Sometimes LLM wraps the array in a "results" or "entities" key
            loaded = json.loads(content)
            if isinstance(loaded, dict):
                # Try to find the list in common keys
                for key in ["results", "entities", "data"]:
                    if key in loaded and isinstance(loaded[key], list):
                        loaded = loaded[key]
                        break
                if isinstance(loaded, dict): # Still a dict? 
                    loaded = [loaded] # Maybe it's just one object

            return [
                ClassificationResult(
                    original_text=item.get("original_text", ""),
                    normalized_value=item.get("normalized_value", ""),
                    category=item.get("category", "UNKNOWN"),
                )
                for item in loaded if isinstance(item, dict)
            ]
        except Exception as e:
            print(f"[LLM Classifier Error] Failed: {e}")
            return []

    def supported_categories(self) -> List[str]:
        return SUPPORTED_CATEGORY_LABELS