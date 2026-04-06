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
            # Matches common phone formats
            "PHONE_NUMBER": r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            # Matches standard Indian/International license plate formats (e.g., WB-02-AD-1234)
            "LICENSE_PLATE": r'\b[A-Z]{2}[-.\s]?\d{2}[-.\s]?[A-Z]{1,2}[-.\s]?\d{4}\b', 
            # Matches standard time
            "TIME": r'\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b'
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
        }

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

        for _, start, end in self.matcher(doc):
            span = doc[start:end]
            value = span.text.strip()
            if re.fullmatch(r"\+?\d[\d\-\.\s\(\)]{7,}", value):
                category = "PHONE_NUMBER"
            else:
                category = "LICENSE_PLATE"

            results.append(
                ClassificationResult(
                    original_text=value,
                    normalized_value=re.sub(r"[^\w\s]", "", value),
                    category=category,
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
