from abc import ABC, abstractmethod
import re
import json

# ==========================================
# 1. THE ENCAPSULATED INTERFACE
# ==========================================
class BaseParserClassifier(ABC):
    """
    A single block that handles BOTH converting words to digits 
    AND classifying the text into entities.
    """
    @abstractmethod
    def process(self, raw_text: str) -> list:
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
        
        # A lightweight Regex word-to-digit map (to replace w2n for simple cases)
        self.word_map = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", 
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "twenty": "20", "thirty": "30" # Expand this dictionary as needed
        }

    def _normalize_text(self, text: str) -> str:
        """Internal method to convert words to digits using regex substitution."""
        normalized = text
        for word, digit in self.word_map.items():
            # \b ensures we only replace whole words (e.g., won't turn 'tone' into 't1')
            normalized = re.sub(rf'\b{word}\b', digit, normalized, flags=re.IGNORECASE)
        return normalized

    def process(self, raw_text: str) -> list:
        # Step 1: Parse and normalize in memory
        normalized_text = self._normalize_text(raw_text)
        
        # Step 2: Classify using the pure regex rules
        results = []
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, normalized_text):
                original = match.group().strip()
                results.append({
                    "original_text": original,
                    # Basic cleanup for the normalised value
                    "normalized_value": re.sub(r'[^\w\s]', '', original), 
                    "Category": label
                })
        return results


# ==========================================
# 3. PATH B: LLM IMPLEMENTATION
# ==========================================
class LLMParserClassifier(BaseParserClassifier):
    def process(self, raw_text: str) -> list:
        # We delegate BOTH the word-to-number parsing and the classification 
        # to the LLM via a strict prompt.
        prompt = f"""
        Task: Extract, convert, and classify numerical data from the following text.
        
        Rules:
        1. Parse: Convert any numbers written in natural language words to standard digits.
        2. Classify: Identify the type of data (e.g., DATE, PHONE_NUMBER, LICENSE_PLATE, TIME, or QUANTITY).
        
        Return ONLY a valid JSON array of dictionaries with the exact keys: 
        "original_text", "normalized_value", "Category".
        
        Text to process:
        {raw_text}
        """
        
        print("Sending unified parsing and classification task to LLM...")
        
        # [!] In production, replace this string with your actual API call 
        # (e.g., openai.ChatCompletion.create or a local PyTorch model call)
        mock_api_response = '''
        [
            {
                "original_text": "twenty two", 
                "normalized_value": "22", 
                "Category": "QUANTITY"
            },
            {
                "original_text": "WB-24-X-9999", 
                "normalized_value": "WB24X9999", 
                "Category": "LICENSE_PLATE"
            }
        ]
        '''
        
        try:
            return json.loads(mock_api_response)
        except json.JSONDecodeError:
            print("LLM returned invalid JSON.")
            return []
