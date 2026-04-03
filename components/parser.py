import spacy
from word2number import w2n
import re

class DataParser:
    def __init__(self):
        # Load the small English NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise Exception("Please run: python -m spacy download en_core_web_sm")

    def convert_words_to_digits(self, text: str) -> str:
        """Attempts to convert text like 'twenty three' to '23'."""
        try:
            return str(w2n.word_to_num(text))
        except ValueError:
            return text # Return original if it's not a valid word-number

    def extract_and_classify(self, text: str) -> list:
        """Finds numerical entities and classifies them."""
        doc = self.nlp(text)
        extracted_data = []

        # Target specific numerical entity types in spaCy
        target_labels = ['DATE', 'TIME', 'MONEY', 'QUANTITY', 'CARDINAL', 'PERCENT']

        for ent in doc.ents:
            if ent.label_ in target_labels:
                original_text = ent.text.strip()
                normalized_value = self.convert_words_to_digits(original_text)
                
                # Basic cleanup
                normalized_value = re.sub(r'\n', ' ', normalized_value)

                extracted_data.append({
                    "original_text": original_text,
                    "normalized_value": normalized_value,
                    "Category": ent.label_
                })

        # --- Fallback Regex for strictly formatted things spaCy might miss ---
        # e.g., Phone numbers (US/India rough format)
        phone_pattern = r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        for match in re.finditer(phone_pattern, text):
            extracted_data.append({
                "original_text": match.group(),
                "normalized_value": re.sub(r'\D', '', match.group()), # Strip non-digits
                "Category": "PHONE_NUMBER"
            })

        return extracted_data