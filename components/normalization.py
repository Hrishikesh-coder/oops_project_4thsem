from abc import ABC, abstractmethod
import re

# ==========================================
# INTERFACE & BASE COMPONENT
# ==========================================
class TextProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass

class BaseTextProcessor(TextProcessor):
    """The core component that we will wrap decorators around."""
    def process(self, text: str) -> str:
        # Simply returns the raw text to start the chain
        return text 

# ==========================================
# BASE DECORATOR
# ==========================================
class TextProcessorDecorator(TextProcessor):
    """Abstract decorator that holds a reference to the next processor."""
    def __init__(self, wrapped_processor: TextProcessor):
        self._wrapped = wrapped_processor

    @abstractmethod
    def process(self, text: str) -> str:
        pass

# ==========================================
# CONCRETE DECORATORS (The Lego Bricks)
# ==========================================
class WhitespaceRemover(TextProcessorDecorator):
    def process(self, text: str) -> str:
        # First, let the inner wrapped component do its job
        processed_text = self._wrapped.process(text)
        # Then, apply this specific behavior
        print(" -> Applying WhitespaceRemover")
        return re.sub(r'\s+', ' ', processed_text).strip()

class PunctuationStripper(TextProcessorDecorator):
    def process(self, text: str) -> str:
        processed_text = self._wrapped.process(text)
        print(" -> Applying PunctuationStripper")
        return re.sub(r'[^\w\s-]', '', processed_text)

class WordToDigitConverter(TextProcessorDecorator):
    def __init__(self, wrapped_processor: TextProcessor):
        super().__init__(wrapped_processor)
        self.word_map = {
            "one": "1", "two": "2", "three": "3", "four": "4", 
            "five": "5", "six": "6", "seven": "7", "eight": "8", 
            "nine": "9", "ten": "10", "twenty": "20"
        }

    def process(self, text: str) -> str:
        processed_text = self._wrapped.process(text)
        print(" -> Applying WordToDigitConverter")
        for word, digit in self.word_map.items():
            processed_text = re.sub(rf'\b{word}\b', digit, processed_text, flags=re.IGNORECASE)
        return processed_text
