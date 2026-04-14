from pydantic import BaseModel
from typing import List

from .extractor import DocumentProcessorFactory
from .normalization import BaseTextProcessor, PunctuationStripper, TextProcessor, WhitespaceRemover, WordToDigitConverter
from .parser import BaseParserClassifier, ClassificationResult, RegexParserClassifier, SpacyParserClassifier, LLMParserClassifier

class PipelineSettings(BaseModel):
    use_whitespace_remover: bool = True
    use_word_converter: bool = True
    use_punctuation_stripper: bool = False

class PipelineOutput(BaseModel):
    processor_name: str
    raw_text: str
    normalized_text: str
    classified_results: List[ClassificationResult]

class ProcessingPipeline:
    """Coordinates extraction, normalization, and cascaded classification."""

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        # Initialize the cascade engines
        self.regex_engine = RegexParserClassifier()
        self.spacy_engine = SpacyParserClassifier()
        self.llm_engine = LLMParserClassifier()

    def _build_text_pipeline(self) -> TextProcessor:
        text_pipeline: TextProcessor = BaseTextProcessor()
        if self.settings.use_whitespace_remover:
            text_pipeline = WhitespaceRemover(text_pipeline)
        if self.settings.use_word_converter:
            text_pipeline = WordToDigitConverter(text_pipeline)
        if self.settings.use_punctuation_stripper:
            text_pipeline = PunctuationStripper(text_pipeline)
        return text_pipeline

    def run(self, pdf_path: str) -> PipelineOutput:
        processor = DocumentProcessorFactory.get_processor(pdf_path)
        raw_text = processor.extract_text(pdf_path)

        text_pipeline = self._build_text_pipeline()
        normalized_text = text_pipeline.process(raw_text)

        # --- CASCADE CLASSIFICATION (Chain of Responsibility) ---
        results: List[ClassificationResult] = []
        
        # 1. Run Fast Deterministic Parsers
        results.extend(self.regex_engine.classify(normalized_text))
        results.extend(self.spacy_engine.classify(normalized_text))

        # Deduplicate based on exact text matches
        seen = set()
        unique_results = []
        for r in results:
            if r.original_text not in seen:
                seen.add(r.original_text)
                unique_results.append(r)

        # 2. Check Confidence / Fallback Trigger
        has_numbers = any(char.isdigit() for char in normalized_text)
        if has_numbers and not unique_results:
            # Fallback to LLM if text clearly has numbers but fast engines missed them
            llm_results = self.llm_engine.classify(normalized_text)
            unique_results.extend(llm_results)

        return PipelineOutput(
            processor_name=processor.__class__.__name__,
            raw_text=raw_text,
            normalized_text=normalized_text,
            classified_results=unique_results,
        )
