from pydantic import BaseModel
from typing import List

from .extractor import DocumentProcessorFactory
from .normalization import BaseTextProcessor, PunctuationStripper, TextProcessor, WhitespaceRemover, WordToDigitConverter
from .parser import BaseParserClassifier, ClassificationResult, RegexParserClassifier, SpacyParserClassifier, LLMParserClassifier

class PipelineSettings(BaseModel):
    use_whitespace_remover: bool = True
    use_word_converter: bool = True
    use_punctuation_stripper: bool = False
    # Added setting to control the extraction-level LLM fallback
    use_llm_extraction: bool = True 
    # New: Control for classification-level LLM
    auto_llm_fallback: bool = True
    force_llm_classification: bool = False
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
        # 1. Pass the extraction LLM setting to the Factory
        processor = DocumentProcessorFactory.get_processor(
            pdf_path, 
            use_llm=self.settings.use_llm_extraction
        )
        raw_text = processor.extract_text(pdf_path)

        text_pipeline = self._build_text_pipeline()
        normalized_text = text_pipeline.process(raw_text)

        # --- CASCADE CLASSIFICATION (Chain of Responsibility) ---
        results: List[ClassificationResult] = []
        
        # Run Fast Deterministic Parsers
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
        
        # Heuristic for "bad performance"
        # 1. Manual Force (User clicked the button)
        # 2. Automatic Fallback (No results found despite numbers being present)
        # 3. Automatic Fallback (Low density of extracted entities vs expected)
        
        should_run_llm = self.settings.force_llm_classification
        
        if not should_run_llm and self.settings.auto_llm_fallback:
            # If we found nothing but there are digits
            if has_numbers and not unique_results:
                should_run_llm = True
            
            # If result count is suspiciously low (e.g., less than 1 entity per 100 chars of numeric text)
            # This is a very rough heuristic for "bad performance"
            num_digits = sum(1 for c in normalized_text if c.isdigit())
            if has_numbers and len(unique_results) < (num_digits / 15): # Heuristic: expect at least 1 entity per 15 digits
                should_run_llm = True

        if should_run_llm:
            # Fallback to LLM
            llm_results = self.llm_engine.classify(normalized_text)
            
            # Merge results, prioritizing LLM results for the same text span if needed
            # For simplicity, we just add unique LLM results
            for lr in llm_results:
                if lr.original_text not in seen:
                    seen.add(lr.original_text)
                    unique_results.append(lr)

        # 3. Safely resolve actual processor name if wrapped by the LLM Decorator
        # Since the decorator wraps the base class, processor.__class__.__name__ would just return "LLMFallbackDecorator"
        processor_class_name = getattr(processor, "base_processor", processor).__class__.__name__
        if self.settings.use_llm_extraction:
            processor_class_name += " (with LLM Fallback)"

        return PipelineOutput(
            processor_name=processor_class_name,
            raw_text=raw_text,
            normalized_text=normalized_text,
            classified_results=unique_results,
        )