from dataclasses import dataclass
from typing import List

from .extractor import DocumentProcessorFactory
from .normalization import (
    BaseTextProcessor,
    PunctuationStripper,
    TextProcessor,
    WhitespaceRemover,
    WordToDigitConverter,
)
from .parser import BaseParserClassifier, ClassificationResult


@dataclass
class PipelineSettings:
    use_whitespace_remover: bool = True
    use_word_converter: bool = True
    use_punctuation_stripper: bool = False


@dataclass
class PipelineOutput:
    processor_name: str
    raw_text: str
    normalized_text: str
    classified_results: List[ClassificationResult]


class ProcessingPipeline:
    """Coordinates extraction, normalization, and classification."""

    def __init__(self, classifier: BaseParserClassifier, settings: PipelineSettings):
        self.classifier = classifier
        self.settings = settings

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

        classified_results = self.classifier.classify(normalized_text)

        return PipelineOutput(
            processor_name=processor.__class__.__name__,
            raw_text=raw_text,
            normalized_text=normalized_text,
            classified_results=classified_results,
        )
