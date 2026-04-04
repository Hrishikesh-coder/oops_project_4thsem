from extractor import DocumentProcessorFactory
from normalization import (
    BaseTextProcessor, 
    WhitespaceRemover, 
    PunctuationStripper, 
    WordToDigitConverter
)

def run_pipeline(pdf_path: str):
    print(f"--- Starting Processing for: {pdf_path} ---")

    # ---------------------------------------------------------
    # 1. FACTORY PATTERN IN ACTION
    # We ask the factory for a processor. We don't care if it returns 
    # the Digital or Scanned class, because both guarantee an 'extract_text' method.
    # ---------------------------------------------------------
    processor = DocumentProcessorFactory.get_processor(pdf_path)
    raw_text = processor.extract_text(pdf_path)
    
    print(f"Extraction complete. Raw length: {len(raw_text)} chars.\n")

    # ---------------------------------------------------------
    # 2. DECORATOR PATTERN IN ACTION
    # Stacking our text processing behaviors dynamically.
    # If a document doesn't need punctuation stripped, we simply omit that line.
    # ---------------------------------------------------------
    
    # Start with the base component
    text_pipeline = BaseTextProcessor()
    
    # Wrap it in decorators
    text_pipeline = WhitespaceRemover(text_pipeline)
    text_pipeline = WordToDigitConverter(text_pipeline)
    text_pipeline = PunctuationStripper(text_pipeline)
    
    # Execute the chain
    normalized_text = text_pipeline.process(raw_text)
    
    print("\n--- Pipeline Complete ---")
    print("Preview of normalized text:")
    print(normalized_text[:200] + "...")
    
    return normalized_text

if __name__ == "__main__":
    # Test execution
    # run_pipeline("sample.pdf")
    pass
