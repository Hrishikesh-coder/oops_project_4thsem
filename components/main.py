import json
import os
from extractor import DocumentExtractor
from parser import DataParser

def process_document(pdf_path: str, output_dir: str = "output"):
    """Runs the full pipeline on a single document."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize Modules
    extractor = DocumentExtractor()
    parser = DataParser()

    # Step 1: Extract Text
    raw_text = extractor.process_pdf(pdf_path)

    # Step 2: Parse and Classify
    extracted_entities = parser.extract_and_classify(raw_text)

    # Step 3: Format Output
    output_filename = os.path.basename(pdf_path).replace('.pdf', '_results.json')
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_entities, f, indent=4)

    print(f"Success! Extracted {len(extracted_entities)} items. Saved to {output_path}")
    return extracted_entities

if __name__ == "__main__":
    # Test it by putting a 'sample.pdf' in the same folder
    if os.path.exists("sample.pdf"):
        process_document("sample.pdf")
    else:
        print("Please place a 'sample.pdf' in the directory to run the test.")