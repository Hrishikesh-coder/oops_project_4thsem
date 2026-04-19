# 📄 OOP-Driven PDF Information Extractor

An intelligent, object-oriented data extraction pipeline that converts messy PDF documents (digital or scanned) into structured JSON data. It utilizes a cascading classification strategy—combining Regex, spaCy NLP, and Llama 3.3 LLM—to achieve high-precision entity recognition.

## 🚀 Key Features

- **Cascading Classification Engine**: Implements a "Chain of Responsibility" logic.
  - **Regex**: Fast, deterministic extraction for standard patterns (Phone, Email, PAN, etc.).
  - **spaCy NLP**: Context-aware entity recognition for dates, locations, and quantities.
  - **LLM Fallback (Llama 3.3)**: High-level reasoning to reconstruct messy OCR or classify complex entities.
- **Dynamic OCR Pipeline**: Automatically detects digital vs. scanned PDFs.
  - Uses **PyMuPDF** for digital extraction.
  - Uses **Tesseract OCR** with multiprocessing for scanned documents.
- **Normalization Decorators**: Dynamically stackable text processors (Whitespace remover, Word-to-Digit converter, etc.) using the Decorator Pattern.
- **Interactive UI**: Built with Streamlit for easy file uploads, real-time setting adjustments, and JSON downloads.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Extraction**: PyMuPDF, Pytesseract, pdf2image
- **NLP**: spaCy (`en_core_web_sm`)
- **LLM**: Llama 3.3 (via Groq/OpenAI API)
- **Core**: Python 3.9+, Pydantic

## 📋 Prerequisites

Ensure you have the following system dependencies installed for OCR support:

- **Tesseract OCR**: [Installation Guide](https://github.com/UB-Mannheim/tesseract/wiki)
- **Poppler**: [Installation Guide](https://github.com/oschwartz10612/poppler-windows/releases) (Required for `pdf2image`)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd oops_project_4thsem
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory (refer to `.env.example`):
   ```env
   LLAMA_API_KEY=your_api_key_here
   LLAMA_BASE_URL=https://api.groq.com/openai/v1
   ```

## 🏃 Usage

Launch the application:
```bash
streamlit run app.py
```

1. **Upload** a PDF document.
2. **Configure** pipeline settings in the sidebar (Normalization, AI Fallback).
3. **Review** the extracted text and classified JSON.
4. **Refine** results manually using the "Run AI Classification Refinement" button if needed.
5. **Download** the final structured JSON.

## 🏗️ Architecture (OOP Patterns)

- **Strategy Pattern**: `IDocumentProcessor` defines the contract for `DigitalPDFProcessor` and `ScannedPDFProcessor`.
- **Decorator Pattern**: 
  - `LLMFallbackDecorator` wraps extraction logic with AI cleanup.
  - `BaseTextProcessor` is decorated with specific normalization behaviors.
- **Factory Pattern**: `DocumentProcessorFactory` encapsulates the logic for choosing the right extraction engine.
- **Chain of Responsibility**: Classification flows from fast Regex → spaCy NER → LLM Fallback.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License.
