# PDF Q&A System


https://github.com/user-attachments/assets/0cda34c0-f659-4642-89a0-4e049c419031



A Python application that implements a PDF text chunking and question-answering (Q&A) system with a web interface.

## Features

- Extract text from PDF documents
- Split text into meaningful chunks (200 words with 20 word overlap)
- Use various language models for Q&A:
  - OpenAI's o3
  - OpenAI's chatgpt-4o
  - Anthropic Sonnet 3.7 (with reasoning flag support)
  - Anthropic Sonnet 3.5
- Web interface for uploading PDFs and interacting with Q&A
- Generate Q&A pairs automatically for each text chunk
- Ask custom questions about the document content
- Regenerate Q&A pairs for specific chunks
- Export Q&A pairs to CSV

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set your API keys as environment variables (optional, you can also input them in the web interface):

```bash
export OPENAI_API_KEY=your-openai-api-key
export ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Usage

### Web Interface

Run the Flask web server:

```bash
python src/app.py
```

Then open your browser and go to http://127.0.0.1:5000/

The web interface allows you to:
1. Enter your API keys (stored locally in your browser)
2. Upload a PDF file
3. Select a language model
4. View the generated Q&A pairs for each text chunk
5. Regenerate Q&A pairs for specific chunks
6. Ask custom questions about the document
7. Export all Q&A pairs to CSV

### Command Line Interface

For a simpler command line interface, you can also use:

```bash
python src/simple_qa_system.py
```

Follow the interactive prompts to:
1. Select a language model
2. Provide the path to a PDF file
3. Ask questions about the PDF content
4. Type 'export' to save Q&A pairs to a CSV file
5. Type 'exit' to quit

## Requirements

- Python 3.8+
- Flask 2.2.0+
- PyPDF2 3.0.0+
- OpenAI API key (for OpenAI models)
- Anthropic API key (for Anthropic models)
- Supports Claude 3.7 reasoning flag when available (SDK v0.21.0+)
- See requirements.txt for full package dependencies

## Enhanced Reasoning with Claude 3.7

This application supports Claude 3.7's reasoning flag feature, which enables:

- Step-by-step analysis of context for better question answering
- Improved accuracy through more methodical reasoning
- Enhanced extraction of insights when generating QA pairs

The reasoning flag is automatically enabled when using the Claude 3.7 model and a compatible Anthropic SDK version (0.21.0+). If an earlier SDK version is detected, the application will still work with Claude 3.7 but without explicit reasoning flag support.
