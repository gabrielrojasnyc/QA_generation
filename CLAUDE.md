# QA-generation-2 Development Guide

## Commands
* Run Flask web app: `python src/app.py`
* Run CLI version: `python src/simple_qa_system.py`
* Install dependencies: `pip install -r requirements.txt`

## Environment
* Set API keys in terminal: `export OPENAI_API_KEY=your-key` and `export ANTHROPIC_API_KEY=your-key`
* Python 3.8+ required

## Style Guide
* Format: Use Black formatter defaults with 4-space indentation
* Imports: Standard library first, third-party second, local imports last
* Typing: Always use type hints for function parameters and return values
* Naming: 
  - Snake case for functions/variables: `extract_text_from_pdf`
  - Classes: CapitalCase
* Error handling: Use try/except with specific error types
* Docstrings: Required for all functions/classes (use triple quotes)
* Maximum line length: 100 characters

## Development
* Create `uploads` folder before running app if it doesn't exist
* Store frontend modifications in `src/static` and `src/templates`