#!/usr/bin/env python3

import os
import csv
import sys
from typing import List, Dict, Any, Optional

import PyPDF2
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor, BM25Retriever, PromptNode, PromptTemplate
from haystack.schema import Document
from haystack.pipelines import Pipeline

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        sys.exit(1)

def preprocess_text(text: str, preprocessor: PreProcessor) -> List[Document]:
    """Split text into chunks using Haystack preprocessor"""
    doc = Document(content=text)
    docs = preprocessor.process([doc])
    print(f"Created {len(docs)} document chunks")
    return docs

def select_language_model() -> Dict[str, Any]:
    """Prompt user to select a language model"""
    models = {
        "1": {"name": "OpenAI o3", "model": "o3", "api_name": "openai"},
        "2": {"name": "OpenAI chatgpt-4o", "model": "gpt-4o", "api_name": "openai"},
        "3": {"name": "Anthropic Sonnet 3.7", "model": "claude-3-sonnet-20240229", "api_name": "anthropic"},
        "4": {"name": "Anthropic Sonnet 3.5", "model": "claude-3-5-sonnet-20240620", "api_name": "anthropic"}
    }
    
    print("Select a language model:")
    for key, model_info in models.items():
        print(f"{key}. {model_info['name']}")
    
    while True:
        choice = input("Enter your choice (1-4): ")
        if choice in models:
            return models[choice]
        print("Invalid choice. Please enter a number between 1 and 4.")

def create_prompt_node(model_info: Dict[str, Any]) -> PromptNode:
    """Create a PromptNode with the selected model"""
    api_name = model_info["api_name"]
    model = model_info["model"]
    
    # Check for API keys
    if api_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is not set.")
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    if api_name == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY environment variable is not set.")
        api_key = input("Please enter your Anthropic API key: ")
        os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Create a QA prompt template
    qa_template = PromptTemplate(
        prompt="""
        Answer the question based only on the following context:
        
        Context: {{documents}}
        
        Question: {{query}}
        
        Answer:
        """,
        output_parser={"type": "DefaultOutputParser"}
    )
    
    # Configure model_kwargs based on model type
    model_kwargs = {"temperature": 0.1}
    
    # Enable reasoning for Claude 3.7 models
    if api_name == "anthropic" and "claude-3-7" in model:
        model_kwargs["system"] = "You are a helpful assistant that answers questions based only on the provided context. Use step-by-step reasoning to analyze the context and arrive at the most accurate answer."
        model_kwargs["temperature"] = 0.1
        
        # Try to use enable_reasoning if supported by the library version
        try:
            # Check if Anthropic SDK supports enable_reasoning parameter
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            import inspect
            if 'enable_reasoning' in inspect.signature(client.messages.create).parameters:
                model_kwargs["enable_reasoning"] = True
        except (ImportError, AttributeError):
            # If we can't check or the param isn't supported, don't use it
            pass
    
    return PromptNode(
        model_name_or_path=model,
        api_key=os.environ.get(f"{api_name.upper()}_API_KEY"),
        default_prompt_template=qa_template,
        model_kwargs=model_kwargs,
        api_name=api_name
    )

def setup_pipeline(documents: List[Document], prompt_node: PromptNode) -> Pipeline:
    """Set up the Haystack pipeline with document store, retriever, and prompt node"""
    # Initialize document store and add documents
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(documents)
    
    # Initialize retriever
    retriever = BM25Retriever(document_store=document_store, top_k=3)
    
    # Create the pipeline
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
    
    return pipeline

def ask_questions(pipeline: Pipeline) -> List[Dict[str, str]]:
    """Interactive CLI for asking questions about the PDF content"""
    qa_pairs = []
    
    print("\nAsk questions about the document (type 'exit' to quit, 'export' to save Q&A to CSV):")
    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break
        elif query.lower() == "export":
            export_qa_to_csv(qa_pairs)
            continue
        
        try:
            result = pipeline.run(query=query)
            answer = result["answers"][0].answer
            print(f"Answer: {answer}")
            
            # Store question and answer
            qa_pairs.append({"question": query, "answer": answer})
        except Exception as e:
            print(f"Error processing question: {e}")
    
    return qa_pairs

def export_qa_to_csv(qa_pairs: List[Dict[str, str]]) -> None:
    """Export questions and answers to a CSV file"""
    if not qa_pairs:
        print("No Q&A pairs to export.")
        return
    
    file_path = input("Enter CSV file path to save Q&A (default: qa_results.csv): ") or "qa_results.csv"
    
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question', 'answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for qa_pair in qa_pairs:
                writer.writerow(qa_pair)
        
        print(f"Q&A pairs exported to {file_path}")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")

def main():
    print("PDF Q&A System using Haystack\n")
    
    # Select language model
    model_info = select_language_model()
    print(f"Selected model: {model_info['name']}")
    
    # Get PDF path
    pdf_path = input("\nEnter the path to the PDF file: ")
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    
    # Extract text from PDF
    print(f"\nExtracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    # Initialize preprocessor for text chunking
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=200,
        split_overlap=20,
        split_respect_sentence_boundary=True
    )
    
    # Preprocess the text into document chunks
    docs = preprocess_text(text, preprocessor)
    
    # Create PromptNode with selected model
    prompt_node = create_prompt_node(model_info)
    
    # Set up the pipeline
    print("Setting up Q&A pipeline...")
    pipeline = setup_pipeline(docs, prompt_node)
    
    # Interactive Q&A
    qa_pairs = ask_questions(pipeline)
    
    print("Thank you for using PDF Q&A System!")

if __name__ == "__main__":
    main()