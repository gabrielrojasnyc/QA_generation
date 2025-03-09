#!/usr/bin/env python3

import os
import csv
import sys
from typing import List, Dict, Any, Optional

import PyPDF2
import openai
import anthropic
from anthropic import Anthropic
from openai import OpenAI

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

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

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

def initialize_model_client(model_info: Dict[str, Any]):
    """Initialize the appropriate model client"""
    api_name = model_info["api_name"]
    
    # Check for API keys
    if api_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is not set.")
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    if api_name == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY environment variable is not set.")
        api_key = input("Please enter your Anthropic API key: ")
        os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Initialize the appropriate client
    if api_name == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:  # anthropic
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def retrieve_relevant_chunks(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Simple keyword-based retrieval to find relevant chunks"""
    # Split query into keywords
    keywords = query.lower().split()
    
    # Score chunks based on keyword matches
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for keyword in keywords if keyword in chunk_lower)
        scored_chunks.append((chunk, score))
    
    # Sort by score (highest first) and get top_k chunks
    sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in sorted_chunks[:top_k]]

def get_answer_from_model(client, model_info: Dict[str, Any], query: str, context: str) -> str:
    """Get answer from the selected model"""
    model_name = model_info["model"]
    api_name = model_info["api_name"]
    
    prompt = f"""
Answer the question based only on the following context:

Context: {context}

Question: {query}

Answer:
"""
    
    if api_name == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    else:  # anthropic
        # Setup message parameters
        params = {
            "model": model_name,
            "max_tokens": 1000,
            "temperature": 0.1,
            "system": "You are a helpful assistant that answers questions based only on the provided context.",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Enable reasoning for Claude 3.7 models
        if "claude-3-7" in model_name:
            params["system"] = "You are a helpful assistant that answers questions based only on the provided context. Use step-by-step reasoning to analyze the context and arrive at the most accurate answer."
            # Try to use enable_reasoning if supported by the library version
            try:
                # First check if it works using an attribute test to avoid actual API calls
                import inspect
                if 'enable_reasoning' in inspect.signature(client.messages.create).parameters:
                    params["enable_reasoning"] = True
            except (ImportError, AttributeError):
                # If we can't check or the param isn't supported, don't use it
                pass
        
        response = client.messages.create(**params)
        return response.content[0].text.strip()

def ask_questions(client, model_info: Dict[str, Any], chunks: List[str]) -> List[Dict[str, str]]:
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
            # Retrieve relevant chunks
            relevant_chunks = retrieve_relevant_chunks(query, chunks)
            context = "\n\n".join(relevant_chunks)
            
            # Get answer from model
            answer = get_answer_from_model(client, model_info, query, context)
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
    print("PDF Q&A System\n")
    
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
    
    # Chunk the text
    print("Splitting text into chunks...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} document chunks")
    
    # Initialize model client
    client = initialize_model_client(model_info)
    
    # Interactive Q&A
    qa_pairs = ask_questions(client, model_info, chunks)
    
    print("Thank you for using PDF Q&A System!")

if __name__ == "__main__":
    main()