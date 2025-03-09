#!/usr/bin/env python3

import os
import csv
import tempfile
import uuid
import time
import json
import psutil
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, render_template, request, jsonify, send_from_directory, session, Response, stream_with_context
import PyPDF2
import openai
import anthropic
import tiktoken
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI

# Import concurrent processing functionality
from concurrent_qa import generate_qa_pairs_concurrently, ask_questions_concurrently
from embedding_retrieval import compare_retrieval_methods, RetrievalSystem

app = Flask(__name__)
app.secret_key = os.urandom(24)
# Use absolute path for uploads to prevent issues
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store sessions data (not persistent across server restarts)
session_data = {}

# Token counters
class TokenCounter:
    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        """Count tokens based on model type"""
        try:
            if "claude" in model.lower():
                # Anthropic's claude-specific token counting (approximation)
                return len(text) // 4  # Rough approximation
            else:
                # Use tiktoken for OpenAI models
                encoding = tiktoken.encoding_for_model(model) if model != "o3" and model != "gpt-4o" else tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Fallback approximation
            return len(text.split()) // 0.75

# Timing Utility
class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.timestamps = {}
        
    def mark(self, name: str) -> None:
        """Mark a timestamp with a name"""
        elapsed = time.time() - self.start_time
        self.timestamps[name] = elapsed
        
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics"""
        result = {}
        
        # Calculate timings between marks
        keys = list(self.timestamps.keys())
        for i in range(1, len(keys)):
            key = f"{keys[i-1]} to {keys[i]}"
            result[key] = self.timestamps[keys[i]] - self.timestamps[keys[i-1]]
            
        # Include total time
        if keys:
            result["total"] = self.timestamps[keys[-1]]
            
        # Add resource usage
        process = psutil.Process(os.getpid())
        result["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        
        return result

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
        return ""

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

def initialize_model_client(model_info: Dict[str, Any]):
    """Initialize the appropriate model client"""
    api_name = model_info["api_name"]
    
    # Initialize the appropriate client
    if api_name == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:  # anthropic
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_answer_from_model(client, model_info: Dict[str, Any], query: str, context: str) -> Dict[str, Any]:
    """Get answer from the selected model with timing and token statistics"""
    model_name = model_info["model"]
    api_name = model_info["api_name"]
    
    timer = Timer()
    timer.mark("start")
    
    prompt = f"""
Answer the question based only on the following context:

Context: {context}

Question: {query}

Answer:
"""
    
    # Count input tokens
    input_tokens = TokenCounter.count_tokens(prompt, model_name)
    timer.mark("token_count_input")
    
    result = {
        "answer": "",
        "timing": {},
        "tokens": {
            "input": input_tokens,
            "output": 0
        }
    }
    
    if api_name == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()
        # Update output token count from response
        result["tokens"]["output"] = response.usage.completion_tokens if hasattr(response, 'usage') else TokenCounter.count_tokens(answer, model_name)
        result["tokens"]["total"] = response.usage.total_tokens if hasattr(response, 'usage') else (result["tokens"]["input"] + result["tokens"]["output"])
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
        answer = response.content[0].text.strip()
        # Count output tokens
        result["tokens"]["output"] = TokenCounter.count_tokens(answer, model_name)
        result["tokens"]["total"] = result["tokens"]["input"] + result["tokens"]["output"]
    
    timer.mark("model_response")
    result["answer"] = answer
    result["timing"] = timer.get_stats()
    
    return result

def generate_qa_pairs_for_chunk(client, model_info: Dict[str, Any], chunk: str) -> Dict[str, Any]:
    """Generate QA pairs for a single chunk of text with timing and token statistics"""
    timer = Timer()
    timer.mark("start")
    
    prompt = f"""
Generate 3 important question-answer pairs based on the following text. For each QA pair, first create a concise question that asks about important information in the text, then provide a direct and accurate answer to that question based solely on the provided text.

Text: {chunk}

Format each QA pair as follows:
Q: [Question]
A: [Answer]

Make sure each question focuses on a different aspect of the text.
"""
    
    api_name = model_info["api_name"]
    model_name = model_info["model"]
    
    # Count input tokens
    input_tokens = TokenCounter.count_tokens(prompt, model_name)
    timer.mark("token_count_input")
    
    result = {
        "qa_pairs": [],
        "timing": {},
        "tokens": {
            "input": input_tokens,
            "output": 0
        }
    }
    
    if api_name == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates question-answer pairs based on provided text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        # Get token usage from response
        result["tokens"]["output"] = response.usage.completion_tokens if hasattr(response, 'usage') else TokenCounter.count_tokens(content, model_name)
        result["tokens"]["total"] = response.usage.total_tokens if hasattr(response, 'usage') else (result["tokens"]["input"] + result["tokens"]["output"])
    else:  # anthropic
        # Setup message parameters
        params = {
            "model": model_name,
            "max_tokens": 1000,
            "temperature": 0.7,
            "system": "You are a helpful assistant that generates question-answer pairs based on provided text.",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Enable reasoning for Claude 3.7 models
        if "claude-3-7" in model_name:
            params["system"] = "You are a helpful assistant that generates question-answer pairs based on provided text. Use step-by-step reasoning to analyze the text and create insightful questions and accurate answers."
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
        content = response.content[0].text.strip()
        # Count output tokens
        result["tokens"]["output"] = TokenCounter.count_tokens(content, model_name)
        result["tokens"]["total"] = result["tokens"]["input"] + result["tokens"]["output"]
    
    timer.mark("model_response")
    
    # Parse the response to extract QA pairs
    qa_pairs = []
    current_q = None
    current_a = None
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('Q:'):
            # If we have a previous QA pair, add it to our list
            if current_q and current_a:
                qa_pairs.append({"question": current_q, "answer": current_a})
            
            # Start a new QA pair
            current_q = line[2:].strip()
            current_a = None
        elif line.startswith('A:'):
            current_a = line[2:].strip()
    
    # Add the last QA pair if it exists
    if current_q and current_a:
        qa_pairs.append({"question": current_q, "answer": current_a})
    
    timer.mark("parsing")
    
    result["qa_pairs"] = qa_pairs
    result["timing"] = timer.get_stats()
    
    return result

def export_qa_to_csv(qa_pairs: List[Dict[str, str]], filename: str) -> str:
    """Export questions and answers to a CSV file"""
    if not qa_pairs:
        return "No Q&A pairs to export."
    
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question', 'answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for qa_pair in qa_pairs:
                writer.writerow(qa_pair)
        
        return csv_path
    except Exception as e:
        return f"Error exporting to CSV: {e}"

@app.route('/')
def index():
    # Get all available models
    models = {
        "1": {"name": "OpenAI o3", "model": "o3-mini-2025-01-31", "api_name": "openai"},
        "2": {"name": "OpenAI chatgpt-4.0", "model": "gpt-4o-2024-11-20", "api_name": "openai"},
        "3": {"name": "Anthropic Sonnet 3.7", "model": "claude-3-7-sonnet-latest", "api_name": "anthropic"},
        "4": {"name": "Anthropic Sonnet 3.5", "model": "claude-3-5-sonnet-latest", "api_name": "anthropic"}
    }
    return render_template('index.html', models=models)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    main_timer = Timer()
    main_timer.mark("start")
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Get API keys from form data or environment
    openai_key = request.form.get('openai_key') or os.environ.get("OPENAI_API_KEY")
    anthropic_key = request.form.get('anthropic_key') or os.environ.get("ANTHROPIC_API_KEY")
    
    # Update environment variables if needed
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    
    # Check API keys
    model_id = request.form.get('model')
    models = {
        "1": {"name": "OpenAI o3", "model": "o3-mini-2025-01-31", "api_name": "openai"},
        "2": {"name": "OpenAI chatgpt-4o", "model": "gpt-4o-2024-11-20", "api_name": "openai"},
        "3": {"name": "Anthropic Sonnet 3.7", "model": "claude-3-7-sonnet-latest", "api_name": "anthropic"},
        "4": {"name": "Anthropic Sonnet 3.5", "model": "claude-3-5-sonnet-latest", "api_name": "anthropic"}
    }
    
    model_info = models.get(model_id)
    if not model_info:
        return jsonify({"error": "Invalid model selection"}), 400
    
    api_name = model_info["api_name"]
    if api_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return jsonify({"error": "OpenAI API key is not set"}), 400
    
    if api_name == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "Anthropic API key is not set"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        # Generate session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Create a filename with session ID to avoid conflicts
        filename = f"{session_id}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        main_timer.mark("file_saved")
        
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        main_timer.mark("pdf_extracted")
        
        # Chunk the text
        chunks = chunk_text(text)
        main_timer.mark("text_chunked")
        
        # Initialize model client
        client = initialize_model_client(model_info)
        main_timer.mark("client_initialized")
        
        # Store session data
        session_data[session_id] = {
            'pdf_path': file_path,
            'model_info': model_info,
            'chunks': chunks,
            'client': client,
            'qa_pairs': [],
            'stats': {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'chunk_stats': []
            }
        }
        
        # For initial display, create QA pairs for all chunks concurrently
        main_timer.mark("concurrent_processing_start")
        
        # Process all chunks in parallel (with up to 5 concurrent workers)
        # Adjust max_workers and rate limits based on your API quotas
        all_qa_pairs, concurrent_stats = generate_qa_pairs_concurrently(
            client=client,
            model_info=model_info,
            chunks=chunks,
            max_workers=5,  # Adjust based on your API limits
            batch_size=10,
            rate_limit_calls=20,  # Adjust based on your API rate limits
            rate_limit_window=60
        )
        
        # Extract stats from concurrent processing
        total_input_tokens = concurrent_stats.get("total_input_tokens", 0)
        total_output_tokens = concurrent_stats.get("total_output_tokens", 0)
        chunk_stats = concurrent_stats.get("chunk_stats", [])
        
        # Organize QA pairs by chunk for individual display
        for i, chunk_stat in enumerate(chunk_stats):
            chunk_id = chunk_stat.get("chunk_id", i)
            # Find QA pairs for this chunk from the flattened list
            # This is an approximation - in production you'd want to track this more precisely
            chunk_qa_pairs = all_qa_pairs[i*3:(i+1)*3] if i*3 < len(all_qa_pairs) else []
            session_data[session_id][f'chunk_{chunk_id}_qa'] = chunk_qa_pairs
        
        main_timer.mark("all_chunks_processed")
        
        # Store all QA pairs and statistics
        session_data[session_id]['qa_pairs'] = all_qa_pairs
        session_data[session_id]['stats']['total_input_tokens'] = total_input_tokens
        session_data[session_id]['stats']['total_output_tokens'] = total_output_tokens
        session_data[session_id]['stats']['chunk_stats'] = chunk_stats
        session_data[session_id]['stats']['overall_timing'] = main_timer.get_stats()
        
        # Return chunks and QA pairs for display
        display_data = []
        for i, chunk in enumerate(chunks):
            chunk_qa = session_data[session_id].get(f'chunk_{i}_qa', [])
            display_data.append({
                'chunk_id': i,
                'text': chunk,
                'qa_pairs': chunk_qa
            })
        
        # Final statistics
        main_timer.mark("response_ready")
        stats = {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'chunks_count': len(chunks),
            'timing': main_timer.get_stats()
        }
        
        return jsonify({
            'success': True,
            'chunks_count': len(chunks),
            'display_data': display_data,
            'session_id': session_id,
            'stats': stats
        })
    
    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

@app.route('/regenerate_qa', methods=['POST'])
def regenerate_qa():
    timer = Timer()
    timer.mark("start")
    
    session_id = request.form.get('session_id')
    chunk_id = int(request.form.get('chunk_id'))
    
    if not session_id or session_id not in session_data:
        return jsonify({"error": "Invalid session"}), 400
    
    session_info = session_data[session_id]
    chunks = session_info['chunks']
    
    if chunk_id < 0 or chunk_id >= len(chunks):
        return jsonify({"error": "Invalid chunk ID"}), 400
    
    # Get the specific chunk
    chunk = chunks[chunk_id]
    timer.mark("chunk_retrieved")
    
    # Generate new QA pairs for this chunk with concurrent processing
    # Even though we're only processing one chunk, we use the concurrent system
    # for consistent handling and better error management
    
    all_qa_pairs, stats = generate_qa_pairs_concurrently(
        client=session_info['client'],
        model_info=session_info['model_info'],
        chunks=[chunk],  # Just the one chunk we want to regenerate
        max_workers=1,   # Only one chunk, so one worker is fine
        batch_size=1,
        rate_limit_calls=10,
        rate_limit_window=60
    )
    
    # The returned QA pairs are for this one chunk
    qa_pairs = all_qa_pairs
    
    timer.mark("qa_generated")
    
    # Update the session data
    session_data[session_id][f'chunk_{chunk_id}_qa'] = qa_pairs
    
    # Rebuild the complete list of QA pairs
    all_session_qa_pairs = []
    for i in range(len(chunks)):
        chunk_qa = session_data[session_id].get(f'chunk_{i}_qa', [])
        if isinstance(chunk_qa, list):
            all_session_qa_pairs.extend(chunk_qa)
        elif isinstance(chunk_qa, dict) and "qa_pairs" in chunk_qa:
            all_session_qa_pairs.extend(chunk_qa["qa_pairs"])
    
    # Update all QA pairs
    session_data[session_id]['qa_pairs'] = all_session_qa_pairs
    
    # Extract token stats from concurrent results
    chunk_tokens = stats.get("total_tokens", 0)
    input_tokens = stats.get("total_input_tokens", 0)
    output_tokens = stats.get("total_output_tokens", 0)
    
    # Update statistics
    chunk_stats = session_data[session_id]['stats'].get('chunk_stats', [])
    for i, stat in enumerate(chunk_stats):
        if stat["chunk_id"] == chunk_id:
            chunk_stats[i] = {
                "chunk_id": chunk_id,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": chunk_tokens
                },
                "timing": stats.get("chunk_stats", [{}])[0].get("timing", {}) if stats.get("chunk_stats") else {}
            }
    
    timer.mark("response_ready")
    
    return jsonify({
        'success': True,
        'qa_pairs': qa_pairs,
        'stats': {
            'tokens': {
                "input": input_tokens,
                "output": output_tokens,
                "total": chunk_tokens
            },
            'timing': stats,
            'regenerate_timing': timer.get_stats()
        }
    })

@app.route('/ask_question', methods=['POST'])
def ask_question():
    timer = Timer()
    timer.mark("start")
    
    session_id = request.form.get('session_id')
    question = request.form.get('question')
    
    if not session_id or session_id not in session_data:
        return jsonify({"error": "Invalid session"}), 400
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    session_info = session_data[session_id]
    chunks = session_info['chunks']
    client = session_info['client']
    model_info = session_info['model_info']
    
    timer.mark("session_loaded")
    
    # Use the concurrent question processing system
    # Note: For a single question, the benefit is minimal, but it uses the same infrastructure
    # and we get the benefit of better error handling
    try:
        # Start timer for concurrent processing
        concurrent_start = time.time()
        
        # Process single question using the concurrent system
        results, stats = ask_questions_concurrently(
            client=client,
            model_info=model_info,
            chunks=chunks,
            questions=[question],
            max_workers=1,  # Only one question, so one worker
            rate_limit_calls=20,
            rate_limit_window=60
        )
        
        # Record elapsed time for concurrent processing
        concurrent_time = time.time() - concurrent_start
        timer.mark("question_processed")
        
        # Extract result (should only be one)
        if results and len(results) > 0:
            result = results[0]
            answer = result.get("answer", "No answer generated")
            
            # Get detailed timing information from the result
            result_timing = result.get("timing", {})
            
            # Add to QA pairs with timing and token stats
            qa_entry = {
                "question": question, 
                "answer": answer,
                "tokens": result.get("tokens", {}),
                "timing": result_timing
            }
            session_info['qa_pairs'].append(qa_entry)
            
            # Update session statistics
            if 'custom_qa_stats' not in session_info:
                session_info['custom_qa_stats'] = []
            
            session_info['custom_qa_stats'].append({
                "question": question,
                "tokens": result.get("tokens", {}),
                "timing": result_timing
            })
            
            timer.mark("response_ready")
            overall_timing = timer.get_stats()
            
            # Enhance timing information for the response
            detailed_timing = {
                'total_request_time': overall_timing.get('total', 0),
                'session_load_time': overall_timing.get('start to session_loaded', 0),
                'model_processing_time': result_timing.get('model', 0) if 'model' in result_timing else 0,
                'retrieval_time': result_timing.get('retrieval', 0) if 'retrieval' in result_timing else 0,
                'concurrent_processing_time': concurrent_time,
                'response_preparation_time': overall_timing.get('question_processed to response_ready', 0)
            }
            
            return jsonify({
                'success': True,
                'answer': answer,
                'stats': {
                    'tokens': result.get("tokens", {}),
                    'timing': detailed_timing,
                    'model_timing': result_timing,
                    'overall_timing': overall_timing,
                    'concurrent_stats': stats
                }
            })
        else:
            return jsonify({"error": "No result returned from processing"}), 500
            
    except Exception as e:
        timer.mark("error_occurred")
        error_timing = timer.get_stats()
        
        return jsonify({
            "error": f"Error getting answer: {str(e)}",
            "timing": error_timing
        }), 500

@app.route('/export_csv', methods=['POST'])
def export_csv():
    timer = Timer()
    timer.mark("start")
    
    session_id = request.form.get('session_id')
    include_stats = request.form.get('include_stats', 'false').lower() == 'true'
    
    if not session_id or session_id not in session_data:
        return jsonify({"error": "Invalid session"}), 400
    
    session_info = session_data[session_id]
    qa_pairs = session_info['qa_pairs']
    
    if not qa_pairs:
        return jsonify({"error": "No Q&A pairs to export"}), 400
    
    timer.mark("session_loaded")
    
    # Create standardized QA pairs for export
    export_qa_pairs = []
    processing_times = []
    
    for item in qa_pairs:
        if isinstance(item, dict):
            if "question" in item and "answer" in item:
                # If it's already a simple QA pair
                export_pair = {"question": item["question"], "answer": item["answer"]}
                
                # Add stats if requested and available
                if include_stats and "tokens" in item:
                    export_pair["input_tokens"] = item["tokens"].get("input", 0)
                    export_pair["output_tokens"] = item["tokens"].get("output", 0)
                    export_pair["total_tokens"] = item["tokens"].get("total", 0)
                    
                    # Collect processing times for p99 calculation
                    if "timing" in item:
                        if "model_response" in item["timing"]:
                            process_time = item["timing"]["model_response"]
                            export_pair["response_time_seconds"] = process_time
                            processing_times.append(process_time)
                        elif "total" in item["timing"]:
                            process_time = item["timing"]["total"]
                            export_pair["response_time_seconds"] = process_time
                            processing_times.append(process_time)
                
                export_qa_pairs.append(export_pair)
    
    timer.mark("pairs_processed")
    
    # Add overall stats as the last row if requested
    if include_stats and 'stats' in session_info:
        stats = session_info['stats']
        
        # Calculate percentile processing times if we have enough data points
        p99_time = 0
        p95_time = 0
        avg_time = 0
        if processing_times:
            processing_times.sort()
            p99_index = min(int(len(processing_times) * 0.99), len(processing_times) - 1)
            p95_index = min(int(len(processing_times) * 0.95), len(processing_times) - 1)
            p99_time = processing_times[p99_index]
            p95_time = processing_times[p95_index]
            avg_time = sum(processing_times) / len(processing_times)
        
        # Get PDF page count if available
        page_count = 0
        if 'pdf_path' in session_info:
            try:
                with open(session_info['pdf_path'], "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    page_count = len(reader.pages)
            except Exception:
                pass  # Ignore errors, just keep page_count as 0
        
        # Calculate per-page metrics if we have pages
        per_page_metrics = {}
        if page_count > 0:
            per_page_metrics = {
                "avg_time_per_page": avg_time / page_count if avg_time > 0 else 0,
                "p99_time_per_page": p99_time / page_count if p99_time > 0 else 0,
                "p95_time_per_page": p95_time / page_count if p95_time > 0 else 0,
                "tokens_per_page": (stats.get('total_input_tokens', 0) + stats.get('total_output_tokens', 0)) / page_count
            }
        
        # Add summary statistics
        summary_stats = {
            "question": "=== SUMMARY STATISTICS ===",
            "answer": "",
            "input_tokens": stats.get('total_input_tokens', 0),
            "output_tokens": stats.get('total_output_tokens', 0),
            "total_tokens": stats.get('total_input_tokens', 0) + stats.get('total_output_tokens', 0),
            "p99_processing_time": p99_time,
            "p95_processing_time": p95_time,
            "avg_processing_time": avg_time,
            "page_count": page_count
        }
        
        # Add per-page metrics if available
        if per_page_metrics:
            summary_stats.update({
                "avg_time_per_page": per_page_metrics["avg_time_per_page"],
                "p99_time_per_page": per_page_metrics["p99_time_per_page"],
                "p95_time_per_page": per_page_metrics["p95_time_per_page"],
                "tokens_per_page": per_page_metrics["tokens_per_page"]
            })
        
        export_qa_pairs.append(summary_stats)
    
    # Export to CSV
    filename = f"{session_id}_qa_pairs.csv"
    
    # Use absolute path for the CSV file
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Determine fieldnames based on whether stats are included
            if include_stats:
                fieldnames = [
                    'question', 'answer', 'input_tokens', 'output_tokens', 'total_tokens', 
                    'response_time_seconds', 'p99_processing_time', 'p95_processing_time', 'avg_processing_time', 
                    'page_count', 'avg_time_per_page', 'p99_time_per_page', 'p95_time_per_page', 'tokens_per_page'
                ]
            else:
                fieldnames = ['question', 'answer']
                
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for qa_pair in export_qa_pairs:
                writer.writerow(qa_pair)
        
        timer.mark("csv_written")
        
        # Verify file exists
        if not os.path.exists(csv_path):
            return jsonify({"error": "Failed to create CSV file"}), 500
        
        # CSV file URL
        csv_url = f"/uploads/{filename}"
        
        return jsonify({
            'success': True,
            'csv_url': csv_url,
            'timing': timer.get_stats()
        })
    except Exception as e:
        return jsonify({"error": f"Error exporting to CSV: {str(e)}"}), 500

# Endpoints for retrieval comparison feature
@app.route('/upload_csv_for_comparison', methods=['POST'])
def upload_csv_for_comparison():
    timer = Timer()
    timer.mark("start")
    
    # Check if files were uploaded
    if 'csvFiles' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('csvFiles')
    
    # Check number of files (max 4)
    if len(files) > 4:
        return jsonify({"error": "Maximum 4 files are allowed"}), 400
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    
    processed_files = []
    
    for file in files:
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": f"File {file.filename} is not a CSV file"}), 400
        
        # Create a temporary filename
        file_uuid = str(uuid.uuid4())
        filename = f"{file_uuid}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read CSV and extract chunks and QA pairs
        try:
            df = pd.read_csv(file_path)
            
            # Extract questions and answers
            if 'question' not in df.columns or 'answer' not in df.columns:
                return jsonify({"error": f"CSV file {file.filename} must contain 'question' and 'answer' columns"}), 400
            
            # Get actual chunks from file
            chunks = []
            for _, row in df.iterrows():
                # For simplicity, treat each QA pair as a chunk
                # Could be enhanced to identify actual doc chunks
                chunks.append(f"Q: {row['question']} A: {row['answer']}")
            
            processed_files.append({
                "filename": file.filename,
                "file_path": file_path,
                "chunk_count": len(chunks),
                "qa_count": len(df),
                "chunks": chunks
            })
            
        except Exception as e:
            return jsonify({"error": f"Error processing CSV file {file.filename}: {str(e)}"}), 500
    
    timer.mark("files_processed")
    
    # Store files in session data with a unique ID
    comparison_id = str(uuid.uuid4())
    session['comparison_id'] = comparison_id
    
    # Store processed files in session data
    session_data[comparison_id] = {
        'files': processed_files,
        'api_key': request.form.get('retrievalApiKey'),
        'processing_times': timer.get_stats()
    }
    
    return jsonify({
        'success': True,
        'files': [
            {
                "filename": f["filename"],
                "chunk_count": f["chunk_count"],
                "qa_count": f["qa_count"]
            } for f in processed_files
        ],
        'comparison_id': comparison_id,
        'timing': timer.get_stats()
    })

@app.route('/compare_retrieval_methods', methods=['POST'])
def compare_retrieval():
    timer = Timer()
    timer.mark("start")
    
    # Get request data
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    comparison_id = session.get('comparison_id')
    if not comparison_id or comparison_id not in session_data:
        return jsonify({"error": "No active comparison session"}), 400
    
    # Get the query and methods
    query = data.get('query')
    methods = data.get('methods', ['keyword'])
    api_key = data.get('api_key', None) or session_data[comparison_id].get('api_key')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Get the chunks from all files
    all_chunks = []
    for file_data in session_data[comparison_id]['files']:
        all_chunks.extend(file_data['chunks'])
    
    timer.mark("data_prepared")
    
    # Run the comparison
    try:
        results = {}
        
        # For each requested method
        if 'keyword' in methods:
            # Always run keyword search as it's fast
            keyword_retriever = RetrievalSystem(method="keyword")
            keyword_retriever.add_chunks(all_chunks)
            results["keyword"] = keyword_retriever.retrieve(query, top_k=5)
            timer.mark("keyword_search_complete")
            
        # Only run embedding methods if API key is provided
        if api_key and ('embedding' in methods or 'hybrid' in methods):
            try:
                # Initialize embedding system
                if 'embedding' in methods:
                    embedding_retriever = RetrievalSystem(method="embedding")
                    embedding_retriever.initialize_embeddings(api_key=api_key)
                    embedding_retriever.add_chunks(all_chunks)
                    results["embedding"] = embedding_retriever.retrieve(query, top_k=5)
                    timer.mark("embedding_search_complete")
                
                if 'hybrid' in methods:
                    hybrid_retriever = RetrievalSystem(method="hybrid")
                    hybrid_retriever.initialize_embeddings(api_key=api_key)
                    hybrid_retriever.add_chunks(all_chunks)
                    results["hybrid"] = hybrid_retriever.retrieve(query, top_k=5)
                    timer.mark("hybrid_search_complete")
                    
            except Exception as e:
                # In case of embedding error, just log it
                print(f"Error with embedding retrieval: {e}")
                results["error"] = str(e)
        
    except Exception as e:
        return jsonify({"error": f"Error comparing retrieval methods: {str(e)}"}), 500
    
    timer.mark("done")
    
    return jsonify({
        'success': True,
        'results': results,
        'timing': timer.get_stats()
    })

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5002, help='Port to run the server on')
    args = parser.parse_args()
    app.run(debug=False, port=args.port)