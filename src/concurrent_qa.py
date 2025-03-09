#!/usr/bin/env python3

import os
import time
import json
import concurrent.futures
from threading import Lock
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# For API rate limiting
import time
from collections import deque


class RateLimiter:
    """
    Rate limiter to control API requests and respect rate limits.
    Uses a sliding window approach for tracking requests.
    """
    
    def __init__(self, max_calls: int = 10, time_window: int = 60):
        """
        Initialize a rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()  # Will hold timestamps of calls
        self.lock = Lock()  # Thread-safe access to calls deque
    
    def _clean_old_calls(self):
        """Remove calls outside the current time window"""
        current_time = time.time()
        with self.lock:
            while self.calls and self.calls[0] < current_time - self.time_window:
                self.calls.popleft()
    
    def can_make_call(self) -> bool:
        """Check if a call can be made without exceeding the rate limit"""
        self._clean_old_calls()
        with self.lock:
            return len(self.calls) < self.max_calls
    
    def wait_if_needed(self) -> float:
        """
        Wait if necessary to respect rate limit.
        Returns the time waited in seconds.
        """
        self._clean_old_calls()
        
        with self.lock:
            if len(self.calls) < self.max_calls:
                # Can make a call immediately
                self.calls.append(time.time())
                return 0
            
            # Need to wait until the oldest call expires
            wait_time = max(0, self.time_window - (time.time() - self.calls[0]))
            
            if wait_time > 0:
                time.sleep(wait_time)
                
            # Record the new call
            self.calls.append(time.time())
            # Remove the oldest call since we've waited for it to expire
            self.calls.popleft()
            
            return wait_time
    
    def record_call(self):
        """Record that a call was made"""
        with self.lock:
            self.calls.append(time.time())


class ProgressTracker:
    """
    Thread-safe progress tracker for monitoring concurrent operations.
    Provides both numerical and visual feedback.
    """
    
    def __init__(self, total: int, description: str = "Progress"):
        """
        Initialize a progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the operation being tracked
        """
        self.total = total
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        self.lock = Lock()
        self.description = description
        self.start_time = time.time()
        self.item_times = []  # Track time taken for each item
        
        # Store additional stats
        self.stats = {
            "total_items": total,
            "completed": 0,
            "failed": 0,
            "avg_time_per_item": 0,
            "estimated_time_remaining": 0,
            "elapsed_time": 0
        }
    
    def start_item(self) -> int:
        """
        Mark an item as started and return its index.
        Thread-safe.
        """
        with self.lock:
            self.in_progress += 1
            return self.completed + self.in_progress
    
    def complete_item(self, time_taken: float = None):
        """
        Mark an item as completed.
        Thread-safe.
        """
        with self.lock:
            self.completed += 1
            self.in_progress -= 1
            
            if time_taken is not None:
                self.item_times.append(time_taken)
            
            # Update stats
            self._update_stats()
    
    def fail_item(self):
        """
        Mark an item as failed.
        Thread-safe.
        """
        with self.lock:
            self.failed += 1
            self.in_progress -= 1
            
            # Update stats
            self._update_stats()
    
    def _update_stats(self):
        """Update internal statistics"""
        elapsed = time.time() - self.start_time
        
        self.stats["completed"] = self.completed
        self.stats["failed"] = self.failed
        self.stats["elapsed_time"] = elapsed
        
        # Calculate average time per item
        if self.item_times:
            avg_time = sum(self.item_times) / len(self.item_times)
            self.stats["avg_time_per_item"] = avg_time
            
            # Estimate remaining time
            remaining_items = self.total - self.completed - self.failed
            self.stats["estimated_time_remaining"] = avg_time * remaining_items
        else:
            # If no item times available, use overall elapsed time
            if self.completed > 0:
                avg_time = elapsed / self.completed
                self.stats["avg_time_per_item"] = avg_time
                
                # Estimate remaining time
                remaining_items = self.total - self.completed - self.failed
                self.stats["estimated_time_remaining"] = avg_time * remaining_items
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress statistics.
        Thread-safe.
        """
        with self.lock:
            # Make a copy of stats to avoid race conditions
            stats_copy = self.stats.copy()
            
            # Add current progress percentage
            if self.total > 0:
                stats_copy["progress_percent"] = (self.completed / self.total) * 100
            else:
                stats_copy["progress_percent"] = 0
                
            # Add current numbers
            stats_copy["current_completed"] = self.completed
            stats_copy["current_failed"] = self.failed
            stats_copy["current_in_progress"] = self.in_progress
            
            return stats_copy
    
    def print_progress(self, show_time: bool = True):
        """
        Print current progress to the console.
        Thread-safe.
        """
        with self.lock:
            percent = (self.completed / self.total) * 100 if self.total > 0 else 0
            
            if show_time and self.item_times:
                avg_time = sum(self.item_times) / len(self.item_times)
                remaining = avg_time * (self.total - self.completed - self.failed)
                time_info = f" | Avg: {avg_time:.2f}s | Est. remaining: {remaining:.0f}s"
            else:
                time_info = ""
            
            print(f"{self.description}: {self.completed}/{self.total} ({percent:.1f}%) | Failed: {self.failed}{time_info}")


class BatchProcessor:
    """
    Processes items in batches with concurrent execution and rate limiting.
    Provides robust error handling and progress tracking.
    """
    
    def __init__(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        max_workers: int = 5,
        batch_size: int = 10,
        max_retries: int = 3,
        rate_limit_calls: int = 20,
        rate_limit_window: int = 60,
        progress_description: str = "Processing"
    ):
        """
        Initialize a batch processor.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            max_workers: Maximum number of concurrent workers
            batch_size: Number of items to process in each batch
            max_retries: Maximum number of retry attempts for failed items
            rate_limit_calls: Maximum number of calls in rate limit window
            rate_limit_window: Rate limit time window in seconds
            progress_description: Description for progress reporting
        """
        self.items = items
        self.process_func = process_func
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.progress = ProgressTracker(len(items), progress_description)
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)
        
        # Results storage
        self.results = []
        self.failed_items = []
        self.error_messages = []
        
        # Locks for thread safety
        self.results_lock = Lock()
        self.errors_lock = Lock()
    
    def _process_item(self, item: Any, item_index: int) -> Tuple[int, Any, float, Optional[Exception]]:
        """
        Process a single item with rate limiting and timing.
        Returns (item_index, result, time_taken, exception if any)
        """
        # Mark item as in progress
        self.progress.start_item()
        
        # Handle rate limiting
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            print(f"Rate limit hit, waited {wait_time:.2f}s before processing item {item_index}")
        
        start_time = time.time()
        exception = None
        result = None
        
        try:
            # Process the item
            result = self.process_func(item)
            success = True
        except Exception as e:
            exception = e
            success = False
        
        time_taken = time.time() - start_time
        
        # Update progress
        if success:
            self.progress.complete_item(time_taken)
        else:
            self.progress.fail_item()
        
        return (item_index, result, time_taken, exception)
    
    def _process_batch(self, batch: List[Tuple[int, Any]]):
        """Process a batch of items concurrently"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_item, item, idx): (idx, item) for idx, item in batch}
            
            for future in as_completed(futures):
                idx, result, time_taken, exception = future.result()
                
                if exception is None:
                    # Success - store result
                    with self.results_lock:
                        self.results.append((idx, result))
                else:
                    # Failure - store error
                    with self.errors_lock:
                        error_info = {
                            "item_index": idx,
                            "item": futures[future][1],
                            "error": str(exception),
                            "traceback": traceback.format_exc()
                        }
                        self.failed_items.append((idx, futures[future][1]))
                        self.error_messages.append(error_info)
                        
                        print(f"Error processing item {idx}: {str(exception)}")
    
    def process_all(self, show_progress: bool = True) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process all items in batches with progress tracking.
        Returns (results ordered by original index, statistics dict)
        """
        # Prepare items with their original indices
        indexed_items = list(enumerate(self.items))
        
        # Process in batches
        for i in range(0, len(indexed_items), self.batch_size):
            batch = indexed_items[i:i + self.batch_size]
            self._process_batch(batch)
            
            if show_progress:
                self.progress.print_progress()
        
        # Handle retries for failed items
        retry_count = 0
        while self.failed_items and retry_count < self.max_retries:
            retry_count += 1
            print(f"\nRetrying {len(self.failed_items)} failed items (attempt {retry_count}/{self.max_retries})...")
            
            retry_items = self.failed_items
            self.failed_items = []  # Reset for this retry round
            
            # Process the retry batch
            self._process_batch(retry_items)
            
            if show_progress:
                self.progress.print_progress()
        
        # Sort results by original index
        with self.results_lock:
            sorted_results = [r for _, r in sorted(self.results, key=lambda x: x[0])]
        
        # Prepare statistics
        stats = self.progress.get_progress()
        stats["total_retries"] = retry_count
        stats["final_failed_count"] = len(self.failed_items)
        stats["error_summary"] = [str(err["error"]) for err in self.error_messages]
        
        return sorted_results, stats


def process_chunks_concurrently(
    chunks: List[str],
    process_func: Callable[[str], Any],
    max_workers: int = 5,
    batch_size: int = 10,
    rate_limit_calls: int = 20,
    rate_limit_window: int = 60,
    max_retries: int = 3,
    show_progress: bool = True
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Process text chunks concurrently with batching and rate limiting.
    
    Args:
        chunks: List of text chunks to process
        process_func: Function to process each chunk, taking a chunk and returning any result
        max_workers: Maximum number of concurrent workers
        batch_size: Number of items to process in each batch
        rate_limit_calls: Maximum number of calls in rate limit window
        rate_limit_window: Rate limit time window in seconds
        max_retries: Maximum number of retry attempts for failed items
        show_progress: Whether to show progress updates
        
    Returns:
        Tuple of (list of results in the same order as chunks, stats dictionary)
    """
    processor = BatchProcessor(
        items=chunks,
        process_func=process_func,
        max_workers=max_workers,
        batch_size=batch_size,
        max_retries=max_retries,
        rate_limit_calls=rate_limit_calls,
        rate_limit_window=rate_limit_window,
        progress_description="Processing chunks"
    )
    
    return processor.process_all(show_progress)


def generate_qa_pairs_concurrently(
    client: Any,
    model_info: Dict[str, Any],
    chunks: List[str],
    max_workers: int = 5,
    batch_size: int = 10,
    rate_limit_calls: int = 10,
    rate_limit_window: int = 60
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate QA pairs for multiple chunks concurrently.
    
    Args:
        client: The API client (OpenAI or Anthropic)
        model_info: Model information dictionary
        chunks: List of text chunks to process
        max_workers: Maximum number of concurrent workers
        batch_size: Number of items to process in each batch
        rate_limit_calls: Maximum number of calls in rate limit window
        rate_limit_window: Rate limit time window in seconds
        
    Returns:
        Tuple of (list of QA pairs, statistics dictionary)
    """
    # Define the function to process each chunk (generate QA pairs)
    def generate_qa_for_chunk(chunk: str) -> Dict[str, Any]:
        api_name = model_info["api_name"]
        model_name = model_info["model"]
        
        # Create timer for accurate timing measurement
        timer = time.time()
        
        prompt = f"""
        Generate 3 important question-answer pairs based on the following text. For each QA pair, first create a concise question that asks about important information in the text, then provide a direct and accurate answer to that question based solely on the provided text.

        Text: {chunk}

        Format each QA pair as follows:
        Q: [Question]
        A: [Answer]

        Make sure each question focuses on a different aspect of the text.
        """
        
        # Calculate input tokens for each request
        from app import TokenCounter
        input_tokens = TokenCounter.count_tokens(prompt, model_name)
        
        result = {
            "qa_pairs": [],
            "chunk": chunk,
            "tokens": {
                "input": input_tokens,
                "output": 0,
                "total": 0
            },
            "timing": {
                "total": 0
            }
        }
        
        # Different handling based on API
        if api_name == "openai":
            # Special handling for o3 model which doesn't support temperature
            if model_name == "o3" or model_name.startswith("o3-"):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates question-answer pairs based on provided text."},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
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
            if hasattr(response, 'usage'):
                result["tokens"]["output"] = response.usage.completion_tokens
                result["tokens"]["total"] = response.usage.total_tokens
            else:
                result["tokens"]["output"] = TokenCounter.count_tokens(content, model_name)
                result["tokens"]["total"] = result["tokens"]["input"] + result["tokens"]["output"]
                
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
        
        # Record timing information
        result["timing"]["total"] = time.time() - timer
        result["qa_pairs"] = qa_pairs
        
        return result
    
    # Process all chunks concurrently
    results, stats = process_chunks_concurrently(
        chunks=chunks,
        process_func=generate_qa_for_chunk,
        max_workers=max_workers,
        batch_size=batch_size,
        rate_limit_calls=rate_limit_calls,
        rate_limit_window=rate_limit_window
    )
    
    # Extract QA pairs from results
    all_qa_pairs = []
    chunk_stats = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, result in enumerate(results):
        if result and "qa_pairs" in result:
            all_qa_pairs.extend(result["qa_pairs"])
            
            # Collect token and timing stats
            if "tokens" in result:
                total_input_tokens += result["tokens"]["input"]
                total_output_tokens += result["tokens"]["output"]
                
                chunk_stats.append({
                    "chunk_id": i,
                    "tokens": result["tokens"],
                    "timing": result["timing"]
                })
    
    # Add token stats to the overall stats
    stats["total_input_tokens"] = total_input_tokens
    stats["total_output_tokens"] = total_output_tokens
    stats["total_tokens"] = total_input_tokens + total_output_tokens
    stats["chunk_stats"] = chunk_stats
    
    return all_qa_pairs, stats


def ask_questions_concurrently(
    client: Any,
    model_info: Dict[str, Any],
    chunks: List[str],
    questions: List[str],
    max_workers: int = 5,
    rate_limit_calls: int = 10,
    rate_limit_window: int = 60
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Process multiple questions concurrently against a document.
    
    Args:
        client: The API client (OpenAI or Anthropic)
        model_info: Model information dictionary
        chunks: List of text chunks for retrieval
        questions: List of questions to answer
        max_workers: Maximum number of concurrent workers
        rate_limit_calls: Maximum number of calls in rate limit window
        rate_limit_window: Rate limit time window in seconds
        
    Returns:
        Tuple of (list of QA pairs with answers, statistics dictionary)
    """
    # Function to process a single question
    def process_question(question: str) -> Dict[str, str]:
        api_name = model_info["api_name"]
        model_name = model_info["model"]
        
        # Start timer
        timer = time.time()
        
        # Find relevant chunks (simple keyword matching)
        keywords = question.lower().split()
        scored_chunks = []
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            scored_chunks.append((chunk, score))
        
        # Get top 3 chunks
        sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, score in sorted_chunks[:3]]
        context = "\n\n".join(top_chunks)
        
        retrieval_time = time.time() - timer
        
        prompt = f"""
        Answer the question based only on the following context:
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Calculate input tokens
        from app import TokenCounter
        input_tokens = TokenCounter.count_tokens(prompt, model_name)
        
        result = {
            "question": question,
            "answer": "",
            "tokens": {
                "input": input_tokens,
                "output": 0,
                "total": 0
            },
            "timing": {
                "retrieval": retrieval_time,
                "total": 0
            }
        }
        
        # Get answer from model
        try:
            model_start = time.time()
            
            if api_name == "openai":
                # Special handling for o3 model which doesn't support temperature
                if model_name == "o3" or model_name.startswith("o3-"):
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                answer = response.choices[0].message.content.strip()
                
                # Get token usage from API response
                if hasattr(response, 'usage'):
                    result["tokens"]["output"] = response.usage.completion_tokens
                    result["tokens"]["total"] = response.usage.total_tokens
                else:
                    result["tokens"]["output"] = TokenCounter.count_tokens(answer, model_name)
                    result["tokens"]["total"] = result["tokens"]["input"] + result["tokens"]["output"]
                
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
                
                # Count output tokens manually
                result["tokens"]["output"] = TokenCounter.count_tokens(answer, model_name)
                result["tokens"]["total"] = result["tokens"]["input"] + result["tokens"]["output"]
            
            model_time = time.time() - model_start
            
            # Record timing info
            result["timing"]["model"] = model_time
            result["timing"]["total"] = time.time() - timer
            result["answer"] = answer
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            result["answer"] = f"Error generating answer: {str(e)}"
            result["timing"]["total"] = time.time() - timer
            result["error"] = str(e)
            return result
    
    # Process all questions concurrently
    results, stats = process_chunks_concurrently(
        chunks=questions,  # Here we're processing questions, not chunks
        process_func=process_question,
        max_workers=max_workers,
        batch_size=len(questions),  # Process all questions in one batch
        rate_limit_calls=rate_limit_calls,
        rate_limit_window=rate_limit_window
    )
    
    # Add up token usage across all questions
    total_input_tokens = sum(r.get("tokens", {}).get("input", 0) for r in results if r)
    total_output_tokens = sum(r.get("tokens", {}).get("output", 0) for r in results if r)
    
    stats["total_input_tokens"] = total_input_tokens
    stats["total_output_tokens"] = total_output_tokens
    stats["total_tokens"] = total_input_tokens + total_output_tokens
    
    return results, stats


if __name__ == "__main__":
    # Example usage
    from time import sleep
    import random
    
    # Simulate processing chunks
    def simulate_chunk_processing(chunk):
        # Simulate API calls with random processing time and occasional failures
        sleep_time = random.uniform(0.1, 1.0)
        sleep(sleep_time)
        
        # Simulate occasional failures (10% chance)
        if random.random() < 0.1:
            raise Exception("Simulated random failure")
            
        return f"Processed: {chunk[:20]}... (took {sleep_time:.2f}s)"
    
    # Test with sample chunks
    sample_chunks = [f"This is chunk {i}" for i in range(20)]
    
    print("Testing concurrent processing...")
    results, stats = process_chunks_concurrently(
        chunks=sample_chunks,
        process_func=simulate_chunk_processing,
        max_workers=3,
        batch_size=5,
        rate_limit_calls=10,
        rate_limit_window=5,
        max_retries=2
    )
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i}: {result}")
    
    print("\nStats:")
    for key, value in stats.items():
        print(f"{key}: {value}")