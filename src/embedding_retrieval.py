#!/usr/bin/env python3

import os
import sys
import json
import pickle
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Union

# Try to import required packages
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

try:
    import faiss
    faiss_available = True
except ImportError:
    faiss_available = False

try:
    import chromadb
    chroma_available = True
except ImportError:
    chroma_available = False

try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False


class TextEmbedder:
    """
    Abstract base class for text embedding models.
    """
    
    def __init__(self):
        self.model_name = "base_embedder"
        self.dim = 0
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts into vector representations.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text. May be optimized differently than document embedding.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        result = self.embed([text])
        return result[0] if result else []
    
    def get_embedding_dimension(self) -> int:
        """
        Return the dimension of the embedding vectors.
        """
        return self.dim


class OpenAIEmbedder(TextEmbedder):
    """
    Text embedder using OpenAI's embedding models.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize OpenAI embedder.
        
        Args:
            model_name: Name of the OpenAI embedding model to use
            api_key: OpenAI API key (defaults to environment variable)
        """
        super().__init__()
        if not openai_available:
            raise ImportError("OpenAI package is required. Install with 'pip install openai'")
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as api_key.")
            
        self.client = OpenAI(api_key=self.api_key)
        
        # Embedding dimensions by model
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.dim = dimensions.get(model_name, 1536)  # Default to 1536 if unknown
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI API"""
        if not texts:
            return []
            
        # Process in batches of 20 for API efficiency
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Error getting embeddings from OpenAI: {e}")
                # Fill with zero embeddings for failed batch
                zero_embeddings = [[0.0] * self.dim for _ in batch]
                all_embeddings.extend(zero_embeddings)
        
        return all_embeddings


class SentenceTransformerEmbedder(TextEmbedder):
    """
    Text embedder using Sentence Transformers models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformers embedder.
        
        Args:
            model_name: Name of the model to use
        """
        super().__init__()
        if not sentence_transformers_available:
            raise ImportError("SentenceTransformers package is required. Install with 'pip install sentence-transformers'")
            
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Sentence Transformers"""
        if not texts:
            return []
            
        try:
            embeddings = self.model.encode(texts)
            # Convert numpy arrays to lists
            return embeddings.tolist()
        except Exception as e:
            print(f"Error getting embeddings from Sentence Transformers: {e}")
            return [[0.0] * self.dim for _ in texts]


class EmbeddingCache:
    """
    Cache for storing and retrieving text embeddings to avoid redundant API calls.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the embedding cache.
        
        Args:
            cache_file: File path to save/load the cache
        """
        self.cache = {}  # Maps text to embedding
        self.cache_file = cache_file
        
        # Load cache if file exists
        if cache_file and os.path.exists(cache_file):
            self.load_cache()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding for text if it exists in cache"""
        return self.cache.get(text)
    
    def put(self, text: str, embedding: List[float]):
        """Store embedding for text in cache"""
        self.cache[text] = embedding
    
    def save_cache(self, cache_file: Optional[str] = None):
        """Save cache to file"""
        file_path = cache_file or self.cache_file
        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                print(f"Error saving embedding cache: {e}")
    
    def load_cache(self, cache_file: Optional[str] = None):
        """Load cache from file"""
        file_path = cache_file or self.cache_file
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
                self.cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "memory_mb": sys.getsizeof(self.cache) / 1024 / 1024
        }


class VectorStore:
    """
    Abstract base class for vector stores.
    """
    
    def __init__(self):
        self.store_type = "base"
    
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add texts and their embeddings to the vector store.
        
        Args:
            texts: The text strings
            embeddings: The corresponding embedding vectors
            metadata: Optional metadata for each text
            
        Returns:
            List of IDs for the added texts
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar texts using a query embedding.
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of results to return
            
        Returns:
            List of dicts containing text, score, and metadata
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete texts from the vector store.
        
        Args:
            ids: IDs of texts to delete
            
        Returns:
            Success or failure
        """
        raise NotImplementedError("This method must be implemented by subclasses")


class FaissVectorStore(VectorStore):
    """
    Vector store implementation using FAISS.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        super().__init__()
        if not faiss_available:
            raise ImportError("FAISS package is required. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
            
        self.store_type = "faiss"
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        
        # Storage for text content and metadata
        self.texts = []
        self.ids = []
        self.metadata = []
    
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts and embeddings to FAISS index"""
        if not texts or not embeddings:
            return []
            
        # Prepare metadata if not provided
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # Create IDs for new entries
        new_ids = [f"id-{len(self.ids) + i}" for i in range(len(texts))]
        
        # Add to storage
        self.texts.extend(texts)
        self.ids.extend(new_ids)
        self.metadata.extend(metadata)
        
        # Convert embeddings to numpy and add to index
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        
        return new_ids
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search FAISS index with query embedding"""
        if not self.ids:
            return []
            
        # Convert query to numpy
        query_np = np.array([query_embedding]).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_np, min(top_k, len(self.ids)))
        
        # Build result
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts) and idx >= 0:  # Ensure valid index
                # Convert distance to similarity score (between 0 and 1)
                # L2 distance can be unbounded, so we normalize it
                distance = float(distances[0][i])
                similarity = 1.0 / (1.0 + distance)
                
                results.append({
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "score": similarity,  # Now between 0 and 1
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete entries from FAISS index"""
        # FAISS doesn't support easy deletion, so we rebuild the index
        if not ids:
            return True
            
        to_delete = set(ids)
        
        # Create new storage with items to keep
        new_texts = []
        new_ids = []
        new_metadata = []
        new_embeddings = []
        
        # Identify indices to keep
        for i, id_val in enumerate(self.ids):
            if id_val not in to_delete:
                new_texts.append(self.texts[i])
                new_ids.append(id_val)
                new_metadata.append(self.metadata[i])
                
                # We need to extract embeddings from FAISS index
                # This is a limitation - we don't store original embeddings
                # In a real system, you'd keep the original embeddings
                new_embeddings.append(self.index.reconstruct(i).tolist())
        
        # Reset state
        self.texts = new_texts
        self.ids = new_ids
        self.metadata = new_metadata
        
        # Create new index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to new index
        if new_embeddings:
            embeddings_np = np.array(new_embeddings).astype('float32')
            self.index.add(embeddings_np)
        
        return True


class RetrievalSystem:
    """
    Retrieval system to find relevant chunks for a query.
    Supports different retrieval methods including keywords and embeddings.
    """
    
    def __init__(self, method: str = "keyword"):
        """
        Initialize retrieval system.
        
        Args:
            method: Retrieval method ("keyword", "embedding", or "hybrid")
        """
        self.method = method
        self.embedder = None
        self.vector_store = None
        self.chunks = []
        self.is_initialized = False
        
    def initialize_embeddings(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize embedding model and vector store for embedding-based retrieval.
        
        Args:
            api_key: API key for embedding model (if using OpenAI)
            model_name: Name of the embedding model to use
        """
        # Default to OpenAI embeddings if available
        if openai_available:
            model = model_name or "text-embedding-3-small"
            self.embedder = OpenAIEmbedder(model_name=model, api_key=api_key)
        elif sentence_transformers_available:
            model = model_name or "all-MiniLM-L6-v2"
            self.embedder = SentenceTransformerEmbedder(model_name=model)
        else:
            raise ImportError("No embedding model available. Install either 'openai' or 'sentence-transformers'")
        
        # Initialize vector store
        if faiss_available:
            self.vector_store = FaissVectorStore(dimension=self.embedder.get_embedding_dimension())
        else:
            raise ImportError("No vector store available. Install 'faiss-cpu'")
    
    def add_chunks(self, chunks: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Add document chunks to the retrieval system.
        
        Args:
            chunks: List of text chunks
            metadata: Optional metadata for each chunk
        """
        self.chunks = chunks
        
        # For embedding-based methods, embed chunks and add to vector store
        if self.method in ["embedding", "hybrid"]:
            if not self.embedder:
                self.initialize_embeddings()
            
            # Embed chunks
            embeddings = self.embedder.embed(chunks)
            
            # Add to vector store
            self.vector_store.add_texts(chunks, embeddings, metadata)
        
        self.is_initialized = True
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with scores
        """
        if not self.is_initialized:
            raise ValueError("Retrieval system not initialized. Call add_chunks first.")
        
        if self.method == "keyword":
            return self._keyword_retrieval(query, top_k)
        elif self.method == "embedding":
            return self._embedding_retrieval(query, top_k)
        elif self.method == "hybrid":
            return self._hybrid_retrieval(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")
    
    def _keyword_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval"""
        # Split query into keywords
        keywords = query.lower().split()
        
        # Score chunks based on keyword matches
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            # Count matches
            matches = sum(1 for keyword in keywords if keyword in chunk_lower)
            # Normalize score to be between 0 and 1
            score = matches / max(len(keywords), 1) if keywords else 0
            scored_chunks.append({"id": f"chunk-{i}", "text": chunk, "score": score, "metadata": {}})
        
        # Sort by score (highest first) and get top_k chunks
        sorted_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)
        return sorted_chunks[:top_k]
    
    def _embedding_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Embedding-based retrieval using vector similarity"""
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)
        
        return results
    
    def _hybrid_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining keyword and embedding approaches"""
        # Get results from both methods
        keyword_results = self._keyword_retrieval(query, top_k * 2)
        embedding_results = self._embedding_retrieval(query, top_k * 2)
        
        # Combine results with weighted scores
        combined_results = {}
        
        # Weights for each method (must sum to 1.0)
        keyword_weight = 0.3
        embedding_weight = 0.7
        
        # Add keyword results with weight
        for result in keyword_results:
            combined_results[result["id"]] = {
                "id": result["id"],
                "text": result["text"],
                "score": result["score"] * keyword_weight,  # Weight for keyword score
                "metadata": result["metadata"],
                "components": {
                    "keyword": result["score"],
                    "embedding": 0.0
                }
            }
        
        # Add embedding results with weight
        for result in embedding_results:
            if result["id"] in combined_results:
                # If already in results, add weighted embedding score
                combined_results[result["id"]]["score"] += result["score"] * embedding_weight
                combined_results[result["id"]]["components"]["embedding"] = result["score"]
            else:
                # Add new result with weighted embedding score
                combined_results[result["id"]] = {
                    "id": result["id"],
                    "text": result["text"],
                    "score": result["score"] * embedding_weight,  # Weight for embedding score
                    "metadata": result["metadata"],
                    "components": {
                        "keyword": 0.0,
                        "embedding": result["score"]
                    }
                }
            
            # Ensure score is between 0 and 1
            combined_results[result["id"]]["score"] = min(combined_results[result["id"]]["score"], 1.0)
        
        # Sort by combined score and return top_k
        sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]


def compare_retrieval_methods(
    query: str,
    chunks: List[str],
    openai_api_key: Optional[str] = None,
    top_k: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare different retrieval methods for a query.
    
    Args:
        query: The query text
        chunks: List of document chunks
        openai_api_key: OpenAI API key for embedding model
        top_k: Number of chunks to retrieve
        
    Returns:
        Dictionary mapping method names to retrieval results
    """
    results = {}
    
    # Keyword retrieval
    keyword_retriever = RetrievalSystem(method="keyword")
    keyword_retriever.add_chunks(chunks)
    results["keyword"] = keyword_retriever.retrieve(query, top_k)
    
    # Embedding retrieval (if dependencies available)
    if openai_available or sentence_transformers_available:
        try:
            embedding_retriever = RetrievalSystem(method="embedding")
            embedding_retriever.initialize_embeddings(api_key=openai_api_key)
            embedding_retriever.add_chunks(chunks)
            results["embedding"] = embedding_retriever.retrieve(query, top_k)
            
            # Hybrid retrieval
            hybrid_retriever = RetrievalSystem(method="hybrid")
            hybrid_retriever.initialize_embeddings(api_key=openai_api_key)
            hybrid_retriever.add_chunks(chunks)
            results["hybrid"] = hybrid_retriever.retrieve(query, top_k)
        except Exception as e:
            print(f"Error with embedding retrieval: {e}")
            # Fall back to just keyword results
    
    return results


if __name__ == "__main__":
    # Simple example usage
    chunks = [
        "Artificial intelligence is revolutionizing many fields including healthcare and finance.",
        "Machine learning models require large amounts of training data to achieve good performance.",
        "Neural networks are a subset of machine learning inspired by the human brain.",
        "Transformers have become the dominant architecture for natural language processing tasks.",
        "Reinforcement learning is used for training agents to make decisions in complex environments."
    ]
    
    query = "How do neural networks work?"
    
    # Check if we can use embeddings
    can_use_embeddings = openai_available or sentence_transformers_available
    
    if can_use_embeddings:
        # Compare all methods
        results = compare_retrieval_methods(query, chunks)
        
        print(f"Query: {query}\n")
        for method, method_results in results.items():
            print(f"\n{method.upper()} RETRIEVAL:")
            for i, result in enumerate(method_results):
                print(f"{i+1}. Score: {result['score']:.4f}")
                print(f"   {result['text']}")
    else:
        # Fallback to keyword retrieval
        retriever = RetrievalSystem(method="keyword")
        retriever.add_chunks(chunks)
        results = retriever.retrieve(query, top_k=2)
        
        print(f"Query: {query}\n")
        print("KEYWORD RETRIEVAL:")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']}")
            print(f"   {result['text']}")