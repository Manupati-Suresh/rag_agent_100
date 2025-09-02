#!/usr/bin/env python3
"""
Text highlighting and extraction utilities for RAG search results
"""

import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextHighlighter:
    """
    Extracts and highlights relevant portions of documents based on query similarity
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def extract_relevant_chunks(self, text: str, query: str, 
                              chunk_size: int = 200, 
                              overlap: int = 50, 
                              top_chunks: int = 3) -> List[Dict]:
        """
        Extract the most relevant chunks from text based on query similarity
        
        Args:
            text: Full document text
            query: Search query
            chunk_size: Size of each text chunk in characters
            overlap: Overlap between chunks in characters
            top_chunks: Number of top relevant chunks to return
            
        Returns:
            List of dictionaries with chunk info and relevance scores
        """
        # Split text into overlapping chunks
        chunks = self._create_chunks(text, chunk_size, overlap)
        
        if not chunks:
            return []
        
        # Calculate semantic similarity scores
        chunk_texts = [chunk['text'] for chunk in chunks]
        semantic_scores = self._calculate_semantic_similarity(chunk_texts, query)
        
        # Calculate TF-IDF similarity scores
        tfidf_scores = self._calculate_tfidf_similarity(chunk_texts, query)
        
        # Combine scores (weighted average)
        combined_scores = []
        for i in range(len(chunks)):
            combined_score = 0.7 * semantic_scores[i] + 0.3 * tfidf_scores[i]
            combined_scores.append(combined_score)
            chunks[i]['relevance_score'] = combined_score
            chunks[i]['semantic_score'] = semantic_scores[i]
            chunks[i]['tfidf_score'] = tfidf_scores[i]
        
        # Sort by relevance and return top chunks
        chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return chunks[:top_chunks]
    
    def highlight_keywords(self, text: str, query: str) -> str:
        """
        Highlight query keywords in text using HTML markup
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        highlighted_text = text
        for keyword in keywords:
            # Case-insensitive highlighting
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">{keyword}</mark>',
                highlighted_text
            )
        
        return highlighted_text
    
    def extract_sentences_around_keywords(self, text: str, query: str, 
                                        context_sentences: int = 2) -> List[str]:
        """
        Extract sentences that contain query keywords with surrounding context
        """
        keywords = self._extract_keywords(query)
        sentences = self._split_into_sentences(text)
        
        relevant_sentences = set()
        
        for i, sentence in enumerate(sentences):
            # Check if sentence contains any keywords
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                # Add the sentence and surrounding context
                start_idx = max(0, i - context_sentences)
                end_idx = min(len(sentences), i + context_sentences + 1)
                
                for j in range(start_idx, end_idx):
                    relevant_sentences.add(j)
        
        # Return sentences in order
        result = []
        for i in sorted(relevant_sentences):
            result.append(sentences[i])
        
        return result
    
    def create_snippet(self, text: str, query: str, max_length: int = 300) -> str:
        """
        Create a concise snippet highlighting the most relevant part
        """
        # Get relevant chunks
        chunks = self.extract_relevant_chunks(text, query, chunk_size=150, top_chunks=1)
        
        if not chunks:
            # Fallback to first part of text
            snippet = text[:max_length]
            if len(text) > max_length:
                snippet += "..."
            return snippet
        
        # Use the most relevant chunk
        best_chunk = chunks[0]['text']
        
        # Highlight keywords in the chunk
        highlighted_chunk = self.highlight_keywords(best_chunk, query)
        
        # Trim if too long
        if len(best_chunk) > max_length:
            # Try to keep complete sentences
            sentences = self._split_into_sentences(highlighted_chunk)
            snippet = ""
            for sentence in sentences:
                if len(snippet + sentence) <= max_length:
                    snippet += sentence + " "
                else:
                    break
            
            if not snippet:  # If no complete sentence fits
                snippet = highlighted_chunk[:max_length] + "..."
        else:
            snippet = highlighted_chunk
        
        return snippet.strip()
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Create overlapping text chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to end at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size * 0.7:  # At least 70% of chunk size
                    end = start + boundary + 1
                    chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text.strip(),
                'start_pos': start,
                'end_pos': end,
                'length': len(chunk_text.strip())
            })
            
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if len(chunk['text']) > 20]  # Filter very short chunks
    
    def _calculate_semantic_similarity(self, chunks: List[str], query: str) -> List[float]:
        """Calculate semantic similarity using sentence transformers"""
        if not chunks:
            return []
        
        # Encode chunks and query
        chunk_embeddings = self.model.encode(chunks)
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        return similarities.tolist()
    
    def _calculate_tfidf_similarity(self, chunks: List[str], query: str) -> List[float]:
        """Calculate TF-IDF similarity"""
        if not chunks:
            return []
        
        try:
            # Fit TF-IDF on chunks + query
            all_texts = chunks + [query]
            tfidf_matrix = self.tfidf.fit_transform(all_texts)
            
            # Calculate similarity between query and each chunk
            query_vector = tfidf_matrix[-1]  # Last item is the query
            chunk_vectors = tfidf_matrix[:-1]  # All except last
            
            similarities = cosine_similarity(query_vector, chunk_vectors)[0]
            
            return similarities.tolist()
        except:
            # Fallback to zero similarity if TF-IDF fails
            return [0.0] * len(chunks)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
        
        # Split query into words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]