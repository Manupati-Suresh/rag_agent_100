#!/usr/bin/env python3
"""
Enhanced text highlighting and extraction utilities for RAG search results
"""

import re
from typing import List, Dict, Tuple, Optional, Set
import hashlib

class TextHighlighter:
    """
    Rule-based text highlighter for keyword matching
    """
    
    def __init__(self):
        # Highlighting styles
        self.highlight_styles = {
            'primary': 'background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: bold;',
        }
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode()).hexdigest()
        
    def extract_relevant_chunks(self, text: str, query: str, 
                              chunk_size: int = 200, 
                              overlap: int = 50, 
                              top_chunks: int = 3,
                              use_semantic_chunking: bool = False) -> List[Dict]:
        """
        Extract relevant chunks from text based on query keywords (rule-based)
        """
        # For rule-based, we simply return chunks containing query keywords
        keywords = self._extract_keywords(query)
        if not keywords:
            return [{'text': text[:chunk_size] + "...", 'start_pos': 0, 'end_pos': chunk_size, 'relevance_score': 0.1, 'keyword_count': 0}] if text else []
            
        all_chunks = self._create_chunks(text, chunk_size, overlap)
        relevant_chunks = []
        
        for chunk in all_chunks:
            score = 0
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', chunk['text'], re.IGNORECASE):
                    score += 1
            if score > 0:
                chunk['relevance_score'] = score / len(keywords) # Simple score based on keyword count
                chunk['keyword_count'] = score
                relevant_chunks.append(chunk)
                
        relevant_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_chunks[:top_chunks]
    
    def highlight_text(self, text: str, query: str, max_length: int = 500) -> str:
        """
        Main highlighting method used by RAG agent (rule-based)
        """
        return self.create_snippet(text, query, max_length=max_length)
    
    def highlight_keywords(self, text: str, query: str, style: str = 'primary') -> str:
        """
        Basic keyword highlighting - EXACT MATCH ONLY (rule-based)
        """
        keywords = self._extract_keywords(query)
        highlighted_text = text
        
        current_style = self.highlight_styles['primary']
        
        for keyword in keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b(?![^<]*</mark>)', re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<mark style="{current_style}">{keyword}</mark>',
                highlighted_text
            )
            
        return highlighted_text
    
    def extract_sentences_around_keywords(self, text: str, query: str,
                                        context_sentences: int = 2) -> List[str]:
        """
        Extract sentences that contain query keywords with surrounding context (rule-based)
        """
        keywords = self._extract_keywords(query)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        relevant_sentences_indices = set()
        
        for i, sentence in enumerate(sentences):
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', sentence, re.IGNORECASE) for keyword in keywords):
                start_idx = max(0, i - context_sentences)
                end_idx = min(len(sentences), i + context_sentences + 1)
                for j in range(start_idx, end_idx):
                    relevant_sentences_indices.add(j)
        
        result = []
        for i in sorted(list(relevant_sentences_indices)):
            result.append(sentences[i])
        
        return result
    
    def create_snippet(self, text: str, query: str, max_length: int = 300) -> str:
        """
        Create a snippet highlighting relevant parts (rule-based)
        """
        # Get relevant chunks using keyword matching
        chunks = self.extract_relevant_chunks(text, query, chunk_size=max_length, top_chunks=1)
        
        if not chunks:
            return self._create_fallback_snippet(text, query, max_length)
        
        chosen_text = chunks[0]['text']
        highlighted_text = self.highlight_keywords(chosen_text, query)
        
        if len(highlighted_text) > max_length:
            snippet = self._smart_trim_snippet(highlighted_text, max_length, query)
        else:
            snippet = highlighted_text
            
        return snippet.strip()
    
    def create_contextual_snippet(self, text: str, query: str, max_length: int = 400) -> Dict:
        """
        Create a rich snippet with context information (rule-based)
        """
        # For rule-based, simply use create_snippet and add dummy info
        snippet_text = self.create_snippet(text, query, max_length=max_length)
        
        return {
            'snippet': snippet_text,
            'start_pos': 0,
            'end_pos': len(snippet_text),
            'relevance_score': 1.0, # Placeholder
            'key_sentences': self.extract_sentences_around_keywords(text, query, context_sentences=1)[:3],
            'keyword_count': len(self._extract_keywords(query)),
            'has_more_content': len(text) > len(snippet_text)
        }
    
    def _create_fallback_snippet(self, text: str, query: str, max_length: int) -> str:
        """Create a fallback snippet when no good chunks are found"""
        keywords = self._extract_keywords(query)
        
        best_pos = 0
        for keyword in keywords:
            pos = text.lower().find(keyword.lower())
            if pos != -1:
                start_pos = max(0, pos - 50)
                best_pos = start_pos
                break
        
        snippet = text[best_pos:best_pos + max_length]
        
        if len(text) > best_pos + max_length:
            last_period = snippet.rfind('.')
            if last_period > max_length * 0.7:
                snippet = snippet[:last_period + 1]
            else:
                snippet += "..."
        
        return self.highlight_keywords(snippet, query)
    
    def _smart_trim_snippet(self, highlighted_text: str, max_length: int, query: str) -> str:
        """Smart trimming that preserves highlights and sentence boundaries"""
        if len(highlighted_text) <= max_length:
            return highlighted_text
        
        for i in range(max_length - 50, max_length):
            if i < len(highlighted_text) and highlighted_text[i] in '.!?':
                if '</mark>' not in highlighted_text[i:i+10]:
                    return highlighted_text[:i+1] + "..."
        
        for i in range(max_length - 20, max_length):
            if i < len(highlighted_text) and highlighted_text[i] == ' ':
                if '</mark>' not in highlighted_text[i:i+10]:
                    return highlighted_text[:i] + "..."
        
        return highlighted_text[:max_length] + "..."
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Create overlapping text chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size * 0.7: 
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
        
        return [chunk for chunk in chunks if len(chunk['text']) > 20]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract basic keywords from query"""
        words = re.findall(r'\b\w+\b', query.lower())
        # Simple stop words for rule-based
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'in', 'of', 'for'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords