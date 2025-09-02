#!/usr/bin/env python3
"""
Enhanced text highlighting and extraction utilities for RAG search results
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import Counter
import hashlib
import json
from functools import lru_cache

class TextHighlighter:
    """
    Enhanced text highlighter with advanced NLP features and performance optimizations
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', auto_setup_nltk: bool = True):
        self.model = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # Initialize NLTK components (download if needed)
        self._init_nltk(auto_setup_nltk)
        
        # Cache for expensive operations
        self._embedding_cache = {}
        self._keyword_cache = {}
        
        # Enhanced stop words
        self.stop_words = self._get_enhanced_stop_words()
        
        # Highlighting styles
        self.highlight_styles = {
            'primary': 'background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: bold;',
            'secondary': 'background-color: #e1f5fe; padding: 1px 3px; border-radius: 2px;',
            'phrase': 'background-color: #f3e5f5; padding: 2px 4px; border-radius: 3px; border-left: 3px solid #9c27b0;'
        }
    
    def _init_nltk(self, auto_setup: bool = True):
        """Initialize NLTK components"""
        try:
            import nltk
            
            # Try to import required components first
            try:
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                from nltk.tokenize import word_tokenize, sent_tokenize
                
                # Test if they work
                test_stops = stopwords.words('english')
                test_lemmatizer = WordNetLemmatizer()
                test_tokens = word_tokenize("test sentence")
                test_sentences = sent_tokenize("Test sentence. Another sentence.")
                
                # If we get here, everything works
                self.nltk_stopwords = set(test_stops)
                self.lemmatizer = test_lemmatizer
                self.word_tokenize = word_tokenize
                self.sent_tokenize = sent_tokenize
                self.nltk_available = True
                print("✅ NLTK initialized successfully with all components")
                return
                
            except Exception as e:
                if auto_setup:
                    print(f"⚠️ NLTK data missing. Attempting to download... Error: {str(e)}")
                    
                    # Download required NLTK data
                    resources_to_download = [
                        'punkt', 'punkt_tab', 'stopwords', 'wordnet', 
                        'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
                        'omw-1.4'
                    ]
                    
                    for resource in resources_to_download:
                        try:
                            nltk.download(resource, quiet=True)
                        except:
                            pass
                    
                    # Try again after download
                    try:
                        from nltk.corpus import stopwords
                        from nltk.stem import WordNetLemmatizer
                        from nltk.tokenize import word_tokenize, sent_tokenize
                        
                        self.nltk_stopwords = set(stopwords.words('english'))
                        self.lemmatizer = WordNetLemmatizer()
                        self.word_tokenize = word_tokenize
                        self.sent_tokenize = sent_tokenize
                        
                        # Test again
                        test_tokens = word_tokenize("test sentence")
                        test_sentences = sent_tokenize("Test sentence. Another sentence.")
                        
                        self.nltk_available = True
                        print("✅ NLTK initialized successfully after download")
                        return
                        
                    except Exception as e2:
                        print(f"⚠️ NLTK still not working after download: {str(e2)}")
                
                # If we get here, NLTK is not working
                print("ℹ️ Using basic text processing (NLTK unavailable)")
                self.nltk_available = False
                
        except ImportError:
            print("ℹ️ NLTK not installed. Using basic text processing.")
            self.nltk_available = False
        except Exception as e:
            print(f"⚠️ NLTK initialization failed: {str(e)}. Using basic text processing.")
            self.nltk_available = False
    
    def _get_enhanced_stop_words(self) -> Set[str]:
        """Get comprehensive stop words list"""
        basic_stops = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when',
            'where', 'why', 'who', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'myself', 'yourself', 'himself', 'herself', 'itself',
            'ourselves', 'yourselves', 'themselves'
        }
        
        if self.nltk_available:
            basic_stops.update(self.nltk_stopwords)
        
        return basic_stops
    
    @lru_cache(maxsize=128)
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode()).hexdigest()
        
    def extract_relevant_chunks(self, text: str, query: str, 
                              chunk_size: int = 200, 
                              overlap: int = 50, 
                              top_chunks: int = 3,
                              use_semantic_chunking: bool = True) -> List[Dict]:
        """
        Extract the most relevant chunks from text based on query similarity
        
        Args:
            text: Full document text
            query: Search query
            chunk_size: Size of each text chunk in characters
            overlap: Overlap between chunks in characters
            top_chunks: Number of top relevant chunks to return
            use_semantic_chunking: Use semantic boundaries for chunking
            
        Returns:
            List of dictionaries with chunk info and relevance scores
        """
        # Use semantic chunking if available and requested
        if use_semantic_chunking and self.nltk_available:
            chunks = self._create_semantic_chunks(text, chunk_size, overlap)
        else:
            chunks = self._create_chunks(text, chunk_size, overlap)
        
        if not chunks:
            return []
        
        # Extract enhanced keywords from query
        enhanced_keywords = self._extract_enhanced_keywords(query)
        
        # Calculate semantic similarity scores
        chunk_texts = [chunk['text'] for chunk in chunks]
        semantic_scores = self._calculate_semantic_similarity(chunk_texts, query)
        
        # Calculate TF-IDF similarity scores
        tfidf_scores = self._calculate_tfidf_similarity(chunk_texts, query)
        
        # Calculate keyword density scores
        keyword_scores = self._calculate_keyword_density_scores(chunk_texts, enhanced_keywords)
        
        # Combine scores with enhanced weighting
        combined_scores = []
        for i in range(len(chunks)):
            # Weighted combination: semantic (50%), TF-IDF (30%), keyword density (20%)
            combined_score = (0.5 * semantic_scores[i] + 
                            0.3 * tfidf_scores[i] + 
                            0.2 * keyword_scores[i])
            combined_scores.append(combined_score)
            chunks[i]['relevance_score'] = combined_score
            chunks[i]['semantic_score'] = semantic_scores[i]
            chunks[i]['tfidf_score'] = tfidf_scores[i]
            chunks[i]['keyword_score'] = keyword_scores[i]
            chunks[i]['enhanced_keywords'] = enhanced_keywords
        
        # Sort by relevance and return top chunks
        chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return chunks[:top_chunks]
    
    def highlight_text(self, text: str, query: str, max_length: int = 500) -> str:
        """
        Main highlighting method used by RAG agent
        Creates a highlighted snippet of the text based on the query
        """
        return self.create_snippet(text, query, max_length=max_length, use_advanced_highlighting=False)
    
    def highlight_keywords(self, text: str, query: str, style: str = 'primary') -> str:
        """
        Basic keyword highlighting - EXACT MATCH ONLY
        """
        # Extract only basic keywords (no NLP expansion)
        basic_keywords = self._extract_keywords(query)
        
        highlighted_text = text
        already_highlighted = set()
        
        # Highlight only exact query terms
        for keyword in basic_keywords:
            # Skip if already highlighted
            if keyword.lower() in already_highlighted:
                continue
            
            # Use primary style for exact matches
            current_style = self.highlight_styles['primary']
            
            # Case-insensitive highlighting with word boundaries, avoiding existing highlights
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b(?![^<]*</mark>)', re.IGNORECASE)
            if pattern.search(highlighted_text):
                highlighted_text = pattern.sub(
                    f'<mark style="{current_style}">{keyword}</mark>',
                    highlighted_text
                )
                already_highlighted.add(keyword.lower())
        
        return highlighted_text
    
    def highlight_keywords_enhanced(self, text: str, query: str, style: str = 'primary') -> str:
        """
        Enhanced keyword highlighting with phrase detection and multiple styles
        """
        # Extract enhanced keywords and phrases
        enhanced_keywords = self._extract_enhanced_keywords(query)
        phrases = self._extract_phrases(query)
        
        highlighted_text = text
        
        # First, highlight phrases (longer matches first)
        phrases_sorted = sorted(phrases, key=len, reverse=True)
        for phrase in phrases_sorted:
            if len(phrase.split()) > 1:  # Multi-word phrases
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                highlighted_text = pattern.sub(
                    f'<mark style="{self.highlight_styles["phrase"]}">{phrase}</mark>',
                    highlighted_text
                )
        
        # Then highlight individual keywords (avoid nested highlights)
        already_highlighted = set()
        
        for keyword_info in enhanced_keywords:
            keyword = keyword_info['word']
            importance = keyword_info['importance']
            
            # Skip if already highlighted
            if keyword.lower() in already_highlighted:
                continue
            
            # Choose style based on importance
            if importance > 0.7:
                current_style = self.highlight_styles['primary']
            else:
                current_style = self.highlight_styles['secondary']
            
            # Case-insensitive highlighting with word boundaries, avoiding existing highlights
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b(?![^<]*</mark>)', re.IGNORECASE)
            if pattern.search(highlighted_text):
                highlighted_text = pattern.sub(
                    f'<mark style="{current_style}">{keyword}</mark>',
                    highlighted_text
                )
                already_highlighted.add(keyword.lower())
        
        return highlighted_text
    
    def highlight_keywords_advanced(self, text: str, query: str, 
                                  include_synonyms: bool = True,
                                  include_related: bool = True) -> str:
        """
        Advanced highlighting with synonym and related term detection
        """
        highlighted_text = text
        
        # Get base keywords
        enhanced_keywords = self._extract_enhanced_keywords(query)
        
        # Add synonyms and related terms if requested
        if include_synonyms or include_related:
            expanded_keywords = self._expand_keywords(enhanced_keywords, 
                                                    include_synonyms, 
                                                    include_related)
            enhanced_keywords.extend(expanded_keywords)
        
        # Sort by importance and length
        enhanced_keywords.sort(key=lambda x: (x['importance'], len(x['word'])), reverse=True)
        
        # Apply highlighting (avoid nested highlights)
        already_highlighted = set()
        
        for keyword_info in enhanced_keywords:
            keyword = keyword_info['word']
            importance = keyword_info['importance']
            is_expanded = keyword_info.get('is_expanded', False)
            
            # Skip if already highlighted
            if keyword.lower() in already_highlighted:
                continue
            
            # Choose style
            if is_expanded:
                style = self.highlight_styles['secondary']
            elif importance > 0.7:
                style = self.highlight_styles['primary']
            else:
                style = self.highlight_styles['secondary']
            
            # Apply highlighting with word boundaries, avoiding existing highlights
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b(?![^<]*</mark>)', re.IGNORECASE)
            if pattern.search(highlighted_text):
                highlighted_text = pattern.sub(
                    f'<mark style="{style}" title="Relevance: {importance:.2f}">{keyword}</mark>',
                    highlighted_text
                )
                already_highlighted.add(keyword.lower())
        
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
    
    def create_snippet(self, text: str, query: str, max_length: int = 300,
                      style: str = 'primary', use_advanced_highlighting: bool = True,
                      include_synonyms: bool = True, include_related: bool = False) -> str:
        """
        Create an intelligent snippet highlighting the most relevant parts
        """
        # Get relevant chunks with enhanced scoring
        chunks = self.extract_relevant_chunks(text, query, chunk_size=200, top_chunks=2)
        
        if not chunks:
            # Fallback to keyword-based snippet
            return self._create_fallback_snippet(text, query, max_length)
        
        # Combine top chunks if they fit
        combined_text = ""
        for chunk in chunks:
            if len(combined_text + chunk['text']) <= max_length * 1.2:  # Allow 20% overflow for better context
                if combined_text:
                    combined_text += " ... " + chunk['text']
                else:
                    combined_text = chunk['text']
            else:
                break
        
        # If combined text is too long, use just the best chunk
        if len(combined_text) > max_length * 1.2:
            combined_text = chunks[0]['text']
        
        # Apply highlighting
        if use_advanced_highlighting:
            highlighted_text = self.highlight_keywords_advanced(
                combined_text, query, include_synonyms=include_synonyms, include_related=include_related
            )
        else:
            highlighted_text = self.highlight_keywords(combined_text, query, style)
        
        # Smart trimming while preserving highlights
        if len(combined_text) > max_length:
            snippet = self._smart_trim_snippet(highlighted_text, max_length, query)
        else:
            snippet = highlighted_text
        
        return snippet.strip()
    
    def create_contextual_snippet(self, text: str, query: str, max_length: int = 400,
                                include_synonyms: bool = True, include_related: bool = False) -> Dict:
        """
        Create a rich snippet with context information
        """
        # Get enhanced keywords
        enhanced_keywords = self._extract_enhanced_keywords(query)
        
        # Find the best passage around keywords
        best_passage = self._find_best_passage(text, enhanced_keywords, max_length)
        
        # Create highlighted snippet
        highlighted_snippet = self.highlight_keywords_advanced(
            best_passage['text'], query, include_synonyms=include_synonyms, include_related=include_related
        )
        
        # Extract key sentences
        key_sentences = self.extract_sentences_around_keywords(text, query, context_sentences=1)
        
        return {
            'snippet': highlighted_snippet,
            'start_pos': best_passage['start_pos'],
            'end_pos': best_passage['end_pos'],
            'relevance_score': best_passage['relevance_score'],
            'key_sentences': key_sentences[:3],
            'keyword_count': best_passage['keyword_count'],
            'has_more_content': len(text) > best_passage['end_pos']
        }
    
    def _create_fallback_snippet(self, text: str, query: str, max_length: int) -> str:
        """Create a fallback snippet when no good chunks are found"""
        # Look for first occurrence of any query keyword
        keywords = self._extract_keywords(query)
        
        best_pos = 0
        for keyword in keywords:
            pos = text.lower().find(keyword.lower())
            if pos != -1:
                # Start snippet a bit before the keyword
                start_pos = max(0, pos - 50)
                best_pos = start_pos
                break
        
        # Extract snippet
        snippet = text[best_pos:best_pos + max_length]
        
        # Try to end at sentence boundary
        if len(text) > best_pos + max_length:
            last_period = snippet.rfind('.')
            if last_period > max_length * 0.7:
                snippet = snippet[:last_period + 1]
            else:
                snippet += "..."
        
        # Highlight keywords
        return self.highlight_keywords(snippet, query)
    
    def _smart_trim_snippet(self, highlighted_text: str, max_length: int, query: str) -> str:
        """Smart trimming that preserves highlights and sentence boundaries"""
        if len(highlighted_text) <= max_length:
            return highlighted_text
        
        # Try to find a good breaking point
        # Priority: sentence end > highlight boundary > word boundary
        
        # Look for sentence endings within the limit
        for i in range(max_length - 50, max_length):
            if i < len(highlighted_text) and highlighted_text[i] in '.!?':
                # Check if we're not in the middle of a highlight
                if '</mark>' not in highlighted_text[i:i+10]:
                    return highlighted_text[:i+1]
        
        # Look for word boundaries
        for i in range(max_length - 20, max_length):
            if i < len(highlighted_text) and highlighted_text[i] == ' ':
                if '</mark>' not in highlighted_text[i:i+10]:
                    return highlighted_text[:i] + "..."
        
        # Last resort: cut at max_length
        return highlighted_text[:max_length] + "..."
    
    def _find_best_passage(self, text: str, enhanced_keywords: List[Dict], 
                          max_length: int) -> Dict:
        """Find the best passage containing the most important keywords"""
        best_passage = {
            'text': text[:max_length],
            'start_pos': 0,
            'end_pos': min(max_length, len(text)),
            'relevance_score': 0.0,
            'keyword_count': 0
        }
        
        # Sliding window approach
        window_size = max_length
        step_size = max_length // 4
        
        for start in range(0, len(text) - window_size + 1, step_size):
            end = start + window_size
            passage = text[start:end]
            
            # Calculate relevance score for this passage
            score = 0.0
            keyword_count = 0
            
            for keyword_info in enhanced_keywords:
                keyword = keyword_info['word']
                importance = keyword_info['importance']
                
                # Count occurrences
                pattern = r'\b' + re.escape(keyword) + r'\b'
                count = len(re.findall(pattern, passage.lower()))
                
                if count > 0:
                    keyword_count += count
                    score += count * importance
            
            # Normalize by passage length
            normalized_score = score / len(passage.split())
            
            if normalized_score > best_passage['relevance_score']:
                best_passage = {
                    'text': passage,
                    'start_pos': start,
                    'end_pos': end,
                    'relevance_score': normalized_score,
                    'keyword_count': keyword_count
                }
        
        return best_passage
    
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
        """Extract basic keywords from query (legacy method)"""
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if len(word) > 2 and word not in self.stop_words]
        return keywords
    
    def _extract_enhanced_keywords(self, query: str) -> List[Dict]:
        """
        Extract enhanced keywords with importance scores and linguistic analysis
        """
        # Check cache first
        query_hash = self._get_text_hash(query)
        if query_hash in self._keyword_cache:
            return self._keyword_cache[query_hash]
        
        enhanced_keywords = []
        
        if self.nltk_available:
            # Use NLTK for advanced processing
            tokens = self._tokenize_words(query)
            
            # POS tagging to identify important word types
            try:
                import nltk
                pos_tags = nltk.pos_tag(tokens)
                
                for word, pos in pos_tags:
                    if (len(word) > 2 and 
                        word not in self.stop_words and 
                        word.isalpha()):
                        
                        # Calculate importance based on POS tag
                        importance = self._calculate_word_importance(word, pos)
                        
                        # Lemmatize the word
                        try:
                            lemmatized = self.lemmatizer.lemmatize(word)
                        except:
                            lemmatized = word
                        
                        enhanced_keywords.append({
                            'word': word,
                            'lemmatized': lemmatized,
                            'pos': pos,
                            'importance': importance,
                            'is_expanded': False
                        })
            except Exception as e:
                print(f"⚠️ POS tagging failed: {str(e)}. Using basic extraction.")
                # Fallback to basic extraction
                for word in tokens:
                    if len(word) > 2 and word not in self.stop_words and word.isalpha():
                        enhanced_keywords.append({
                            'word': word,
                            'lemmatized': word,
                            'pos': 'UNKNOWN',
                            'importance': 0.6,  # Slightly higher than basic
                            'is_expanded': False
                        })
        else:
            # Basic extraction without NLTK
            tokens = self._tokenize_words(query)
            for word in tokens:
                if len(word) > 2 and word not in self.stop_words and word.isalpha():
                    enhanced_keywords.append({
                        'word': word,
                        'lemmatized': word,
                        'pos': 'UNKNOWN',
                        'importance': 0.5,
                        'is_expanded': False
                    })
        
        # Cache the result
        self._keyword_cache[query_hash] = enhanced_keywords
        
        return enhanced_keywords
    
    def _extract_phrases(self, query: str) -> List[str]:
        """Extract meaningful phrases from query"""
        phrases = []
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        phrases.extend(quoted_phrases)
        
        # Extract noun phrases (basic approach)
        if self.nltk_available:
            try:
                import nltk
                tokens = self._tokenize_words(query)
                pos_tags = nltk.pos_tag(tokens)
                
                # Simple noun phrase extraction
                current_phrase = []
                for word, pos in pos_tags:
                    if pos.startswith('NN') or pos.startswith('JJ'):  # Nouns and adjectives
                        current_phrase.append(word)
                    else:
                        if len(current_phrase) > 1:
                            phrases.append(' '.join(current_phrase))
                        current_phrase = []
                
                # Don't forget the last phrase
                if len(current_phrase) > 1:
                    phrases.append(' '.join(current_phrase))
            except Exception as e:
                print(f"⚠️ Phrase extraction failed: {str(e)}")
                pass
        
        # Remove duplicates and filter
        phrases = list(set(phrases))
        phrases = [p for p in phrases if len(p.split()) > 1 and len(p) > 5]
        
        return phrases
    
    def _calculate_word_importance(self, word: str, pos: str) -> float:
        """Calculate importance score for a word based on POS tag"""
        # Importance weights for different POS tags
        pos_weights = {
            'NN': 0.9,    # Noun
            'NNS': 0.9,   # Plural noun
            'NNP': 0.95,  # Proper noun
            'NNPS': 0.95, # Plural proper noun
            'VB': 0.7,    # Verb
            'VBD': 0.7,   # Past tense verb
            'VBG': 0.7,   # Gerund
            'VBN': 0.7,   # Past participle
            'VBP': 0.7,   # Present tense verb
            'VBZ': 0.7,   # 3rd person singular verb
            'JJ': 0.8,    # Adjective
            'JJR': 0.8,   # Comparative adjective
            'JJS': 0.8,   # Superlative adjective
            'RB': 0.6,    # Adverb
            'RBR': 0.6,   # Comparative adverb
            'RBS': 0.6,   # Superlative adverb
        }
        
        # Get base importance from POS tag
        base_importance = pos_weights.get(pos[:2], 0.4)
        
        # Adjust based on word length (longer words often more specific)
        length_bonus = min(0.2, len(word) * 0.02)
        
        # Adjust based on capitalization (proper nouns)
        cap_bonus = 0.1 if word[0].isupper() else 0.0
        
        return min(1.0, base_importance + length_bonus + cap_bonus)
    
    def _expand_keywords(self, keywords: List[Dict], 
                        include_synonyms: bool = True,
                        include_related: bool = True) -> List[Dict]:
        """Expand keywords with synonyms and related terms"""
        expanded = []
        
        if not self.nltk_available:
            return expanded
        
        try:
            from nltk.corpus import wordnet
            
            for keyword_info in keywords:
                word = keyword_info['word']
                
                if include_synonyms:
                    # Get synonyms from WordNet
                    synonyms = set()
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace('_', ' ')
                            if synonym != word and len(synonym) > 2:
                                synonyms.add(synonym)
                    
                    # Add synonyms with reduced importance
                    for synonym in list(synonyms)[:3]:  # Limit to top 3
                        expanded.append({
                            'word': synonym,
                            'lemmatized': synonym,
                            'pos': keyword_info['pos'],
                            'importance': keyword_info['importance'] * 0.7,
                            'is_expanded': True,
                            'expansion_type': 'synonym'
                        })
                
                if include_related:
                    # Get related terms (hypernyms, hyponyms)
                    related = set()
                    for syn in wordnet.synsets(word):
                        # Hypernyms (more general terms)
                        for hyper in syn.hypernyms():
                            for lemma in hyper.lemmas():
                                related_term = lemma.name().replace('_', ' ')
                                if related_term != word and len(related_term) > 2:
                                    related.add(related_term)
                        
                        # Hyponyms (more specific terms)
                        for hypo in syn.hyponyms():
                            for lemma in hypo.lemmas():
                                related_term = lemma.name().replace('_', ' ')
                                if related_term != word and len(related_term) > 2:
                                    related.add(related_term)
                    
                    # Add related terms with reduced importance
                    for related_term in list(related)[:2]:  # Limit to top 2
                        expanded.append({
                            'word': related_term,
                            'lemmatized': related_term,
                            'pos': keyword_info['pos'],
                            'importance': keyword_info['importance'] * 0.5,
                            'is_expanded': True,
                            'expansion_type': 'related'
                        })
        except:
            pass  # WordNet not available or other error
        
        return expanded
    
    def _calculate_keyword_density_scores(self, chunks: List[str], 
                                        enhanced_keywords: List[Dict]) -> List[float]:
        """Calculate keyword density scores for chunks"""
        scores = []
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            total_score = 0.0
            total_words = len(chunk.split())
            
            for keyword_info in enhanced_keywords:
                keyword = keyword_info['word']
                importance = keyword_info['importance']
                
                # Count occurrences (with word boundaries)
                pattern = r'\b' + re.escape(keyword) + r'\b'
                count = len(re.findall(pattern, chunk_lower))
                
                # Calculate density score
                if count > 0:
                    density = count / total_words
                    total_score += density * importance
            
            scores.append(total_score)
        
        # Normalize scores
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [score / max_score for score in scores]
        
        return scores
    
    def _create_semantic_chunks(self, text: str, target_size: int, overlap: int) -> List[Dict]:
        """Create chunks based on semantic boundaries (sentences, paragraphs)"""
        chunks = []
        
        if not self.nltk_available:
            return self._create_chunks(text, target_size, overlap)
        
        try:
            # Split into sentences
            sentences = self._split_into_sentences(text)
            
            current_chunk = ""
            current_start = 0
            
            for sentence in sentences:
                # Check if adding this sentence would exceed target size
                if len(current_chunk + sentence) > target_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'text': current_chunk.strip(),
                        'start_pos': current_start,
                        'end_pos': current_start + len(current_chunk),
                        'length': len(current_chunk.strip()),
                        'sentence_count': len(self._split_into_sentences(current_chunk))
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                    current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence) - 1
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # Don't forget the last chunk
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_pos': current_start,
                    'end_pos': current_start + len(current_chunk),
                    'length': len(current_chunk.strip()),
                    'sentence_count': len(self._split_into_sentences(current_chunk))
                })
        
        except Exception as e:
            print(f"⚠️ Semantic chunking failed: {str(e)}. Using character-based chunking.")
            # Fallback to character-based chunking
            return self._create_chunks(text, target_size, overlap)
        
        return [chunk for chunk in chunks if len(chunk['text']) > 20]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.nltk_available:
            try:
                return self.sent_tokenize(text)
            except:
                pass
        
        # Fallback: Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if self.nltk_available:
            try:
                return self.word_tokenize(text.lower())
            except:
                pass
        
        # Fallback: Simple word tokenization
        return re.findall(r'\b\w+\b', text.lower())