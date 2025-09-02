from document_store import DocumentStore
from document_loader import DocumentLoader
from text_highlighter import TextHighlighter
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

class RAGAgent:
    """
    RAG Agent that combines document retrieval with generation capabilities
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', storage_path: str = 'document_storage'):
        self.document_store = DocumentStore(model_name, storage_path)
        self.text_highlighter = TextHighlighter(model_name)
        self.is_initialized = False
        self.storage_path = storage_path
        
        # Initialize Gemini
        self._setup_gemini()
        
        # Try to load existing documents
        self.load_existing_documents()
        
    def _setup_gemini(self):
        """
        Setup Google Gemini API
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Warning: GOOGLE_API_KEY not found in environment variables")
            self.gemini_model = None
            return
            
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize Gemini 2.5 Flash model
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Test the connection
            test_response = self.gemini_model.generate_content("Hello")
            print("✅ Gemini 2.5 Flash initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing Gemini: {str(e)}")
            self.gemini_model = None
        
    def load_documents(self, documents: List[Dict] = None, directory: str = None, max_docs: int = 100):
        """
        Load documents into the agent
        """
        if documents:
            # Load provided documents
            for doc in documents[:max_docs]:
                self.document_store.add_document(
                    doc_id=doc['id'],
                    content=doc['content'],
                    metadata=doc.get('metadata', {})
                )
        elif directory:
            # Load from directory
            docs = DocumentLoader.load_from_directory(directory, max_docs)
            for doc in docs:
                self.document_store.add_document(
                    doc_id=doc['id'],
                    content=doc['content'],
                    metadata=doc['metadata']
                )
        else:
            # Load sample documents for demo
            sample_docs = DocumentLoader.create_sample_documents()
            for doc in sample_docs:
                self.document_store.add_document(
                    doc_id=doc['id'],
                    content=doc['content'],
                    metadata=doc['metadata']
                )
                
        print(f"Loaded {len(self.document_store.documents)} documents")
        
    def initialize(self):
        """
        Build the search index
        """
        if not self.document_store.documents:
            raise ValueError("No documents loaded. Call load_documents() first")
            
        self.document_store.build_index()
        self.is_initialized = True
        print("RAG Agent initialized successfully!")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search on the document collection
        """
        if not self.is_initialized:
            raise ValueError("Agent not initialized. Call initialize() first")
            
        results = self.document_store.search(query, top_k)
        return results
        
    def generate_response(self, query: str, top_k: int = 3, use_llm: bool = True) -> Dict:
        """
        Generate a response using retrieved documents with optional LLM enhancement
        """
        # Retrieve relevant documents
        retrieved_docs = self.search(query, top_k)
        
        # Create context from retrieved documents
        context = "\n\n".join([
            f"Document {doc['rank']}: {doc['content']}"
            for doc in retrieved_docs
        ])
        
        response = {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'context': context,
            'summary': f"Found {len(retrieved_docs)} relevant documents for your query: '{query}'"
        }
        
        # Generate LLM response if available and requested
        if use_llm and self.gemini_model:
            try:
                llm_response = self._generate_llm_response(query, context)
                response['llm_response'] = llm_response
                response['has_llm_response'] = True
            except Exception as e:
                response['llm_error'] = str(e)
                response['has_llm_response'] = False
        else:
            response['has_llm_response'] = False
        
        return response
        
    def _generate_llm_response(self, query: str, context: str) -> str:
        """
        Generate response using Gemini based on query and retrieved context
        """
        prompt = f"""Based on the following context documents, please provide a comprehensive and accurate answer to the user's question.

Context Documents:
{context}

User Question: {query}

Instructions:
- Use only the information provided in the context documents
- If the context doesn't contain enough information to answer the question, say so clearly
- Provide specific details and examples from the documents when relevant
- Structure your response clearly and concisely
- If multiple documents provide different perspectives, acknowledge this

Answer:"""

        response = self.gemini_model.generate_content(prompt)
        return response.text
        
    def generate_summary(self, query: str, top_k: int = 5) -> Dict:
        """
        Generate a summary of retrieved documents using Gemini
        """
        if not self.gemini_model:
            return {'error': 'Gemini model not available'}
            
        # Retrieve relevant documents
        retrieved_docs = self.search(query, top_k)
        
        # Create context
        context = "\n\n".join([
            f"Document {doc['rank']}: {doc['content']}"
            for doc in retrieved_docs
        ])
        
        prompt = f"""Please create a comprehensive summary of the following documents related to the query: "{query}"

Documents:
{context}

Instructions:
- Create a well-structured summary that captures the key points
- Organize information logically
- Highlight the most important insights
- Keep the summary concise but informative
- Use bullet points or sections if helpful

Summary:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            return {
                'query': query,
                'summary': response.text,
                'source_documents': len(retrieved_docs),
                'success': True
            }
        except Exception as e:
            return {
                'query': query,
                'error': str(e),
                'success': False
            }
            
    def ask_question(self, question: str, top_k: int = 3, response_style: str = 'comprehensive') -> Dict:
        """
        Ask a specific question and get an AI-generated answer based on your documents
        """
        if not self.gemini_model:
            return {'error': 'Gemini model not available. Please check your API key.'}
            
        # Get relevant documents
        retrieved_docs = self.search(question, top_k)
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': 'No relevant documents found for your question.',
                'confidence': 'low',
                'sources': []
            }
        
        # Create context
        context = "\n\n".join([
            f"Source {doc['rank']}: {doc['content']}"
            for doc in retrieved_docs
        ])
        
        # Customize prompt based on response style
        style_instructions = {
            'comprehensive': 'Provide a detailed, comprehensive answer with examples and explanations.',
            'concise': 'Provide a brief, direct answer focusing on the key points.',
            'analytical': 'Provide an analytical response that examines different aspects and implications.',
            'practical': 'Focus on practical applications and actionable insights.'
        }
        
        style_instruction = style_instructions.get(response_style, style_instructions['comprehensive'])
        
        prompt = f"""You are an AI assistant helping to answer questions based on a document collection.

Question: {question}

Relevant Sources:
{context}

Instructions:
- {style_instruction}
- Base your answer strictly on the provided sources
- If the sources don't contain sufficient information, clearly state this
- Cite specific sources when making claims
- Maintain accuracy and avoid speculation
- If sources contradict each other, acknowledge this

Answer:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            
            return {
                'question': question,
                'answer': response.text,
                'response_style': response_style,
                'sources': [
                    {
                        'rank': doc['rank'],
                        'document_id': doc['document_id'],
                        'score': doc['score']
                    }
                    for doc in retrieved_docs
                ],
                'source_count': len(retrieved_docs),
                'success': True
            }
            
        except Exception as e:
            return {
                'question': question,
                'error': str(e),
                'success': False
            }
        
    def save_agent(self, filepath: str):
        """
        Save the agent to disk
        """
        self.document_store.save(filepath)
        
    def load_agent(self, filepath: str):
        """
        Load the agent from disk
        """
        self.document_store.load(filepath)
        self.is_initialized = True
        
    def load_existing_documents(self):
        """
        Load existing documents from storage if available
        """
        if self.document_store.load():
            self.is_initialized = True
            print(f"Loaded existing document collection with {len(self.document_store.documents)} documents")
        
    def add_documents_from_files(self, file_paths: List[str]) -> Dict:
        """
        Add documents from selected file paths
        """
        results = {
            'added': 0,
            'skipped': 0,
            'errors': 0,
            'messages': []
        }
        
        documents = DocumentLoader.load_selected_files(file_paths, 
                                                     self.document_store.max_documents - len(self.document_store.documents))
        
        for doc in documents:
            try:
                if self.document_store.add_document(doc['id'], doc['content'], doc['metadata']):
                    results['added'] += 1
                    results['messages'].append(f"Added: {doc['id']}")
                else:
                    results['skipped'] += 1
                    results['messages'].append(f"Skipped (duplicate): {doc['id']}")
            except Exception as e:
                results['errors'] += 1
                results['messages'].append(f"Error with {doc['id']}: {str(e)}")
        
        # Save after adding documents
        if results['added'] > 0:
            self.save_documents()
            
        return results
    
    def save_documents(self):
        """
        Save documents to persistent storage
        """
        self.document_store.save()
        print(f"Saved {len(self.document_store.documents)} documents to storage")
        
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and save
        """
        if self.document_store.remove_document(doc_id):
            self.save_documents()
            return True
        return False
        
    def clear_all_documents(self):
        """
        Clear all documents and save
        """
        self.document_store.clear_all_documents()
        self.is_initialized = False
        self.save_documents()
        
    def get_stats(self) -> Dict:
        """
        Get statistics about the document collection
        """
        stats = self.document_store.get_document_stats()
        stats.update({
            'is_initialized': self.is_initialized,
            'embedding_dimension': self.document_store.embedding_dim if self.is_initialized else None
        })
        return stats
        
    def get_document_list(self) -> List[Dict]:
        """
        Get list of all stored documents with metadata
        """
        return [{
            'id': doc['id'],
            'content_preview': doc['content'][:100] + '...' if len(doc['content']) > 100 else doc['content'],
            'added_date': doc.get('added_date', 'unknown'),
            'file_type': doc['metadata'].get('file_type', 'unknown'),
            'file_size': doc['metadata'].get('file_size', 0)
        } for doc in self.document_store.documents]
    
    def search_with_highlights(self, query: str, top_k: int = 5, snippet_length: int = 300,
                             use_advanced_highlighting: bool = True,
                             include_contextual_info: bool = True,
                             exact_match_only: bool = False,
                             include_synonyms: bool = True,
                             include_related: bool = False) -> List[Dict]:
        """
        Enhanced search that returns only relevant highlighted portions of documents
        """
        if not self.is_initialized:
            raise ValueError("Agent not initialized. Call initialize() first")
            
        # Get basic search results
        results = self.document_store.search(query, top_k)
        
        # Enhance each result with highlighted snippets
        enhanced_results = []
        for result in results:
            # Extract relevant chunks from the document with enhanced features
            relevant_chunks = self.text_highlighter.extract_relevant_chunks(
                result['content'], query, chunk_size=200, top_chunks=2, use_semantic_chunking=True
            )
            
            # Create enhanced snippet based on options
            if exact_match_only:
                snippet = self.text_highlighter.create_snippet(
                    result['content'], query, max_length=snippet_length,
                    use_advanced_highlighting=False
                )
                snippet_info = {}
            elif include_contextual_info:
                contextual_snippet = self.text_highlighter.create_contextual_snippet(
                    result['content'], query, max_length=snippet_length,
                    include_synonyms=include_synonyms, include_related=include_related
                )
                snippet = contextual_snippet['snippet']
                snippet_info = {
                    'relevance_score': contextual_snippet['relevance_score'],
                    'keyword_count': contextual_snippet['keyword_count'],
                    'has_more_content': contextual_snippet['has_more_content']
                }
            else:
                snippet = self.text_highlighter.create_snippet(
                    result['content'], query, max_length=snippet_length,
                    use_advanced_highlighting=use_advanced_highlighting,
                    include_synonyms=include_synonyms, include_related=include_related
                )
                snippet_info = {}
            
            # Extract sentences around keywords
            relevant_sentences = self.text_highlighter.extract_sentences_around_keywords(
                result['content'], query, context_sentences=1
            )
            
            # Apply highlighting to sentences based on options
            if exact_match_only:
                highlighted_sentences = [
                    self.text_highlighter.highlight_keywords(sentence, query)
                    for sentence in relevant_sentences[:3]
                ]
            elif use_advanced_highlighting:
                highlighted_sentences = [
                    self.text_highlighter.highlight_keywords_advanced(
                        sentence, query, include_synonyms=include_synonyms, include_related=include_related
                    )
                    for sentence in relevant_sentences[:3]
                ]
            else:
                highlighted_sentences = [
                    self.text_highlighter.highlight_keywords(sentence, query)
                    for sentence in relevant_sentences[:3]
                ]
            
            enhanced_result = {
                'rank': result['rank'],
                'score': result['score'],
                'document_id': result['document_id'],
                'highlighted_snippet': snippet,
                'snippet_info': snippet_info,
                'relevant_chunks': relevant_chunks,
                'relevant_sentences': highlighted_sentences,
                'metadata': result['metadata'],
                'full_content': result['content']  # Keep full content for reference
            }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def generate_enhanced_response(self, query: str, top_k: int = 3, use_llm: bool = True) -> Dict:
        """
        Generate RAG response using highlighted snippets with optional LLM enhancement
        """
        # Get enhanced search results
        enhanced_results = self.search_with_highlights(query, top_k, snippet_length=200)
        
        # Create context from highlighted snippets
        context_parts = []
        for result in enhanced_results:
            context_parts.append(f"Document {result['rank']} ({result['document_id']}):")
            context_parts.append(result['highlighted_snippet'])
            context_parts.append("")  # Empty line for separation
        
        context = "\n".join(context_parts)
        
        response = {
            'query': query,
            'enhanced_results': enhanced_results,
            'context': context,
            'summary': f"Found {len(enhanced_results)} relevant document excerpts for your query: '{query}'"
        }
        
        # Generate LLM response using highlighted snippets
        if use_llm and self.gemini_model:
            try:
                llm_response = self._generate_enhanced_llm_response(query, enhanced_results)
                response['llm_response'] = llm_response
                response['has_llm_response'] = True
            except Exception as e:
                response['llm_error'] = str(e)
                response['has_llm_response'] = False
        else:
            response['has_llm_response'] = False
        
        return response
        
    def _generate_enhanced_llm_response(self, query: str, enhanced_results: List[Dict]) -> str:
        """
        Generate LLM response using enhanced search results with highlights
        """
        # Create focused context from highlighted snippets
        context_parts = []
        for result in enhanced_results:
            context_parts.append(f"Source {result['rank']} (Score: {result['score']:.3f}):")
            context_parts.append(f"Document: {result['document_id']}")
            context_parts.append(f"Relevant excerpt: {result['highlighted_snippet']}")
            
            # Add relevant sentences if available
            if result.get('relevant_sentences'):
                context_parts.append("Key sentences:")
                for sentence in result['relevant_sentences'][:2]:  # Limit to 2 sentences
                    context_parts.append(f"- {sentence}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following highlighted excerpts from relevant documents, provide a comprehensive answer to the user's question.

Query: {query}

Relevant Document Excerpts:
{context}

Instructions:
- Focus on the highlighted information in the excerpts
- Synthesize information from multiple sources when relevant
- Provide a clear, well-structured answer
- Reference specific sources when making claims
- If the excerpts don't fully answer the question, acknowledge what's missing
- Use the relevance scores to prioritize information from higher-scoring sources

Answer:"""

        response = self.gemini_model.generate_content(prompt)
        return response.text
        
    def chat_with_documents(self, message: str, conversation_history: List[Dict] = None, top_k: int = 3) -> Dict:
        """
        Have a conversational interaction with your documents using Gemini
        """
        if not self.gemini_model:
            return {'error': 'Gemini model not available. Please check your API key.'}
        
        # Get relevant documents for the current message
        retrieved_docs = self.search(message, top_k)
        
        # Create context from documents
        context = "\n\n".join([
            f"Document {doc['rank']}: {doc['content'][:500]}..."  # Limit context length
            for doc in retrieved_docs
        ])
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n".join([
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in conversation_history[-3:]  # Last 3 turns
            ])
        
        prompt = f"""You are an AI assistant having a conversation about documents in a knowledge base.

Previous conversation:
{conversation_context}

Current user message: {message}

Relevant documents:
{context}

Instructions:
- Respond conversationally and naturally
- Use information from the documents to inform your response
- Maintain context from the previous conversation
- If the user asks follow-up questions, reference previous parts of the conversation
- Be helpful and engaging while staying grounded in the document content

Response:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            
            return {
                'user_message': message,
                'assistant_response': response.text,
                'sources_used': len(retrieved_docs),
                'success': True
            }
            
        except Exception as e:
            return {
                'user_message': message,
                'error': str(e),
                'success': False
            }
            
    def get_model_status(self) -> Dict:
        """
        Get status of the Gemini model integration
        """
        return {
            'gemini_available': self.gemini_model is not None,
            'api_key_configured': os.getenv('GOOGLE_API_KEY') is not None,
            'model_name': 'gemini-2.0-flash-exp' if self.gemini_model else None
        }