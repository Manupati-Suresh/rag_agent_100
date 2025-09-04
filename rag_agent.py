from document_store import DocumentStore
from document_loader import DocumentLoader
from text_highlighter import TextHighlighter
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv

class RAGAgent:
    """
    RAG Agent that combines document retrieval with rule-based generation capabilities
    """
    
    def __init__(self, storage_path: str = 'document_storage'):
        self.document_store = DocumentStore(storage_path)
        self.text_highlighter = TextHighlighter() # Still using highlighter for rule-based snippets
        self.is_initialized = False
        self.storage_path = storage_path
        self.faq_rules: List[Dict] = []
        
        # Load existing documents
        self.load_existing_documents()
        # Load FAQ rules if present
        self._load_faq_rules()

    def _load_faq_rules(self, faq_path: str = 'faq_rules.json') -> None:
        try:
            if os.path.exists(faq_path):
                with open(faq_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.faq_rules = data
                        print(f"Loaded {len(self.faq_rules)} FAQ rules")
        except Exception as e:
            print(f"Failed to load faq rules: {e}")

    def _match_faq(self, query: str) -> Optional[Dict]:
        """
        Match query against FAQ rules. Each rule: {patterns: [..], answer: str}
        Returns matched rule dict or None.
        """
        if not self.faq_rules:
            return None
        import re
        for rule in self.faq_rules:
            patterns = rule.get('patterns', [])
            for pat in patterns:
                try:
                    if re.search(pat, query, flags=re.IGNORECASE):
                        return rule
                except re.error:
                    # fallback to simple substring match if regex invalid
                    if pat.lower() in query.lower():
                        return rule
        return None
        
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
        Build the search index (rule-based: no explicit index, just documents)
        """
        if not self.document_store.documents:
            raise ValueError("No documents loaded. Call load_documents() first")
            
        self.document_store.build_index()
        self.is_initialized = True
        print("RAG Agent initialized successfully for rule-based operations!")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform rule-based search on the document collection (e.g., keyword matching)
        """
        if not self.is_initialized:
            raise ValueError("Agent not initialized. Call initialize() first")
            
        # Placeholder for actual rule-based search implementation
        # For now, it will use the simplified DocumentStore search (returns all docs)
        results = self.document_store.search(query, top_k)
        return results
        
    def generate_response(self, query: str, top_k: int = 3) -> Dict:
        """
        Generate a response using retrieved documents without LLM.
        This will be rule-based, returning relevant snippets or predefined FAQ answers.
        """
        # First try FAQ match
        faq = self._match_faq(query)
        if faq:
            return {
                'query': query,
                'retrieved_documents': [],
                'context': '',
                'answer': faq.get('answer', ''),
                'faq_matched': True,
                'has_llm_response': False
            }

        retrieved_docs = self.search(query, top_k)
        
        response = {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'context': "", # Context will be built dynamically or by rule
            'answer': 'No specific answer found. Here are relevant documents.',
            'has_llm_response': False # No LLM used
        }
        
        if retrieved_docs:
            # Use highlighted snippet from top doc
            snippet = self.text_highlighter.create_snippet(retrieved_docs[0]['content'], query, max_length=300)
            response['answer'] = f"From {retrieved_docs[0]['document_id']}: {snippet}"
            response['context'] = "\n\n".join([
                f"Document {doc['rank']}: {doc['content']}"
                for doc in retrieved_docs
            ])
            
        return response
        
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
            'embedding_dimension': None # No embeddings in rule-based
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
        Enhanced search that returns only relevant highlighted portions of documents (rule-based)
        """
        if not self.is_initialized:
            raise ValueError("Agent not initialized. Call initialize() first")
            
        # Get basic search results (currently all documents from document_store)
        results = self.document_store.search(query, top_k)
        
        # For rule-based, we'll implement simple keyword highlighting
        enhanced_results = []
        for result in results:
            highlighted_content = self.text_highlighter.highlight_keywords(result['content'], query)
            snippet = highlighted_content[:snippet_length] + "..." if len(highlighted_content) > snippet_length else highlighted_content
            
            enhanced_result = {
                'rank': result['rank'],
                'score': result['score'],
                'document_id': result['document_id'],
                'highlighted_snippet': snippet,
                'snippet_info': {}, # No advanced snippet info for rule-based
                'relevant_chunks': [], # Not using semantic chunks
                'relevant_sentences': [snippet], # Simplified for rule-based
                'metadata': result['metadata'],
                'full_content': result['content']
            }
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def generate_enhanced_response(self, query: str, top_k: int = 3) -> Dict:
        """
        Generate RAG response using highlighted snippets (rule-based).
        """
        # Try FAQ first
        faq = self._match_faq(query)
        if faq:
            return {
                'query': query,
                'enhanced_results': [],
                'context': '',
                'answer': faq.get('answer', ''),
                'faq_matched': True,
                'has_llm_response': False
            }

        enhanced_results = self.search_with_highlights(query, top_k, snippet_length=200)
        
        context_parts = []
        for result in enhanced_results:
            context_parts.append(f"Document {result['rank']} ({result['document_id']}):")
            context_parts.append(result['highlighted_snippet'])
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        response = {
            'query': query,
            'enhanced_results': enhanced_results,
            'context': context,
            'answer': f"Found {len(enhanced_results)} relevant document excerpts for your query: '{query}'",
            'has_llm_response': False
        }
        
        return response
        
    def chat_with_documents(self, message: str, conversation_history: List[Dict] = None, top_k: int = 3) -> Dict:
        """
        Have a conversational interaction with your documents using rule-based responses.
        """
        response = self.generate_enhanced_response(message, top_k)
        
        return {
            'user_message': message,
            'assistant_response': response.get('answer', response['context']),
            'sources_used': len(response['enhanced_results']),
            'success': True
        }