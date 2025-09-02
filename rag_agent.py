from document_store import DocumentStore
from document_loader import DocumentLoader
from text_highlighter import TextHighlighter
from typing import List, Dict
import json

class RAGAgent:
    """
    RAG Agent that combines document retrieval with generation capabilities
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', storage_path: str = 'document_storage'):
        self.document_store = DocumentStore(model_name, storage_path)
        self.text_highlighter = TextHighlighter(model_name)
        self.is_initialized = False
        self.storage_path = storage_path
        
        # Try to load existing documents
        self.load_existing_documents()
        
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
        
    def generate_response(self, query: str, top_k: int = 3) -> Dict:
        """
        Generate a response using retrieved documents
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
                             include_contextual_info: bool = True) -> List[Dict]:
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
            
            # Create enhanced snippet
            if include_contextual_info:
                contextual_snippet = self.text_highlighter.create_contextual_snippet(
                    result['content'], query, max_length=snippet_length
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
                    use_advanced_highlighting=use_advanced_highlighting
                )
                snippet_info = {}
            
            # Extract sentences around keywords
            relevant_sentences = self.text_highlighter.extract_sentences_around_keywords(
                result['content'], query, context_sentences=1
            )
            
            # Apply advanced highlighting to sentences
            if use_advanced_highlighting:
                highlighted_sentences = [
                    self.text_highlighter.highlight_keywords_advanced(sentence, query)
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
    
    def generate_enhanced_response(self, query: str, top_k: int = 3) -> Dict:
        """
        Generate RAG response using highlighted snippets instead of full documents
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
        
        return response