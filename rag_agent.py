from document_store import DocumentStore
from document_loader import DocumentLoader
from typing import List, Dict
import json

class RAGAgent:
    """
    RAG Agent that combines document retrieval with generation capabilities
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.document_store = DocumentStore(model_name)
        self.is_initialized = False
        
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
        
    def get_stats(self) -> Dict:
        """
        Get statistics about the document collection
        """
        return {
            'total_documents': len(self.document_store.documents),
            'is_initialized': self.is_initialized,
            'embedding_dimension': self.document_store.embedding_dim if self.is_initialized else None
        }