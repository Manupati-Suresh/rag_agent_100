from document_store import DocumentStore
from document_loader import DocumentLoader
from typing import List, Dict
import json

class RAGAgent:
    """
    RAG Agent that combines document retrieval with generation capabilities
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', storage_path: str = 'document_storage'):
        self.document_store = DocumentStore(model_name, storage_path)
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