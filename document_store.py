import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import pickle
from datetime import datetime
import hashlib

class DocumentStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', storage_path: str = 'document_storage'):
        """
        Initialize the document store with a sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.storage_path = storage_path
        self.max_documents = 100
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """
        Add a document to the store (max 100 documents)
        """
        if len(self.documents) >= self.max_documents:
            raise ValueError(f"Maximum document limit ({self.max_documents}) reached")
            
        # Generate unique hash for content
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check for duplicates
        for doc in self.documents:
            if doc.get('content_hash') == content_hash:
                print(f"Document with similar content already exists: {doc['id']}")
                return False
        
        document = {
            'id': doc_id,
            'content': content,
            'content_hash': content_hash,
            'metadata': metadata or {},
            'added_date': datetime.now().isoformat()
        }
        self.documents.append(document)
        return True
        
    def build_index(self):
        """
        Build FAISS index from all documents
        """
        if not self.documents:
            raise ValueError("No documents to index")
            
        # Extract content for embedding
        contents = [doc['content'] for doc in self.documents]
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.model.encode(contents, show_progress_bar=True)
        
        # Build FAISS index
        self.embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index built with {len(self.documents)} documents")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first")
            
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                doc = self.documents[idx]
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'document_id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata']
                })
                
        return results
        
    def save(self, filepath: str = None):
        """
        Save the document store to disk
        """
        if filepath is None:
            filepath = os.path.join(self.storage_path, 'document_store.pkl')
            
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim,
            'max_documents': self.max_documents,
            'saved_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        # Save FAISS index separately
        if self.index is not None:
            faiss_path = filepath.replace('.pkl', '.faiss')
            faiss.write_index(self.index, faiss_path)
            
        # Save document metadata as JSON for easy inspection
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        metadata = {
            'total_documents': len(self.documents),
            'document_list': [{'id': doc['id'], 'added_date': doc.get('added_date', 'unknown')} for doc in self.documents],
            'saved_date': datetime.now().isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def load(self, filepath: str = None):
        """
        Load the document store from disk
        """
        if filepath is None:
            filepath = os.path.join(self.storage_path, 'document_store.pkl')
            
        if not os.path.exists(filepath):
            print(f"No saved document store found at {filepath}")
            return False
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.embedding_dim = data['embedding_dim']
        self.max_documents = data.get('max_documents', 100)
        
        # Load FAISS index
        faiss_path = filepath.replace('.pkl', '.faiss')
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
            
        print(f"Loaded {len(self.documents)} documents from {filepath}")
        return True
        
    def remove_document(self, doc_id: str):
        """
        Remove a document by ID
        """
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                self.documents.pop(i)
                # Need to rebuild index after removal
                if self.embeddings is not None:
                    self.build_index()
                return True
        return False
        
    def clear_all_documents(self):
        """
        Clear all documents from the store
        """
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def get_document_stats(self):
        """
        Get statistics about stored documents
        """
        return {
            'total_documents': len(self.documents),
            'max_documents': self.max_documents,
            'remaining_slots': self.max_documents - len(self.documents),
            'storage_path': self.storage_path,
            'has_index': self.index is not None
        }