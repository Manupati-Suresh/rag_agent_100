import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import pickle

class DocumentStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the document store with a sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """
        Add a document to the store
        """
        document = {
            'id': doc_id,
            'content': content,
            'metadata': metadata or {}
        }
        self.documents.append(document)
        
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
        
    def save(self, filepath: str):
        """
        Save the document store to disk
        """
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        # Save FAISS index separately
        if self.index is not None:
            faiss.write_index(self.index, filepath.replace('.pkl', '.faiss'))
            
    def load(self, filepath: str):
        """
        Load the document store from disk
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.embedding_dim = data['embedding_dim']
        
        # Load FAISS index
        faiss_path = filepath.replace('.pkl', '.faiss')
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)