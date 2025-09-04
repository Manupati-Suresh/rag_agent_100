import os
import json
import numpy as np
from typing import List, Dict, Tuple
import pickle
from datetime import datetime
import hashlib

class DocumentStore:
    def __init__(self, storage_path: str = 'document_storage'):
        """
        Initialize the document store.
        """
        self.documents = []
        self.index = None # No FAISS index for rule-based
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
        Placeholder for rule-based indexing. No FAISS index will be built.
        """
        print("Rule-based document store: No explicit search index to build.")
        self.index = True  # Indicate that indexing conceptually happened
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Rule-based keyword search: score documents by keyword matches and return top_k.
        """
        if not self.documents:
            return []

        # Extract simple keywords
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = {"the","a","an","and","or","but","is","are","was","were","to","in","of","for"}
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]

        scored: List[Tuple[float, Dict]] = []
        for doc in self.documents:
            text_lower = doc['content'].lower()
            score = 0.0
            matches = 0
            for kw in keywords:
                # word-boundary match count
                count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
                if count > 0:
                    matches += count
                    # weight by frequency with diminishing returns
                    score += min(1.0, 0.5 + 0.25 * count)

            # Lightweight bonus for title/id keyword presence
            if any(kw in doc['id'].lower() for kw in keywords):
                score += 0.5

            if score > 0:
                scored.append((score, doc))

        # If nothing matched, return empty
        if not scored:
            return []

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict] = []
        for rank, (score, doc) in enumerate(scored[:top_k], start=1):
            results.append({
                'rank': rank,
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
            'embedding_dim': None, # No embeddings
            'max_documents': self.max_documents,
            'saved_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        # No FAISS index to save
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
        self.index = None # No FAISS index
        self.max_documents = data.get('max_documents', 100)
        
        # No FAISS index to load
        print(f"Loaded {len(self.documents)} documents from {filepath}")
        return True
        
    def remove_document(self, doc_id: str):
        """
        Remove a document by ID
        """
        for i, doc in enumerate(self.documents):
            if doc['id'] == doc_id:
                self.documents.pop(i)
                # Rebuild index not needed for rule-based if not doing complex indexing
                self.index = None # Reset index state
                return True
        return False
        
    def clear_all_documents(self):
        """
        Clear all documents from the store
        """
        self.documents = []
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