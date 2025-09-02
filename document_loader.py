import os
import json
from typing import List, Dict
import docx
import PyPDF2
from io import StringIO

class DocumentLoader:
    """
    Utility class to load documents from various formats
    """
    
    @staticmethod
    def load_text_file(filepath: str) -> str:
        """Load content from a text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_pdf_file(filepath: str) -> str:
        """Load content from a PDF file"""
        content = ""
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        return content
    
    @staticmethod
    def load_docx_file(filepath: str) -> str:
        """Load content from a Word document"""
        doc = docx.Document(filepath)
        content = ""
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"
        return content
    
    @staticmethod
    def load_json_file(filepath: str) -> str:
        """Load content from a JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    
    @staticmethod
    def load_from_directory(directory: str, max_files: int = 100) -> List[Dict]:
        """
        Load documents from a directory
        """
        documents = []
        supported_extensions = {'.txt', '.pdf', '.docx', '.json', '.md'}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if len(documents) >= max_files:
                    break
                    
                filepath = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in supported_extensions:
                    try:
                        if file_ext == '.txt' or file_ext == '.md':
                            content = DocumentLoader.load_text_file(filepath)
                        elif file_ext == '.pdf':
                            content = DocumentLoader.load_pdf_file(filepath)
                        elif file_ext == '.docx':
                            content = DocumentLoader.load_docx_file(filepath)
                        elif file_ext == '.json':
                            content = DocumentLoader.load_json_file(filepath)
                        else:
                            continue
                            
                        documents.append({
                            'id': file,
                            'content': content,
                            'metadata': {
                                'filepath': filepath,
                                'file_type': file_ext,
                                'file_size': os.path.getsize(filepath)
                            }
                        })
                        
                    except Exception as e:
                        print(f"Error loading {filepath}: {str(e)}")
                        
            if len(documents) >= max_files:
                break
                
        return documents
    
    @staticmethod
    def create_sample_documents() -> List[Dict]:
        """
        Create sample documents for testing
        """
        sample_docs = [
            {
                'id': 'doc_1',
                'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.',
                'metadata': {'category': 'AI', 'topic': 'machine_learning'}
            },
            {
                'id': 'doc_2', 
                'content': 'Python is a high-level programming language known for its simplicity and readability.',
                'metadata': {'category': 'Programming', 'topic': 'python'}
            },
            {
                'id': 'doc_3',
                'content': 'Natural language processing enables computers to understand and process human language.',
                'metadata': {'category': 'AI', 'topic': 'nlp'}
            },
            {
                'id': 'doc_4',
                'content': 'Deep learning uses neural networks with multiple layers to model complex patterns in data.',
                'metadata': {'category': 'AI', 'topic': 'deep_learning'}
            },
            {
                'id': 'doc_5',
                'content': 'Data science combines statistics, programming, and domain expertise to extract insights from data.',
                'metadata': {'category': 'Data Science', 'topic': 'analytics'}
            }
        ]
        
        return sample_docs