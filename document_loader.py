import os
import json
from typing import List, Dict, Optional
import docx
import PyPDF2
from io import StringIO
import mimetypes
from pathlib import Path

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
    def get_supported_files(directory: str) -> List[Dict]:
        """
        Get list of supported files in directory with metadata
        """
        supported_extensions = {'.txt', '.pdf', '.docx', '.json', '.md'}
        files_info = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in supported_extensions:
                    try:
                        file_size = os.path.getsize(filepath)
                        files_info.append({
                            'filename': file,
                            'filepath': filepath,
                            'extension': file_ext,
                            'size': file_size,
                            'size_mb': round(file_size / (1024 * 1024), 2),
                            'relative_path': os.path.relpath(filepath, directory)
                        })
                    except Exception as e:
                        print(f"Error accessing {filepath}: {str(e)}")
        
        return sorted(files_info, key=lambda x: x['filename'])
    
    @staticmethod
    def load_selected_files(file_paths: List[str], max_files: int = 100) -> List[Dict]:
        """
        Load specific files by their paths
        """
        documents = []
        
        for filepath in file_paths[:max_files]:
            try:
                file_ext = os.path.splitext(filepath)[1].lower()
                filename = os.path.basename(filepath)
                
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
                
                # Skip empty content
                if not content.strip():
                    print(f"Skipping empty file: {filename}")
                    continue
                    
                documents.append({
                    'id': filename,
                    'content': content,
                    'metadata': {
                        'filepath': filepath,
                        'file_type': file_ext,
                        'file_size': os.path.getsize(filepath),
                        'filename': filename
                    }
                })
                
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                
        return documents
    
    @staticmethod
    def load_from_directory(directory: str, max_files: int = 100) -> List[Dict]:
        """
        Load documents from a directory
        """
        supported_files = DocumentLoader.get_supported_files(directory)
        file_paths = [f['filepath'] for f in supported_files[:max_files]]
        return DocumentLoader.load_selected_files(file_paths, max_files)
    
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