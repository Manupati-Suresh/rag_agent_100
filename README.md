# RAG Document Search Agent

A powerful Retrieval-Augmented Generation (RAG) agent that can store up to 100 documents and perform semantic search using sentence transformers and FAISS indexing.

## Features

- 🔍 **Semantic Search**: Uses sentence transformers for meaningful document retrieval
- 📚 **Permanent Storage**: Automatically saves up to 100 documents with persistent storage
- 📁 **File Browser**: Browse and select documents from your computer
- ⚡ **Fast Similarity Search**: FAISS-powered vector search for quick results
- 📄 **Multiple Formats**: Supports TXT, PDF, DOCX, JSON, and Markdown files
- 🎯 **RAG Capabilities**: Combines retrieval with response generation
- 🔄 **Duplicate Detection**: Prevents storing identical content
- 🌐 **Web Interface**: Enhanced Streamlit UI with file management
- 💻 **CLI Tools**: Command-line interface for document management

## Installation

### Option 1: Automatic Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
python setup.py
```

### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/Manupati-Suresh/rag_agent_100.git
cd rag_agent_100
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8 or higher
- 2GB+ RAM (for embedding models)
- Internet connection (for downloading models)

## Quick Start

### Command Line Usage

```python
from rag_agent import RAGAgent

# Initialize the agent with persistent storage
agent = RAGAgent(storage_path='my_documents')

# Add documents from files
file_paths = ['/path/to/document1.txt', '/path/to/document2.pdf']
results = agent.add_documents_from_files(file_paths)

# Initialize search (if not already done)
if not agent.is_initialized:
    agent.initialize()

# Search for relevant documents
results = agent.search("machine learning algorithms", top_k=5)

# Generate RAG response
response = agent.generate_response("What is machine learning?")
```

### CLI Document Manager

```bash
# List all stored documents
python document_manager.py list

# Add documents from files
python document_manager.py add file1.txt file2.pdf file3.docx

# Search documents
python document_manager.py search "machine learning"

# Browse directory for files
python document_manager.py browse /path/to/documents

# Remove a document
python document_manager.py remove document.txt

# Clear all documents
python document_manager.py clear

# Show statistics
python document_manager.py stats
```

### Web Interface

Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

### Example Scripts

Run the examples to see the agent in action:

```bash
# Basic example with sample documents
python example_usage.py

# Persistent storage demonstration
python persistent_example.py
```

## Core Components

### 1. DocumentStore (`document_store.py`)
- Handles document storage and embedding generation
- Uses sentence-transformers for creating embeddings
- FAISS indexing for efficient similarity search
- Supports saving/loading collections

### 2. DocumentLoader (`document_loader.py`)
- Loads documents from various file formats
- Batch processing from directories
- Sample document generation for testing

### 3. RAGAgent (`rag_agent.py`)
- Main agent class combining retrieval and generation
- Semantic search functionality
- Response generation using retrieved context

### 4. Streamlit App (`streamlit_app.py`)
- User-friendly web interface
- Real-time search and results display
- Document upload capabilities

## Usage Examples

### Loading Your Own Documents

```python
from rag_agent import RAGAgent
from document_loader import DocumentLoader

# Initialize with persistent storage
agent = RAGAgent(storage_path='my_documents')

# Option 1: Add documents from file paths
file_paths = ['/path/to/doc1.txt', '/path/to/doc2.pdf']
results = agent.add_documents_from_files(file_paths)

# Option 2: Browse directory and select files
available_files = DocumentLoader.get_supported_files('/path/to/documents')
selected_paths = [f['filepath'] for f in available_files[:10]]  # Select first 10
results = agent.add_documents_from_files(selected_paths)

# Option 3: Load from directory (all supported files)
agent.load_documents(directory="path/to/your/documents", max_docs=100)

# Initialize search index
if not agent.is_initialized:
    agent.initialize()
```

### Performing Searches

```python
# Simple search
results = agent.search("artificial intelligence", top_k=5)

# Access results
for result in results:
    print(f"Document: {result['document_id']}")
    print(f"Similarity: {result['score']:.3f}")
    print(f"Content: {result['content']}")
```

### RAG Response Generation

```python
# Generate contextual response
response = agent.generate_response("Explain machine learning", top_k=3)

print(f"Query: {response['query']}")
print(f"Summary: {response['summary']}")
print(f"Context: {response['context']}")
```

## Configuration

### Model Selection
You can use different sentence transformer models:

```python
# Faster, smaller model
agent = RAGAgent(model_name='all-MiniLM-L6-v2')

# More accurate, larger model
agent = RAGAgent(model_name='all-mpnet-base-v2')
```

### Search Parameters
- `top_k`: Number of results to return (default: 5)
- Similarity threshold can be adjusted in the code

## File Structure

```
├── requirements.txt          # Python dependencies
├── document_store.py        # Core document storage and indexing
├── document_loader.py       # Document loading utilities
├── rag_agent.py            # Main RAG agent class
├── streamlit_app.py        # Web interface
├── example_usage.py        # Usage examples
└── README.md              # This file
```

## Performance Notes

- **Embedding Model**: Uses `all-MiniLM-L6-v2` by default (384 dimensions)
- **Index Type**: FAISS IndexFlatIP for cosine similarity
- **Memory Usage**: ~1-2MB per 1000 documents (depending on content length)
- **Search Speed**: Sub-second search on 100 documents

## Extending the Agent

### Adding New File Types
Extend `DocumentLoader` to support additional formats:

```python
@staticmethod
def load_custom_format(filepath: str) -> str:
    # Your custom loading logic
    return content
```

### Custom Embedding Models
Use any sentence-transformers compatible model:

```python
agent = RAGAgent(model_name='your-custom-model')
```

### Integration with LLMs
The RAG response can be enhanced by integrating with OpenAI, Anthropic, or other LLM APIs for better response generation.

## Troubleshooting

### Common Issues

1. **Installation Issues**: 
   - Ensure Python 3.8+ is installed
   - Try: `python -m pip install --upgrade pip`
   - Use virtual environment: `python -m venv venv && source venv/bin/activate`

2. **Memory Errors**: 
   - Reduce batch size in document processing
   - Use smaller embedding model: `RAGAgent(model_name='all-MiniLM-L6-v2')`

3. **TensorFlow/Keras Errors**:
   - Install tf-keras: `pip install tf-keras`
   - Set environment variable: `TF_ENABLE_ONEDNN_OPTS=0`

4. **File Loading Errors**: 
   - Check file permissions and formats
   - Ensure files are not corrupted
   - Try with sample documents first

5. **Search Quality**: 
   - Try different embedding models
   - Adjust similarity thresholds
   - Use more specific queries

6. **Streamlit Issues**:
   - Install streamlit: `pip install streamlit`
   - Run: `streamlit run streamlit_app.py`
   - Check port availability (default: 8501)

## License

This project is open source and available under the MIT License.