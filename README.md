# RAG Document Search Agent

A powerful Retrieval-Augmented Generation (RAG) agent that can store up to 100 documents and perform semantic search using sentence transformers and FAISS indexing.

## Features

- 🔍 **Semantic Search**: Uses sentence transformers for meaningful document retrieval
- 📚 **Document Storage**: Efficiently stores and indexes up to 100 documents
- ⚡ **Fast Similarity Search**: FAISS-powered vector search for quick results
- 📄 **Multiple Formats**: Supports TXT, PDF, DOCX, JSON, and Markdown files
- 🎯 **RAG Capabilities**: Combines retrieval with response generation
- 💾 **Persistence**: Save and load your document collections
- 🌐 **Web Interface**: Streamlit-based UI for easy interaction

## Installation

1. Clone or download the project files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```python
from rag_agent import RAGAgent

# Initialize the agent
agent = RAGAgent()

# Load documents (sample documents for demo)
agent.load_documents()

# Build the search index
agent.initialize()

# Search for relevant documents
results = agent.search("machine learning algorithms", top_k=5)

# Generate RAG response
response = agent.generate_response("What is machine learning?")
```

### Web Interface

Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

### Example Script

Run the example to see the agent in action:

```bash
python example_usage.py
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

agent = RAGAgent()

# Option 1: Load from directory
agent.load_documents(directory="path/to/your/documents", max_docs=100)

# Option 2: Load custom documents
documents = [
    {
        'id': 'doc1',
        'content': 'Your document content here...',
        'metadata': {'category': 'research', 'date': '2024-01-01'}
    }
]
agent.load_documents(documents=documents)

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

1. **Installation Issues**: Ensure you have Python 3.8+ and pip installed
2. **Memory Errors**: Reduce batch size or use a smaller embedding model
3. **File Loading Errors**: Check file permissions and formats
4. **Search Quality**: Try different embedding models or adjust similarity thresholds

## License

This project is open source and available under the MIT License.