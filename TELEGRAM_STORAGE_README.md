# 📱 Telegram Document Storage for RAG Agent

Transform your Telegram account into a powerful document storage backend for your RAG (Retrieval-Augmented Generation) system! This integration allows you to store up to 100 documents in Telegram and use them with AI-powered search, chat, and analysis features.

## 🌟 Why Telegram Storage?

- **🆓 Free**: No cloud storage costs
- **🔒 Secure**: End-to-end encryption available
- **📱 Accessible**: Access documents from any device
- **🌐 Reliable**: Telegram's robust infrastructure
- **💾 Persistent**: Documents stored permanently
- **🔄 Sync**: Automatic synchronization across devices

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Telegram API Credentials

1. Go to [https://my.telegram.org/apps](https://my.telegram.org/apps)
2. Log in with your phone number
3. Create a new application
4. Copy your **API ID** and **API Hash**

### 3. Run Setup Script

```bash
python setup_telegram_storage.py
```

This interactive script will:
- Guide you through configuration
- Test your Telegram connection
- Verify document storage functionality

### 4. Start Using!

**Streamlit Web Interface:**
```bash
streamlit run streamlit_telegram_app.py
```

**Python Example:**
```bash
python telegram_example.py
```

## 📋 Configuration

Add these to your `.env` file:

```env
# Telegram API Configuration
TELEGRAM_API_ID=your_api_id_here
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_PHONE_NUMBER=+1234567890

# Optional: Private channel for storage
TELEGRAM_STORAGE_CHANNEL=@your_private_channel

# AI Configuration (for chat features)
GEMINI_API_KEY=your_gemini_api_key
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   RAG Agent      │    │   Telegram      │
│   Web App       │◄──►│   (Local Cache)  │◄──►│   Storage       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Gemini AI      │
                       │   (Chat & Q&A)   │
                       └──────────────────┘
```

## 📚 Storage Options

### Option 1: Saved Messages (Default)
- Documents stored in your "Saved Messages"
- Private and secure
- No additional setup required

### Option 2: Private Channel (Recommended)
- Create a private Telegram channel
- Better organization
- Easier to manage large document collections

## 🔧 Features

### 📄 Document Management
- **Add Documents**: Store up to 100 documents with metadata
- **Search Documents**: Semantic search using AI embeddings
- **Remove Documents**: Delete documents from Telegram storage
- **Sync**: Automatic synchronization between Telegram and local cache

### 🤖 AI-Powered Features
- **Chat Interface**: Natural conversation with your documents
- **Q&A System**: Ask specific questions with different response styles
- **Document Summaries**: AI-generated summaries of your content
- **Semantic Search**: Find relevant documents using meaning, not just keywords

### 🎨 Web Interface
- **Modern UI**: Clean, responsive Streamlit interface
- **Real-time Chat**: Interactive chat with conversation history
- **Document Upload**: Easy document addition through web interface
- **Search & Filter**: Advanced search with highlighting
- **Statistics**: Storage usage and performance metrics

## 📖 Usage Examples

### Basic Document Operations

```python
import asyncio
from rag_agent_telegram import create_telegram_rag_agent

async def main():
    # Initialize agent with Telegram storage
    agent = await create_telegram_rag_agent(use_telegram=True)
    
    # Add a document
    await agent.add_document(
        doc_id="python_guide",
        content="Python is a versatile programming language...",
        metadata={"category": "programming", "language": "python"}
    )
    
    # Search documents
    results = agent.search_documents("Python programming", top_k=5)
    
    # Chat with documents
    response = agent.chat_with_documents("What is Python used for?")
    print(response['response'])
    
    # Close connections
    await agent.close()

asyncio.run(main())
```

### Advanced Q&A

```python
# Ask questions with different response styles
styles = ["brief", "comprehensive", "bullet_points", "step_by_step"]

for style in styles:
    result = agent.ask_question(
        "How do I learn Python?", 
        response_style=style
    )
    print(f"{style}: {result['answer']}")
```

## 🔒 Security & Privacy

### Data Protection
- Documents stored in your personal Telegram account
- Optional end-to-end encryption with private channels
- No third-party cloud storage required
- Full control over your data

### API Security
- Telegram API uses OAuth 2.0 authentication
- Session tokens stored locally
- No sensitive data transmitted to external services

## 🛠️ Troubleshooting

### Common Issues

**1. "Telegram not configured" Error**
```bash
# Run the setup script
python setup_telegram_storage.py
```

**2. Phone Verification Required**
- Telegram will send a verification code to your phone
- Enter the code when prompted during initialization

**3. "Index not built" Error**
- The system automatically builds search indexes
- If issues persist, restart the application

**4. Connection Timeout**
- Check your internet connection
- Verify Telegram API credentials
- Try using a different network

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance

### Storage Limits
- **Maximum Documents**: 100 documents
- **Document Size**: Up to 50MB per document (Telegram limit)
- **Total Storage**: Effectively unlimited (Telegram provides free storage)

### Speed Optimization
- **Local Caching**: Frequently accessed documents cached locally
- **Batch Operations**: Multiple documents processed efficiently
- **Async Operations**: Non-blocking I/O for better performance

## 🔄 Migration

### From Local Storage
```python
# Export from local storage
local_agent = RAGAgent()
local_agent.load()

# Import to Telegram storage
telegram_agent = await create_telegram_rag_agent(use_telegram=True)

for doc in local_agent.document_store.documents:
    await telegram_agent.add_document(
        doc['id'], 
        doc['content'], 
        doc.get('metadata', {})
    )
```

### Backup & Restore
```python
# Backup all documents
documents = await telegram_agent.telegram_integration.sync_from_telegram()

# Save backup
import json
with open('backup.json', 'w') as f:
    json.dump(documents, f, indent=2)

# Restore from backup
with open('backup.json', 'r') as f:
    documents = json.load(f)

for doc in documents:
    await telegram_agent.add_document(
        doc['id'], 
        doc['content'], 
        doc.get('metadata', {})
    )
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **Multi-language Support**: Document processing in different languages
- **Advanced Search**: More sophisticated search algorithms
- **UI Enhancements**: Better web interface features
- **Performance**: Optimization for large document collections
- **Integrations**: Support for more AI models and storage backends

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check this README and inline code comments
- **Examples**: See `telegram_example.py` for usage examples
- **Setup Help**: Run `python setup_telegram_storage.py` for guided setup

---

**🎉 Enjoy your AI-powered document storage with Telegram!**

Transform your documents into an intelligent, searchable knowledge base that's always accessible from anywhere in the world.