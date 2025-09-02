#!/usr/bin/env python3
"""
RAG Agent with Telegram Document Storage
Enhanced version that uses Telegram as document storage backend
"""

import os
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import json

# Import existing components
from document_store import DocumentStore
from telegram_document_store import TelegramDocumentStore, TelegramRAGIntegration
from telegram_config import TelegramConfig
from text_highlighter import TextHighlighter

# Import AI components
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TelegramRAGAgent:
    """RAG Agent with Telegram document storage backend"""
    
    def __init__(self, use_telegram: bool = True):
        """
        Initialize RAG Agent with optional Telegram storage
        
        Args:
            use_telegram: Whether to use Telegram storage (default: True)
        """
        self.use_telegram = use_telegram
        self.document_store = DocumentStore()
        self.telegram_store = None
        self.telegram_integration = None
        self.highlighter = TextHighlighter()
        
        # Initialize Gemini AI
        self.gemini_model = None
        self._initialize_gemini()
        
        # Storage stats
        self.storage_stats = {}
    
    def _initialize_gemini(self):
        """Initialize Gemini AI model"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("⚠️ GEMINI_API_KEY not found in environment variables")
                return False
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini 2.0 Flash initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            return False
    
    async def initialize_telegram(self):
        """Initialize Telegram storage"""
        if not self.use_telegram:
            print("📁 Using local storage only")
            return True
        
        try:
            # Validate Telegram configuration
            TelegramConfig.validate_config()
            
            # Initialize Telegram store
            self.telegram_store = TelegramDocumentStore(
                api_id=TelegramConfig.API_ID,
                api_hash=TelegramConfig.API_HASH,
                phone_number=TelegramConfig.PHONE_NUMBER,
                channel_username=TelegramConfig.STORAGE_CHANNEL
            )
            
            await self.telegram_store.initialize()
            
            # Create integration layer
            self.telegram_integration = TelegramRAGIntegration(self.telegram_store)
            
            print("✅ Telegram storage initialized successfully!")
            return True
            
        except ValueError as e:
            print(f"❌ Telegram configuration error: {e}")
            print("\n" + TelegramConfig.get_setup_instructions())
            return False
        except Exception as e:
            print(f"❌ Failed to initialize Telegram storage: {e}")
            return False
    
    async def load_documents(self):
        """Load documents from storage (Telegram or local)"""
        if self.use_telegram and self.telegram_integration:
            # Load from Telegram
            documents = await self.telegram_integration.sync_from_telegram()
            
            # Add to local document store for processing
            for doc in documents:
                self.document_store.add_document(
                    doc_id=doc['id'],
                    content=doc['content'],
                    metadata=doc.get('metadata', {})
                )
            
            print(f"📚 Loaded {len(documents)} documents from Telegram")
        else:
            # Load from local storage
            self.document_store.load()
            print(f"📚 Loaded {len(self.document_store.documents)} documents from local storage")
    
    async def add_document(self, doc_id: str, content: str, metadata: Dict = None) -> bool:
        """
        Add a document to storage
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Optional metadata
            
        Returns:
            bool: Success status
        """
        try:
            if self.use_telegram and self.telegram_integration:
                # Add to Telegram storage
                success = await self.telegram_integration.add_document_to_telegram(
                    doc_id, content, metadata
                )
                if success:
                    # Also add to local store for immediate use
                    self.document_store.add_document(doc_id, content, metadata)
                return success
            else:
                # Add to local storage only
                return self.document_store.add_document(doc_id, content, metadata)
                
        except Exception as e:
            print(f"❌ Failed to add document '{doc_id}': {e}")
            return False
    
    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document from storage"""
        try:
            if self.use_telegram and self.telegram_store:
                # Remove from Telegram
                success = await self.telegram_store.remove_document(doc_id)
                if success:
                    # Also remove from local store
                    self.document_store.remove_document(doc_id)
                return success
            else:
                # Remove from local storage only
                return self.document_store.remove_document(doc_id)
                
        except Exception as e:
            print(f"❌ Failed to remove document '{doc_id}': {e}")
            return False
    
    def initialize_search_index(self):
        """Initialize the search index"""
        if len(self.document_store.documents) == 0:
            print("⚠️ No documents loaded. Load documents first.")
            return False
        
        try:
            self.document_store.build_index()
            print("✅ Search index built successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to build search index: {e}")
            return False
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documents using semantic similarity"""
        try:
            if self.document_store.index is None:
                print("⚠️ Search index not built. Building now...")
                if not self.initialize_search_index():
                    return []
            
            results = self.document_store.search(query, top_k)
            
            # Add highlighting
            for result in results:
                highlighted = self.highlighter.highlight_text(
                    result['content'], 
                    query,
                    max_length=500
                )
                result['highlighted_content'] = highlighted
            
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def chat_with_documents(self, user_message: str, max_context_docs: int = 3) -> Dict:
        """Chat with documents using Gemini AI"""
        try:
            if not self.gemini_model:
                return {
                    'success': False,
                    'error': 'Gemini AI not initialized',
                    'response': 'AI chat is not available. Please check your Gemini API key.'
                }
            
            # Ensure index is built
            if self.document_store.index is None:
                if not self.initialize_search_index():
                    return {
                        'success': False,
                        'error': 'Search index not available',
                        'response': 'Cannot search documents. Please ensure documents are loaded and indexed.'
                    }
            
            # Search for relevant documents
            relevant_docs = self.search_documents(user_message, top_k=max_context_docs)
            
            if not relevant_docs:
                return {
                    'success': True,
                    'response': "I don't have any relevant documents to answer your question. Please add some documents first.",
                    'sources': []
                }
            
            # Prepare context
            context = "\n\n".join([
                f"Document {i+1} (ID: {doc['document_id']}):\n{doc['content'][:1000]}..."
                for i, doc in enumerate(relevant_docs)
            ])
            
            # Create prompt
            prompt = f"""Based on the following documents, please answer the user's question. Be helpful and informative.

Documents:
{context}

User Question: {user_message}

Please provide a comprehensive answer based on the documents provided. If the documents don't contain enough information to fully answer the question, please say so."""
            
            # Generate response
            response = self.gemini_model.generate_content(prompt)
            
            return {
                'success': True,
                'response': response.text,
                'sources': [
                    {
                        'document_id': doc['document_id'],
                        'score': doc['score'],
                        'snippet': doc['content'][:200] + "..."
                    }
                    for doc in relevant_docs
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': f'Sorry, I encountered an error: {str(e)}'
            }
    
    def ask_question(self, question: str, response_style: str = "comprehensive") -> Dict:
        """Ask a specific question about the documents"""
        try:
            if not self.gemini_model:
                return {
                    'success': False,
                    'error': 'Gemini AI not initialized'
                }
            
            # Ensure index is built
            if self.document_store.index is None:
                if not self.initialize_search_index():
                    return {
                        'success': False,
                        'error': 'Search index not available'
                    }
            
            # Search for relevant documents
            relevant_docs = self.search_documents(question, top_k=5)
            
            if not relevant_docs:
                return {
                    'success': True,
                    'answer': "No relevant documents found for your question.",
                    'sources': []
                }
            
            # Prepare context
            context = "\n\n".join([
                f"Document {doc['document_id']}:\n{doc['content']}"
                for doc in relevant_docs
            ])
            
            # Style-specific prompts
            style_prompts = {
                "brief": "Provide a brief, concise answer.",
                "comprehensive": "Provide a detailed, comprehensive answer.",
                "bullet_points": "Provide the answer in bullet points.",
                "step_by_step": "Provide a step-by-step explanation if applicable."
            }
            
            style_instruction = style_prompts.get(response_style, style_prompts["comprehensive"])
            
            prompt = f"""Based on the following documents, answer this question: {question}

{style_instruction}

Documents:
{context}

Answer:"""
            
            response = self.gemini_model.generate_content(prompt)
            
            return {
                'success': True,
                'question': question,
                'answer': response.text,
                'style': response_style,
                'sources': [
                    {
                        'document_id': doc['document_id'],
                        'relevance_score': doc['score']
                    }
                    for doc in relevant_docs
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_summary(self, focus_topic: str = None) -> Dict:
        """Generate a summary of documents"""
        try:
            if not self.gemini_model:
                return {
                    'success': False,
                    'error': 'Gemini AI not initialized'
                }
            
            if not self.document_store.documents:
                return {
                    'success': False,
                    'error': 'No documents available'
                }
            
            # Prepare content
            if focus_topic:
                # Search for topic-specific documents
                relevant_docs = self.search_documents(focus_topic, top_k=10)
                content = "\n\n".join([doc['content'] for doc in relevant_docs])
                prompt = f"Summarize the following documents focusing on '{focus_topic}':\n\n{content}"
            else:
                # Summarize all documents
                content = "\n\n".join([doc['content'] for doc in self.document_store.documents])
                prompt = f"Provide a comprehensive summary of the following documents:\n\n{content}"
            
            response = self.gemini_model.generate_content(prompt)
            
            return {
                'success': True,
                'summary': response.text,
                'focus_topic': focus_topic,
                'documents_count': len(self.document_store.documents)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_storage_stats(self) -> Dict:
        """Get comprehensive storage statistics"""
        local_stats = self.document_store.get_document_stats()
        
        if self.use_telegram and self.telegram_store:
            telegram_stats = await self.telegram_store.get_storage_stats()
            return {
                'storage_backend': 'Telegram + Local Cache',
                'telegram': telegram_stats,
                'local_cache': local_stats
            }
        else:
            return {
                'storage_backend': 'Local Only',
                'local': local_stats
            }
    
    async def close(self):
        """Close connections and cleanup"""
        if self.telegram_store:
            await self.telegram_store.close()
        print("🔌 RAG Agent connections closed")


# Async wrapper functions for easier use
async def create_telegram_rag_agent(use_telegram: bool = True) -> TelegramRAGAgent:
    """Create and initialize a Telegram RAG Agent"""
    agent = TelegramRAGAgent(use_telegram=use_telegram)
    
    if use_telegram:
        success = await agent.initialize_telegram()
        if not success:
            print("⚠️ Falling back to local storage")
            agent.use_telegram = False
    
    await agent.load_documents()
    
    if len(agent.document_store.documents) > 0:
        agent.initialize_search_index()
    
    return agent