# rag_agent_telegram.py
import os
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import json
from document_store import DocumentStore
from telegram_document_store import TelegramDocumentStore, TelegramRAGIntegration
from telegram_config import TelegramConfig
from text_highlighter import TextHighlighter
from rag_agent import RAGAgent # Import the simplified RAGAgent
from telethon import events
from telethon.tl.types import DocumentAttributeFilename
import tempfile

class RuleBasedTelegramRAGAgent:
    def __init__(self):
        self.config = TelegramConfig()
        self.config.validate_config()
        
        self.telegram_store = TelegramDocumentStore(
            api_id=int(self.config.API_ID),
            api_hash=self.config.API_HASH,
            phone_number=self.config.PHONE_NUMBER,
            channel_username=self.config.STORAGE_CHANNEL
        )
        
        self.rag_agent = RAGAgent() # Our simplified RAGAgent
        self.telegram_client = None
        
    async def _setup_telegram_client(self):
        """Initialize Telegram client and authenticate"""
        await self.telegram_store.initialize()
        self.telegram_client = self.telegram_store.client
        
        @self.telegram_client.on(events.NewMessage(pattern='/start'))
        async def start_handler(event):
            await event.reply("Hello! I\'m a rule-based RAG chatbot. Send me a query!")
            
        @self.telegram_client.on(events.NewMessage(pattern='/add_document'))
        async def add_document_handler(event):
            # This is a placeholder. A proper document upload would involve more steps.
            await event.reply("Please send a document or a PDF file to add.")
            
        @self.telegram_client.on(events.NewMessage(pattern='/list_documents'))
        async def list_documents_handler(event):
            docs = await self.telegram_store.list_documents()
            if docs:
                doc_list_str = "\n".join([f"- {doc['id']} (Added: {doc['added_date'][:10]})" for doc in docs])
                await event.reply(f"Current documents in storage:\n{doc_list_str}")
            else:
                await event.reply("No documents found in storage.")

        @self.telegram_client.on(events.NewMessage(pattern='/list_faqs'))
        async def list_faqs_handler(event):
            rules = getattr(self.rag_agent, 'faq_rules', [])
            if not rules:
                await event.reply("No FAQ rules loaded.")
                return
            preview = []
            for i, r in enumerate(rules[:10], start=1):
                pats = r.get('patterns', [])
                ans = r.get('answer', '')
                preview.append(f"{i}. Patterns: {', '.join(pats[:3])} | Answer: {ans[:60]}...")
            await event.reply("Loaded FAQ rules (showing up to 10):\n" + "\n".join(preview))
                
        @self.telegram_client.on(events.NewMessage)
        async def message_handler(event):
            if event.raw_text.startswith(tuple(['/start', '/add_document', '/list_documents'])):
                return # Handled by specific handlers
                
            if event.message.document:
                await self._handle_document_upload(event)
            elif event.raw_text:
                await self.handle_text_query(event)
                
    async def _handle_document_upload(self, event):
        doc_name = "unknown"
        if event.message.document.attributes:
            for attr in event.message.document.attributes:
                if isinstance(attr, DocumentAttributeFilename):
                    doc_name = attr.file_name
                    break
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, doc_name)
            await event.message.download_media(file=file_path)
            
            # Assuming add_documents_from_files can handle a single file path
            # For simplicity, we'll read the file content and add directly
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc_id = os.path.basename(doc_name)
                metadata = {'file_name': doc_name, 'file_type': event.message.document.mime_type}
                
                success = await self.telegram_store.add_document(doc_id, content, metadata)
                if success:
                    await self.rag_agent.document_store.add_document(doc_id, content, metadata) # Add to local RAG store
                    self.rag_agent.initialize() # Re-initialize RAG agent after adding doc
                    await event.reply(f"✅ Document '{doc_name}' added to storage and RAG agent.")
                else:
                    await event.reply(f"❌ Failed to add document '{doc_name}'. It might already exist or an error occurred.")
            except Exception as e:
                await event.reply(f"❌ Error processing document: {str(e)}")
        
    async def _load_documents_from_telegram(self):
        """Load existing documents from Telegram into the RAGAgent's DocumentStore"""
        print("🔄 Loading documents from Telegram into RAG agent...")
        telegram_docs = await self.telegram_store.get_all_documents()
        for doc in telegram_docs:
            self.rag_agent.document_store.add_document(
                doc_id=doc['id'],
                content=doc['content'],
                metadata=doc['metadata']
            )
        if telegram_docs:
            self.rag_agent.initialize() # Initialize RAG agent with loaded docs
        print(f"✅ Loaded {len(telegram_docs)} documents from Telegram into RAG agent.")
        
    async def handle_text_query(self, event):
        user_query = event.raw_text
        print(f"Received query: {user_query}")
        
        # Use the rule-based RAGAgent to generate a response
        response = self.rag_agent.generate_enhanced_response(user_query, top_k=3)
        
        answer_text = response.get('answer', "I'm sorry, I couldn't find a direct answer.")
        
        # Optionally include snippets for transparency
        if response.get('enhanced_results'):
            snippets = "\n\nRelevant Snippets:\n"
            for i, res in enumerate(response['enhanced_results'][:2]): # Limit to 2 snippets
                snippets += f"---\nDocument ID: {res['document_id']}\nSnippet: {res['highlighted_snippet']}\n"
            answer_text += snippets
            
        await event.reply(answer_text)
        
    async def start(self):
        """Start the Telegram bot"""
        await self._setup_telegram_client()
        await self._load_documents_from_telegram()
        
        print("🚀 Rule-Based Telegram RAG Agent started! Listening for messages...")
        await self.telegram_client.run_until_disconnected()
        
async def run_telegram_bot():
    agent = RuleBasedTelegramRAGAgent()
    await agent.start()

if __name__ == '__main__':
    asyncio.run(run_telegram_bot())