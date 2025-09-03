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
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class TelegramRAGAgent:
    def __init__(self, use_telegram: bool = True):
        self.use_telegram = use_telegram

    async def initialize_telegram(self):
        # Placeholder: Implement actual Telegram initialization logic
        return True

    async def load_documents(self):
        # Placeholder: Implement document loading logic
        pass

# Async wrapper functions for easier use
async def create_telegram_rag_agent(use_telegram: bool = True) -> TelegramRAGAgent:
    """Create and initialize a Telegram RAG Agent"""
    agent = TelegramRAGAgent(use_telegram=use_telegram)
    
    if use_telegram:
        success = await agent.initialize_telegram()
        if not success:
            print(" Falling back to local storage")
            agent.use_telegram = False
    
    await agent.load_documents()
    
    if len(agent.document_store.documents) > 0:
        agent.initialize_search_index()
    
    return agent