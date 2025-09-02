#!/usr/bin/env python3
"""
Telegram Document Storage System
Store and retrieve documents using Telegram as a cloud storage backend
"""

import os
import json
import asyncio
import aiofiles
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import base64
from telethon import TelegramClient
from telethon.tl.types import DocumentAttributeFilename
import tempfile
import pickle

class TelegramDocumentStore:
    def __init__(self, api_id: str, api_hash: str, phone_number: str, channel_username: str = None):
        """
        Initialize Telegram Document Store
        
        Args:
            api_id: Telegram API ID (get from https://my.telegram.org)
            api_hash: Telegram API Hash
            phone_number: Your phone number
            channel_username: Optional private channel for storage (recommended)
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.channel_username = channel_username
        self.client = None
        self.documents_index = {}  # Maps doc_id to message_id
        self.max_documents = 100
        
    async def initialize(self):
        """Initialize Telegram client and authenticate"""
        self.client = TelegramClient('rag_session', self.api_id, self.api_hash)
        await self.client.start(phone=self.phone_number)
        
        # Load existing document index
        await self.load_document_index()
        
        print(f"✅ Telegram client initialized for {self.phone_number}")
        if self.channel_username:
            print(f"📁 Using channel: {self.channel_username}")
        else:
            print("📁 Using 'Saved Messages' for storage")
    
    async def load_document_index(self):
        """Load document index from Telegram"""
        try:
            # Try to find and download the index file
            entity = await self.get_storage_entity()
            
            async for message in self.client.iter_messages(entity, limit=50):
                if (message.document and 
                    message.document.attributes and 
                    any(attr.file_name == 'rag_documents_index.json' 
                        for attr in message.document.attributes 
                        if hasattr(attr, 'file_name'))):
                    
                    # Download and load index
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        await self.client.download_media(message, tmp.name)
                        
                        with open(tmp.name, 'r') as f:
                            self.documents_index = json.load(f)
                        
                        os.unlink(tmp.name)
                    
                    print(f"📋 Loaded document index with {len(self.documents_index)} entries")
                    return
                    
        except Exception as e:
            print(f"⚠️ Could not load document index: {e}")
            self.documents_index = {}
    
    async def save_document_index(self):
        """Save document index to Telegram"""
        try:
            # Create temporary index file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(self.documents_index, tmp, indent=2)
                tmp_path = tmp.name
            
            # Upload to Telegram
            entity = await self.get_storage_entity()
            await self.client.send_file(
                entity,
                tmp_path,
                caption=f"📋 RAG Documents Index - Updated: {datetime.now().isoformat()}",
                attributes=[DocumentAttributeFilename('rag_documents_index.json')]
            )
            
            os.unlink(tmp_path)
            print("💾 Document index saved to Telegram")
            
        except Exception as e:
            print(f"❌ Failed to save document index: {e}")
    
    async def get_storage_entity(self):
        """Get the entity (channel or saved messages) for storage"""
        if self.channel_username:
            return await self.client.get_entity(self.channel_username)
        else:
            return 'me'  # Saved Messages
    
    async def add_document(self, doc_id: str, content: str, metadata: Dict = None) -> bool:
        """
        Add a document to Telegram storage
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Optional metadata dictionary
            
        Returns:
            bool: Success status
        """
        if len(self.documents_index) >= self.max_documents:
            raise ValueError(f"Maximum document limit ({self.max_documents}) reached")
        
        if doc_id in self.documents_index:
            print(f"⚠️ Document {doc_id} already exists")
            return False
        
        try:
            # Create document package
            doc_package = {
                'id': doc_id,
                'content': content,
                'metadata': metadata or {},
                'added_date': datetime.now().isoformat(),
                'content_hash': hashlib.md5(content.encode()).hexdigest()
            }
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(doc_package, tmp, indent=2, ensure_ascii=False)
                tmp_path = tmp.name
            
            # Upload to Telegram
            entity = await self.get_storage_entity()
            message = await self.client.send_file(
                entity,
                tmp_path,
                caption=f"📄 Document: {doc_id}\n🕒 Added: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                attributes=[DocumentAttributeFilename(f'doc_{doc_id}.json')]
            )
            
            # Update index
            self.documents_index[doc_id] = {
                'message_id': message.id,
                'added_date': doc_package['added_date'],
                'content_hash': doc_package['content_hash'],
                'metadata': metadata or {}
            }
            
            # Save updated index
            await self.save_document_index()
            
            os.unlink(tmp_path)
            print(f"✅ Document '{doc_id}' uploaded to Telegram")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add document '{doc_id}': {e}")
            return False
    
    async def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a document from Telegram storage
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dict: Document data or None if not found
        """
        if doc_id not in self.documents_index:
            print(f"❌ Document '{doc_id}' not found in index")
            return None
        
        try:
            entity = await self.get_storage_entity()
            message_id = self.documents_index[doc_id]['message_id']
            
            # Get the specific message
            message = await self.client.get_messages(entity, ids=message_id)
            
            if not message or not message.document:
                print(f"❌ Document message not found for '{doc_id}'")
                return None
            
            # Download document
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                await self.client.download_media(message, tmp.name)
                
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                os.unlink(tmp.name)
            
            return doc_data
            
        except Exception as e:
            print(f"❌ Failed to retrieve document '{doc_id}': {e}")
            return None
    
    async def get_all_documents(self) -> List[Dict]:
        """
        Retrieve all documents from Telegram storage
        
        Returns:
            List[Dict]: List of all documents
        """
        documents = []
        
        for doc_id in self.documents_index.keys():
            doc = await self.get_document(doc_id)
            if doc:
                documents.append(doc)
        
        print(f"📚 Retrieved {len(documents)} documents from Telegram")
        return documents
    
    async def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from Telegram storage
        
        Args:
            doc_id: Document identifier
            
        Returns:
            bool: Success status
        """
        if doc_id not in self.documents_index:
            print(f"❌ Document '{doc_id}' not found")
            return False
        
        try:
            entity = await self.get_storage_entity()
            message_id = self.documents_index[doc_id]['message_id']
            
            # Delete the message
            await self.client.delete_messages(entity, message_id)
            
            # Remove from index
            del self.documents_index[doc_id]
            
            # Save updated index
            await self.save_document_index()
            
            print(f"🗑️ Document '{doc_id}' removed from Telegram")
            return True
            
        except Exception as e:
            print(f"❌ Failed to remove document '{doc_id}': {e}")
            return False
    
    async def list_documents(self) -> List[Dict]:
        """
        List all documents with metadata
        
        Returns:
            List[Dict]: Document metadata list
        """
        doc_list = []
        for doc_id, info in self.documents_index.items():
            doc_list.append({
                'id': doc_id,
                'added_date': info['added_date'],
                'content_hash': info['content_hash'],
                'metadata': info.get('metadata', {})
            })
        
        return doc_list
    
    async def get_storage_stats(self) -> Dict:
        """
        Get storage statistics
        
        Returns:
            Dict: Storage statistics
        """
        return {
            'total_documents': len(self.documents_index),
            'max_documents': self.max_documents,
            'remaining_slots': self.max_documents - len(self.documents_index),
            'storage_backend': 'Telegram',
            'channel': self.channel_username or 'Saved Messages',
            'phone_number': self.phone_number
        }
    
    async def clear_all_documents(self):
        """Clear all documents from Telegram storage"""
        try:
            entity = await self.get_storage_entity()
            
            # Delete all document messages
            for doc_id, info in self.documents_index.items():
                try:
                    await self.client.delete_messages(entity, info['message_id'])
                except:
                    pass  # Continue even if some deletions fail
            
            # Clear index
            self.documents_index = {}
            
            # Save empty index
            await self.save_document_index()
            
            print("🗑️ All documents cleared from Telegram storage")
            
        except Exception as e:
            print(f"❌ Failed to clear documents: {e}")
    
    async def close(self):
        """Close Telegram client connection"""
        if self.client:
            await self.client.disconnect()
            print("🔌 Telegram client disconnected")


class TelegramRAGIntegration:
    """Integration class to use Telegram storage with existing RAG system"""
    
    def __init__(self, telegram_store: TelegramDocumentStore):
        self.telegram_store = telegram_store
        self.local_documents = []
    
    async def sync_from_telegram(self):
        """Sync documents from Telegram to local storage"""
        print("🔄 Syncing documents from Telegram...")
        self.local_documents = await self.telegram_store.get_all_documents()
        print(f"✅ Synced {len(self.local_documents)} documents")
        return self.local_documents
    
    async def add_document_to_telegram(self, doc_id: str, content: str, metadata: Dict = None):
        """Add document to Telegram and sync locally"""
        success = await self.telegram_store.add_document(doc_id, content, metadata)
        if success:
            # Add to local cache
            doc = {
                'id': doc_id,
                'content': content,
                'metadata': metadata or {},
                'added_date': datetime.now().isoformat()
            }
            self.local_documents.append(doc)
        return success
    
    def get_documents_for_rag(self) -> List[Dict]:
        """Get documents in format compatible with existing RAG system"""
        return self.local_documents