#!/usr/bin/env python3
"""
Direct test of Telegram integration with hardcoded credentials
"""

import asyncio
import os
from telegram_document_store import TelegramDocumentStore

async def test_telegram_direct():
    """Test Telegram with direct credentials"""
    print("🧪 Testing Telegram with Direct Credentials")
    print("=" * 50)
    
    # Your actual credentials
    API_ID = "20066595"
    API_HASH = "337227b1ca9a5c77bf2fcb8f0cc1696d"
    PHONE_NUMBER = "+917286066438"
    
    try:
        # Create Telegram store
        telegram_store = TelegramDocumentStore(
            api_id=API_ID,
            api_hash=API_HASH,
            phone_number=PHONE_NUMBER,
            channel_username=None  # Use Saved Messages
        )
        
        print("📱 Initializing Telegram client...")
        print("⚠️ You may need to enter a verification code sent to your phone")
        
        await telegram_store.initialize()
        
        print("✅ Telegram client initialized successfully!")
        
        # Test basic operations
        print("\n🧪 Testing document operations...")
        
        # Add a test document
        test_doc_id = "test_direct_doc"
        test_content = "This is a test document for direct Telegram integration testing."
        
        success = await telegram_store.add_document(test_doc_id, test_content, {
            'type': 'direct_test',
            'created_at': '2025-01-01'
        })
        
        if success:
            print("✅ Document upload successful")
            
            # Try to retrieve it
            retrieved_doc = await telegram_store.get_document(test_doc_id)
            if retrieved_doc:
                print("✅ Document retrieval successful")
                print(f"   Content: {retrieved_doc['content'][:50]}...")
                
                # Clean up test document
                await telegram_store.remove_document(test_doc_id)
                print("✅ Document cleanup successful")
            else:
                print("❌ Document retrieval failed")
        else:
            print("❌ Document upload failed")
        
        # Get storage stats
        stats = await telegram_store.get_storage_stats()
        print(f"\n📊 Storage Statistics:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Storage backend: {stats['storage_backend']}")
        print(f"   Channel: {stats['channel']}")
        print(f"   Phone: {stats['phone_number']}")
        
        await telegram_store.close()
        print("\n✅ Telegram storage test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Telegram test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_telegram_direct())
        if success:
            print("\n🎉 Telegram integration is working!")
            print("You can now use the full RAG system with Telegram storage.")
        else:
            print("\n⚠️ Telegram integration needs attention.")
            print("The system will fall back to local storage.")
    except KeyboardInterrupt:
        print("\n❌ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")