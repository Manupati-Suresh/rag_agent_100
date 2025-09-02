#!/usr/bin/env python3
"""
Setup script for Telegram Document Storage
Helps users configure and test Telegram storage integration
"""

import os
import asyncio
from dotenv import load_dotenv, set_key
from telegram_config import TelegramConfig
from telegram_document_store import TelegramDocumentStore
from rag_agent_telegram import create_telegram_rag_agent

def setup_environment():
    """Interactive setup for Telegram configuration"""
    print("🔧 Telegram Document Storage Setup")
    print("=" * 50)
    
    # Load existing .env
    load_dotenv()
    
    # Check if .env exists
    env_file = '.env'
    if not os.path.exists(env_file):
        print("📝 Creating .env file...")
        with open(env_file, 'w') as f:
            f.write("# Telegram Configuration\n")
    
    print("\n📋 You'll need to get Telegram API credentials from: https://my.telegram.org/apps")
    print("1. Log in with your phone number")
    print("2. Create a new application")
    print("3. Copy the API ID and API Hash")
    
    # Get API credentials
    api_id = input("\n🔑 Enter your Telegram API ID: ").strip()
    api_hash = input("🔑 Enter your Telegram API Hash: ").strip()
    phone_number = input("📱 Enter your phone number (with country code, e.g., +1234567890): ").strip()
    
    # Optional channel
    print("\n📁 Storage Options:")
    print("1. Use 'Saved Messages' (default, private)")
    print("2. Use a private channel (recommended for organization)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    storage_channel = ""
    if choice == "2":
        print("\n📺 To use a private channel:")
        print("1. Create a private Telegram channel")
        print("2. Add yourself as admin")
        print("3. Get the channel username (e.g., @my_docs_storage)")
        
        storage_channel = input("Enter channel username (or press Enter to skip): ").strip()
    
    # Save to .env
    set_key(env_file, 'TELEGRAM_API_ID', api_id)
    set_key(env_file, 'TELEGRAM_API_HASH', api_hash)
    set_key(env_file, 'TELEGRAM_PHONE_NUMBER', phone_number)
    
    if storage_channel:
        set_key(env_file, 'TELEGRAM_STORAGE_CHANNEL', storage_channel)
    
    print(f"\n✅ Configuration saved to {env_file}")
    
    return api_id, api_hash, phone_number, storage_channel

async def test_telegram_connection():
    """Test Telegram connection and basic functionality"""
    print("\n🧪 Testing Telegram Connection")
    print("=" * 40)
    
    try:
        # Validate configuration
        TelegramConfig.validate_config()
        print("✅ Configuration valid")
        
        # Create Telegram store
        telegram_store = TelegramDocumentStore(
            api_id=TelegramConfig.API_ID,
            api_hash=TelegramConfig.API_HASH,
            phone_number=TelegramConfig.PHONE_NUMBER,
            channel_username=TelegramConfig.STORAGE_CHANNEL
        )
        
        # Initialize (this will prompt for phone verification if needed)
        print("📱 Initializing Telegram client...")
        print("⚠️ You may need to enter a verification code sent to your phone")
        
        await telegram_store.initialize()
        
        # Test basic operations
        print("🧪 Testing document operations...")
        
        # Add a test document
        test_doc_id = "test_setup_doc"
        test_content = f"This is a test document created during setup at {os.getcwd()}"
        
        success = await telegram_store.add_document(test_doc_id, test_content, {
            'type': 'setup_test',
            'created_at': '2025-01-01'
        })
        
        if success:
            print("✅ Document upload successful")
            
            # Try to retrieve it
            retrieved_doc = await telegram_store.get_document(test_doc_id)
            if retrieved_doc:
                print("✅ Document retrieval successful")
                
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
        
        await telegram_store.close()
        print("\n✅ Telegram storage test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Telegram test failed: {e}")
        return False

async def test_full_rag_integration():
    """Test full RAG integration with Telegram storage"""
    print("\n🤖 Testing Full RAG Integration")
    print("=" * 40)
    
    try:
        # Create RAG agent with Telegram storage
        agent = await create_telegram_rag_agent(use_telegram=True)
        
        # Add some sample documents
        sample_docs = [
            {
                'id': 'sample_1',
                'content': 'Python is a high-level programming language known for its simplicity and readability.',
                'metadata': {'category': 'programming', 'language': 'python'}
            },
            {
                'id': 'sample_2', 
                'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.',
                'metadata': {'category': 'ai', 'topic': 'machine_learning'}
            }
        ]
        
        print("📤 Adding sample documents...")
        for doc in sample_docs:
            success = await agent.add_document(doc['id'], doc['content'], doc['metadata'])
            if success:
                print(f"   ✅ Added: {doc['id']}")
            else:
                print(f"   ❌ Failed: {doc['id']}")
        
        # Test search
        print("\n🔍 Testing search functionality...")
        results = agent.search_documents("Python programming", top_k=2)
        if results:
            print(f"   ✅ Search returned {len(results)} results")
        else:
            print("   ❌ Search failed")
        
        # Test AI chat (if Gemini is configured)
        print("\n🤖 Testing AI chat...")
        chat_response = agent.chat_with_documents("What is Python?")
        if chat_response['success']:
            print("   ✅ AI chat working")
        else:
            print(f"   ⚠️ AI chat issue: {chat_response.get('error', 'Unknown')}")
        
        # Cleanup sample documents
        print("\n🗑️ Cleaning up sample documents...")
        for doc in sample_docs:
            await agent.remove_document(doc['id'])
        
        await agent.close()
        print("\n✅ Full RAG integration test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG integration test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Welcome to Telegram Document Storage Setup!")
    print("This will help you configure Telegram as your document storage backend.")
    print()
    
    # Step 1: Environment setup
    try:
        setup_environment()
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        return
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return
    
    # Step 2: Test connection
    print("\n" + "="*60)
    test_connection = input("🧪 Test Telegram connection now? (y/n): ").lower().strip()
    
    if test_connection == 'y':
        try:
            success = asyncio.run(test_telegram_connection())
            if not success:
                print("❌ Connection test failed. Please check your configuration.")
                return
        except KeyboardInterrupt:
            print("\n❌ Test cancelled by user")
            return
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            return
    
    # Step 3: Test full integration
    print("\n" + "="*60)
    test_integration = input("🤖 Test full RAG integration? (y/n): ").lower().strip()
    
    if test_integration == 'y':
        try:
            success = asyncio.run(test_full_rag_integration())
            if success:
                print("\n🎉 Setup completed successfully!")
                print("\nNext steps:")
                print("1. Run: streamlit run streamlit_telegram_app.py")
                print("2. Or use: python -c 'import asyncio; from rag_agent_telegram import create_telegram_rag_agent; asyncio.run(create_telegram_rag_agent())'")
            else:
                print("\n⚠️ Integration test had issues, but basic setup is complete.")
        except KeyboardInterrupt:
            print("\n❌ Integration test cancelled by user")
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
    
    print("\n✅ Setup process completed!")
    print("Your Telegram document storage is ready to use!")

if __name__ == "__main__":
    main()