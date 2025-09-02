#!/usr/bin/env python3
"""
Quick test script for Telegram Document Storage integration
"""

import asyncio
import os
from dotenv import load_dotenv
from telegram_config import TelegramConfig

# Load environment variables
load_dotenv()

async def test_telegram_setup():
    """Test if Telegram is properly configured"""
    print("🧪 Testing Telegram Document Storage Setup")
    print("=" * 50)
    
    # Test 1: Configuration validation
    print("1. Testing configuration...")
    try:
        TelegramConfig.validate_config()
        print("   ✅ Telegram configuration is valid")
        
        print(f"   📱 Phone: {TelegramConfig.PHONE_NUMBER}")
        print(f"   🆔 API ID: {TelegramConfig.API_ID}")
        print(f"   🔑 API Hash: {'*' * (len(TelegramConfig.API_HASH) - 4) + TelegramConfig.API_HASH[-4:]}")
        
        if TelegramConfig.STORAGE_CHANNEL:
            print(f"   📺 Channel: {TelegramConfig.STORAGE_CHANNEL}")
        else:
            print("   📁 Storage: Saved Messages")
            
    except ValueError as e:
        print(f"   ❌ Configuration error: {e}")
        print("\n🔧 Run setup script: python setup_telegram_storage.py")
        return False
    
    # Test 2: Import dependencies
    print("\n2. Testing dependencies...")
    try:
        from telegram_document_store import TelegramDocumentStore
        from rag_agent_telegram import create_telegram_rag_agent
        print("   ✅ All imports successful")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Install dependencies: pip install -r requirements.txt")
        return False
    
    # Test 3: Basic connection (without full initialization)
    print("\n3. Testing basic connection...")
    try:
        from telethon import TelegramClient
        
        # Create client (don't start it yet)
        client = TelegramClient(
            'test_session', 
            TelegramConfig.API_ID, 
            TelegramConfig.API_HASH
        )
        
        print("   ✅ Telegram client created successfully")
        print("   ℹ️ Full connection test requires phone verification")
        
    except Exception as e:
        print(f"   ❌ Connection setup failed: {e}")
        return False
    
    # Test 4: Gemini AI configuration (optional)
    print("\n4. Testing AI configuration...")
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print("   ✅ Gemini API key found")
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            print("   ✅ Gemini AI configured")
        except Exception as e:
            print(f"   ⚠️ Gemini setup issue: {e}")
    else:
        print("   ⚠️ Gemini API key not found (AI features will be limited)")
    
    print("\n" + "=" * 50)
    print("✅ Basic setup test completed!")
    print("\n📋 Next steps:")
    print("1. Run full test: python telegram_example.py")
    print("2. Start web app: streamlit run streamlit_telegram_app.py")
    print("3. Or run setup: python setup_telegram_storage.py")
    
    return True

def test_local_fallback():
    """Test local storage fallback"""
    print("\n🔄 Testing local storage fallback...")
    
    try:
        from rag_agent import RAGAgent
        from document_store import DocumentStore
        
        # Test local components
        doc_store = DocumentStore()
        print("   ✅ Local document store works")
        
        agent = RAGAgent()
        print("   ✅ Local RAG agent works")
        
        print("   ℹ️ Local fallback is available if Telegram fails")
        return True
        
    except Exception as e:
        print(f"   ❌ Local fallback failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 RAG Agent with Telegram Storage - Setup Test")
    print("This script verifies your setup without requiring phone verification")
    print()
    
    # Test Telegram setup
    telegram_ok = asyncio.run(test_telegram_setup())
    
    # Test local fallback
    local_ok = test_local_fallback()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Telegram Setup: {'✅ Ready' if telegram_ok else '❌ Needs Setup'}")
    print(f"   Local Fallback: {'✅ Available' if local_ok else '❌ Issues'}")
    
    if telegram_ok:
        print("\n🎉 Your system is ready for Telegram document storage!")
        print("Run 'python telegram_example.py' for a full demonstration.")
    else:
        print("\n🔧 Please run 'python setup_telegram_storage.py' to complete setup.")
    
    if not local_ok:
        print("\n⚠️ Local storage issues detected. Check your installation.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Please check your installation and configuration.")