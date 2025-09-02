#!/usr/bin/env python3
"""
Quick test script for Gemini integration
"""

from rag_agent import RAGAgent
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

def test_gemini_integration():
    print("🧪 Testing Gemini Integration")
    print("=" * 40)
    
    # Check environment
    api_key = os.getenv('GOOGLE_API_KEY')
    print(f"API Key configured: {'✅' if api_key else '❌'}")
    
    if not api_key:
        print("Please set GOOGLE_API_KEY in your .env file")
        return
    
    # Initialize agent
    print("\n1. Initializing RAG Agent...")
    agent = RAGAgent()
    
    # Check model status
    status = agent.get_model_status()
    print(f"   Gemini Available: {'✅' if status['gemini_available'] else '❌'}")
    print(f"   Model: {status['model_name']}")
    
    if not status['gemini_available']:
        print("❌ Gemini not available")
        return
    
    # Load documents if needed
    if not agent.is_initialized:
        print("\n2. Loading sample documents...")
        agent.load_documents()
        agent.initialize()
    else:
        print(f"\n2. Using existing collection ({len(agent.document_store.documents)} documents)")
        # Make sure index is built
        if not hasattr(agent.document_store, 'index') or agent.document_store.index is None:
            print("   Building search index...")
            agent.initialize()
    
    # Test basic LLM response
    print("\n3. Testing basic LLM response...")
    try:
        response = agent.generate_response("What is artificial intelligence?", use_llm=True)
        if response.get('has_llm_response'):
            print("✅ LLM Response generated successfully!")
            print(f"Response length: {len(response['llm_response'])} characters")
        else:
            print("❌ No LLM response generated")
            if 'llm_error' in response:
                print(f"Error: {response['llm_error']}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test question answering
    print("\n4. Testing question answering...")
    try:
        answer = agent.ask_question("What are the benefits of renewable energy?")
        if answer.get('success'):
            print("✅ Question answering works!")
            print(f"Answer length: {len(answer['answer'])} characters")
        else:
            print(f"❌ Error: {answer.get('error')}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 40)
    print("✅ Gemini integration test completed!")

if __name__ == "__main__":
    test_gemini_integration()