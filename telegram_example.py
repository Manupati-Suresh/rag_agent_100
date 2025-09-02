#!/usr/bin/env python3
"""
Example usage of Telegram Document Storage with RAG Agent
"""

import asyncio
import os
from rag_agent_telegram import create_telegram_rag_agent

async def main():
    """Example usage of Telegram RAG Agent"""
    print("🚀 Telegram RAG Agent Example")
    print("=" * 40)
    
    # Create agent with Telegram storage
    print("🔄 Initializing RAG Agent with Telegram storage...")
    agent = await create_telegram_rag_agent(use_telegram=True)
    
    # Add some example documents
    sample_documents = [
        {
            'id': 'python_basics',
            'content': '''Python is a high-level, interpreted programming language with dynamic semantics. 
            Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
            make it very attractive for Rapid Application Development, as well as for use as a scripting 
            or glue language to connect existing components together.''',
            'metadata': {'category': 'programming', 'language': 'python', 'level': 'beginner'}
        },
        {
            'id': 'ai_overview',
            'content': '''Artificial Intelligence (AI) is intelligence demonstrated by machines, 
            in contrast to the natural intelligence displayed by humans and animals. Leading AI 
            textbooks define the field as the study of "intelligent agents": any device that 
            perceives its environment and takes actions that maximize its chance of successfully 
            achieving its goals.''',
            'metadata': {'category': 'technology', 'topic': 'artificial_intelligence', 'level': 'intermediate'}
        },
        {
            'id': 'machine_learning',
            'content': '''Machine learning (ML) is a type of artificial intelligence (AI) that allows 
            software applications to become more accurate at predicting outcomes without being explicitly 
            programmed to do so. Machine learning algorithms use historical data as input to predict 
            new output values.''',
            'metadata': {'category': 'technology', 'topic': 'machine_learning', 'level': 'intermediate'}
        }
    ]
    
    # Add documents to Telegram storage
    print("\n📤 Adding documents to Telegram storage...")
    for doc in sample_documents:
        success = await agent.add_document(doc['id'], doc['content'], doc['metadata'])
        if success:
            print(f"   ✅ Added: {doc['id']}")
        else:
            print(f"   ❌ Failed to add: {doc['id']}")
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    search_queries = [
        "Python programming language",
        "artificial intelligence",
        "machine learning algorithms"
    ]
    
    for query in search_queries:
        print(f"\n   Query: '{query}'")
        results = agent.search_documents(query, top_k=2)
        for result in results:
            print(f"      📄 {result['document_id']} (Score: {result['score']:.3f})")
    
    # Test AI chat functionality
    print("\n🤖 Testing AI chat functionality...")
    chat_questions = [
        "What is Python?",
        "Explain the difference between AI and machine learning",
        "What are the benefits of using Python for programming?"
    ]
    
    for question in chat_questions:
        print(f"\n   Question: '{question}'")
        response = agent.chat_with_documents(question)
        if response['success']:
            print(f"   Answer: {response['response'][:200]}...")
            if response.get('sources'):
                print(f"   Sources: {[s['document_id'] for s in response['sources']]}")
        else:
            print(f"   ❌ Error: {response.get('error')}")
    
    # Test Q&A with different styles
    print("\n❓ Testing Q&A with different response styles...")
    question = "What is artificial intelligence?"
    styles = ["brief", "comprehensive", "bullet_points"]
    
    for style in styles:
        print(f"\n   Style: {style}")
        result = agent.ask_question(question, response_style=style)
        if result['success']:
            print(f"   Answer: {result['answer'][:150]}...")
        else:
            print(f"   ❌ Error: {result.get('error')}")
    
    # Test summary generation
    print("\n📄 Testing summary generation...")
    summary_result = agent.generate_summary("programming and AI")
    if summary_result['success']:
        print(f"   Summary: {summary_result['summary'][:200]}...")
    else:
        print(f"   ❌ Error: {summary_result.get('error')}")
    
    # Get storage statistics
    print("\n📊 Storage Statistics:")
    stats = await agent.get_storage_stats()
    if 'telegram' in stats:
        print(f"   Telegram Documents: {stats['telegram']['total_documents']}")
        print(f"   Storage Channel: {stats['telegram']['channel']}")
    if 'local_cache' in stats:
        print(f"   Local Cache: {stats['local_cache']['total_documents']} documents")
    
    # Cleanup (optional - remove test documents)
    cleanup = input("\n🗑️ Remove test documents? (y/n): ").lower().strip()
    if cleanup == 'y':
        print("🗑️ Cleaning up test documents...")
        for doc in sample_documents:
            success = await agent.remove_document(doc['id'])
            if success:
                print(f"   ✅ Removed: {doc['id']}")
            else:
                print(f"   ❌ Failed to remove: {doc['id']}")
    
    # Close connections
    await agent.close()
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Example cancelled by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("\nMake sure you have:")
        print("1. Configured Telegram API credentials in .env")
        print("2. Run: python setup_telegram_storage.py")
        print("3. Installed requirements: pip install -r requirements.txt")