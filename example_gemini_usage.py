#!/usr/bin/env python3
"""
Comprehensive example showcasing Gemini integration with RAG Agent
"""

from rag_agent import RAGAgent
import json

def main():
    print("🚀 RAG Agent with Gemini 2.5 Flash - Complete Example")
    print("=" * 60)
    
    # 1. Initialize the agent
    print("\n1. 🔧 Initializing RAG Agent...")
    agent = RAGAgent()
    
    # Check Gemini status
    status = agent.get_model_status()
    print(f"   • Gemini Available: {'✅' if status['gemini_available'] else '❌'}")
    print(f"   • API Key Configured: {'✅' if status['api_key_configured'] else '❌'}")
    print(f"   • Model: {status['model_name']}")
    
    if not status['gemini_available']:
        print("\n❌ Gemini not available. Please:")
        print("   1. Get API key from https://makersuite.google.com/app/apikey")
        print("   2. Add GOOGLE_API_KEY=your_key to .env file")
        return
    
    # 2. Load documents
    if not agent.is_initialized:
        print("\n2. 📚 Loading sample documents...")
        agent.load_documents()
        agent.initialize()
        print(f"   • Loaded {len(agent.document_store.documents)} documents")
    else:
        print(f"\n2. 📚 Using existing collection ({len(agent.document_store.documents)} documents)")
    
    # 3. Basic AI-powered search
    print("\n" + "=" * 60)
    print("3. 🔍 AI-POWERED SEARCH")
    print("=" * 60)
    
    query = "renewable energy benefits"
    print(f"\nQuery: '{query}'")
    print("-" * 40)
    
    response = agent.generate_response(query, top_k=3, use_llm=True)
    
    if response.get('has_llm_response'):
        print("🤖 AI Response:")
        print(response['llm_response'])
        print(f"\n📊 Sources: {len(response['retrieved_documents'])} documents")
    else:
        print("❌ No AI response generated")
    
    # 4. Question Answering with different styles
    print("\n" + "=" * 60)
    print("4. ❓ QUESTION ANSWERING")
    print("=" * 60)
    
    questions = [
        ("What are the main types of renewable energy?", "comprehensive"),
        ("How does solar energy work?", "concise"),
        ("What are the economic impacts of renewable energy?", "analytical")
    ]
    
    for question, style in questions:
        print(f"\n❓ Question: {question}")
        print(f"📝 Style: {style}")
        print("-" * 50)
        
        answer = agent.ask_question(question, response_style=style)
        
        if answer.get('success'):
            print("💡 Answer:")
            print(answer['answer'])
            print(f"\n📖 Sources: {answer['source_count']} documents")
        else:
            print(f"❌ Error: {answer.get('error')}")
    
    # 5. Conversational Chat
    print("\n" + "=" * 60)
    print("5. 💬 CONVERSATIONAL CHAT")
    print("=" * 60)
    
    conversation_history = []
    
    # First message
    message1 = "Tell me about climate change impacts"
    print(f"\n👤 User: {message1}")
    
    chat1 = agent.chat_with_documents(message1, conversation_history)
    if chat1.get('success'):
        print(f"🤖 Assistant: {chat1['assistant_response']}")
        conversation_history.append({
            'user': message1,
            'assistant': chat1['assistant_response']
        })
    
    # Follow-up message
    message2 = "What solutions are available to address these impacts?"
    print(f"\n👤 User: {message2}")
    
    chat2 = agent.chat_with_documents(message2, conversation_history)
    if chat2.get('success'):
        print(f"🤖 Assistant: {chat2['assistant_response']}")
    
    # 6. Document Summaries
    print("\n" + "=" * 60)
    print("6. 📄 DOCUMENT SUMMARIES")
    print("=" * 60)
    
    summary_topics = [
        "artificial intelligence applications",
        "sustainable development goals",
        "technology innovation trends"
    ]
    
    for topic in summary_topics:
        print(f"\n📋 Topic: {topic}")
        print("-" * 40)
        
        summary = agent.generate_summary(topic, top_k=4)
        
        if summary.get('success'):
            print("📄 Summary:")
            print(summary['summary'])
            print(f"\n📚 Based on {summary['source_documents']} documents")
        else:
            print(f"❌ Error: {summary.get('error')}")
        
        print()  # Extra spacing
    
    # 7. Enhanced RAG with Highlighting
    print("\n" + "=" * 60)
    print("7. ✨ ENHANCED RAG WITH HIGHLIGHTING")
    print("=" * 60)
    
    enhanced_query = "machine learning algorithms"
    print(f"\nQuery: '{enhanced_query}'")
    print("-" * 40)
    
    enhanced_response = agent.generate_enhanced_response(
        enhanced_query, 
        top_k=3, 
        use_llm=True
    )
    
    if enhanced_response.get('has_llm_response'):
        print("🎯 Enhanced AI Response:")
        print(enhanced_response['llm_response'])
        
        print("\n🔍 Highlighted Sources:")
        for i, result in enumerate(enhanced_response['enhanced_results'], 1):
            print(f"\n  {i}. Document: {result['document_id']}")
            print(f"     Score: {result['score']:.3f}")
            print(f"     Snippet: {result['highlighted_snippet'][:150]}...")
    
    # 8. Advanced Search with Highlighting Options
    print("\n" + "=" * 60)
    print("8. 🎨 ADVANCED HIGHLIGHTING OPTIONS")
    print("=" * 60)
    
    search_query = "sustainable energy"
    print(f"\nQuery: '{search_query}'")
    print("-" * 40)
    
    # Test different highlighting options
    highlighting_options = [
        {"exact_match_only": True, "name": "Exact Match Only"},
        {"exact_match_only": False, "include_synonyms": True, "name": "With Synonyms"},
        {"exact_match_only": False, "include_related": True, "name": "With Related Terms"}
    ]
    
    for options in highlighting_options:
        name = options.pop('name')
        print(f"\n🎨 {name}:")
        
        results = agent.search_with_highlights(
            search_query, 
            top_k=2, 
            snippet_length=200,
            **options
        )
        
        for result in results[:1]:  # Show first result only
            print(f"   • {result['document_id']}")
            print(f"   • Snippet: {result['highlighted_snippet'][:100]}...")
    
    # 9. Performance and Statistics
    print("\n" + "=" * 60)
    print("9. 📊 STATISTICS & PERFORMANCE")
    print("=" * 60)
    
    stats = agent.get_stats()
    print(f"\n📈 Document Collection Stats:")
    print(f"   • Total Documents: {stats['total_documents']}")
    print(f"   • Remaining Slots: {stats['remaining_slots']}")
    print(f"   • Storage Path: {stats['storage_path']}")
    print(f"   • Initialized: {stats['is_initialized']}")
    print(f"   • Embedding Dimension: {stats['embedding_dimension']}")
    
    # Test caching performance
    print(f"\n⚡ Testing Cache Performance:")
    import time
    
    # First query (no cache)
    start_time = time.time()
    agent.search_with_highlights("renewable energy", top_k=3)
    first_time = time.time() - start_time
    
    # Second query (with cache)
    start_time = time.time()
    agent.search_with_highlights("renewable energy", top_k=3)
    cached_time = time.time() - start_time
    
    speedup = first_time / cached_time if cached_time > 0 else float('inf')
    print(f"   • First query: {first_time:.3f}s")
    print(f"   • Cached query: {cached_time:.3f}s")
    print(f"   • Speedup: {speedup:.1f}x")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE DEMO FINISHED!")
    print("=" * 60)
    print("\n🎉 Your RAG Agent is fully operational with:")
    print("   • Google Gemini 2.5 Flash AI integration")
    print("   • Advanced highlighting and search")
    print("   • Conversational capabilities")
    print("   • Question answering with multiple styles")
    print("   • Document summarization")
    print("   • Performance optimization with caching")
    print("\n🚀 Ready for production use!")

if __name__ == "__main__":
    main()