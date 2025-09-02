#!/usr/bin/env python3
"""
Demo script showcasing Gemini 2.5 Flash integration with RAG Agent
"""

from rag_agent import RAGAgent
import json

def main():
    print("🚀 RAG Agent with Gemini 2.5 Flash Integration Demo")
    print("=" * 60)
    
    # Initialize the RAG agent
    print("\n1. Initializing RAG Agent...")
    agent = RAGAgent()
    
    # Check Gemini status
    status = agent.get_model_status()
    print(f"   Gemini Available: {status['gemini_available']}")
    print(f"   API Key Configured: {status['api_key_configured']}")
    print(f"   Model: {status['model_name']}")
    
    if not status['gemini_available']:
        print("❌ Gemini not available. Please check your .env file and API key.")
        return
    
    # Load sample documents if needed
    if not agent.is_initialized:
        print("\n2. Loading sample documents...")
        agent.load_documents()
        agent.initialize()
    else:
        print(f"\n2. Using existing document collection ({len(agent.document_store.documents)} documents)")
    
    # Demo queries
    demo_queries = [
        "What are the main challenges in climate change?",
        "How does artificial intelligence impact society?",
        "What are the benefits of renewable energy?"
    ]
    
    print("\n" + "=" * 60)
    print("🤖 GEMINI-POWERED RAG RESPONSES")
    print("=" * 60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        
        # Generate response with Gemini
        response = agent.generate_response(query, top_k=3, use_llm=True)
        
        if response.get('has_llm_response'):
            print("🎯 AI-Generated Answer:")
            print(response['llm_response'])
            print(f"\n📚 Based on {len(response['retrieved_documents'])} relevant documents")
        else:
            print("❌ LLM response not available")
            if 'llm_error' in response:
                print(f"Error: {response['llm_error']}")
    
    # Demo enhanced responses with highlighting
    print("\n" + "=" * 60)
    print("✨ ENHANCED RESPONSES WITH HIGHLIGHTING")
    print("=" * 60)
    
    query = "renewable energy benefits"
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    enhanced_response = agent.generate_enhanced_response(query, top_k=2, use_llm=True)
    
    if enhanced_response.get('has_llm_response'):
        print("🎯 Enhanced AI Answer:")
        print(enhanced_response['llm_response'])
        
        print("\n📋 Source Highlights:")
        for result in enhanced_response['enhanced_results']:
            print(f"  • Document {result['rank']}: {result['document_id']}")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Snippet: {result['highlighted_snippet'][:100]}...")
    
    # Demo question answering
    print("\n" + "=" * 60)
    print("❓ QUESTION ANSWERING DEMO")
    print("=" * 60)
    
    questions = [
        "What are the main types of renewable energy?",
        "How can AI help with climate change?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 50)
        
        answer = agent.ask_question(question, top_k=3, response_style='comprehensive')
        
        if answer.get('success'):
            print("💡 Answer:")
            print(answer['answer'])
            print(f"\n📖 Sources: {answer['source_count']} documents")
        else:
            print(f"❌ Error: {answer.get('error', 'Unknown error')}")
    
    # Demo conversation
    print("\n" + "=" * 60)
    print("💬 CONVERSATIONAL DEMO")
    print("=" * 60)
    
    conversation_history = []
    
    # First message
    message1 = "Tell me about climate change impacts"
    print(f"\n👤 User: {message1}")
    
    chat_response1 = agent.chat_with_documents(message1, conversation_history)
    if chat_response1.get('success'):
        print(f"🤖 Assistant: {chat_response1['assistant_response']}")
        conversation_history.append({
            'user': message1,
            'assistant': chat_response1['assistant_response']
        })
    
    # Follow-up message
    message2 = "What solutions are mentioned for these impacts?"
    print(f"\n👤 User: {message2}")
    
    chat_response2 = agent.chat_with_documents(message2, conversation_history)
    if chat_response2.get('success'):
        print(f"🤖 Assistant: {chat_response2['assistant_response']}")
    
    # Demo summary generation
    print("\n" + "=" * 60)
    print("📄 DOCUMENT SUMMARY DEMO")
    print("=" * 60)
    
    summary_query = "artificial intelligence applications"
    print(f"\nGenerating summary for: {summary_query}")
    print("-" * 50)
    
    summary = agent.generate_summary(summary_query, top_k=4)
    if summary.get('success'):
        print("📋 Summary:")
        print(summary['summary'])
        print(f"\n📚 Based on {summary['source_documents']} documents")
    else:
        print(f"❌ Error: {summary.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed! Your RAG agent is now powered by Gemini 2.5 Flash")
    print("=" * 60)

if __name__ == "__main__":
    main()