#!/usr/bin/env python3
"""
Demo showing the difference between regular search and enhanced highlighting search
"""

from rag_agent import RAGAgent

def main():
    print("🔍 RAG Agent: Before vs After Comparison")
    print("=" * 60)
    
    # Initialize agent
    agent = RAGAgent(storage_path='test_documents')
    
    # Ensure index is built
    if len(agent.document_store.documents) > 0:
        if not agent.is_initialized or agent.document_store.index is None:
            print("Building search index...")
            agent.initialize()
    else:
        print("Please run test_highlighting.py first to set up test documents")
        return
    
    query = "machine learning algorithms"
    
    print(f"Query: '{query}'")
    print("\n" + "="*60)
    
    # BEFORE: Regular search (full documents)
    print("❌ BEFORE - Full Document Results:")
    print("-" * 40)
    
    regular_results = agent.search(query, top_k=2)
    for result in regular_results:
        print(f"\n📄 {result['document_id']} (Score: {result['score']:.3f})")
        print("Full Content:")
        print(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
    
    print("\n" + "="*60)
    
    # AFTER: Enhanced search (highlighted snippets)
    print("✅ AFTER - Highlighted Snippet Results:")
    print("-" * 40)
    
    enhanced_results = agent.search_with_highlights(query, top_k=2, snippet_length=200)
    for result in enhanced_results:
        print(f"\n📄 {result['document_id']} (Score: {result['score']:.3f})")
        print("🎯 Highlighted Snippet:")
        # Clean HTML for console display
        snippet = result['highlighted_snippet'].replace('<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">', '**').replace('</mark>', '**')
        print(snippet)
        
        print("\n📝 Key Sentences:")
        for i, sentence in enumerate(result['relevant_sentences'][:2], 1):
            clean_sentence = sentence.replace('<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">', '**').replace('</mark>', '**')
            print(f"  {i}. {clean_sentence}")
    
    print("\n" + "="*60)
    print("🎯 KEY IMPROVEMENTS:")
    print("✅ Only relevant portions shown (not entire documents)")
    print("✅ Keywords highlighted for quick scanning")
    print("✅ Multiple relevance chunks per document")
    print("✅ Context-aware sentence extraction")
    print("✅ Faster reading and better user experience")
    print("✅ Boss requirement fulfilled! 🎉")

if __name__ == "__main__":
    main()