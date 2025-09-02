#!/usr/bin/env python3
"""
Test script for enhanced highlighting features
"""

from text_highlighter import TextHighlighter
from rag_agent import RAGAgent
import json

def test_enhanced_highlighting():
    """Test the enhanced highlighting features"""
    
    print("🚀 Testing Enhanced Highlighting Features")
    print("=" * 50)
    
    # Initialize highlighter
    highlighter = TextHighlighter()
    
    # Sample text for testing
    sample_text = """
    Machine learning is a powerful subset of artificial intelligence that enables computers 
    to learn and improve from experience without being explicitly programmed. The core concept 
    involves algorithms that can identify patterns in data and make predictions or decisions 
    based on those patterns. There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning. Supervised learning uses labeled training 
    data to learn a mapping function from input variables to output variables. Deep learning, 
    a subset of machine learning, uses neural networks with multiple layers to model and understand 
    complex patterns in data. Natural language processing is another important application of 
    machine learning that focuses on the interaction between computers and human language.
    """
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "supervised learning neural networks",
        "artificial intelligence patterns",
        "deep learning applications"
    ]
    
    print("\n1. Testing Enhanced Keyword Extraction")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Basic keywords
        basic_keywords = highlighter._extract_keywords(query)
        print(f"Basic keywords: {basic_keywords}")
        
        # Enhanced keywords
        enhanced_keywords = highlighter._extract_enhanced_keywords(query)
        print("Enhanced keywords:")
        for kw in enhanced_keywords:
            print(f"  - {kw['word']} (importance: {kw['importance']:.2f}, POS: {kw['pos']})")
        
        # Phrases
        phrases = highlighter._extract_phrases(query)
        print(f"Phrases: {phrases}")
    
    print("\n\n2. Testing Advanced Highlighting")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("Basic highlighting:")
        basic_highlight = highlighter.highlight_keywords(sample_text, query)
        print(basic_highlight[:200] + "...")
        
        print("\nAdvanced highlighting:")
        advanced_highlight = highlighter.highlight_keywords_advanced(sample_text, query)
        print(advanced_highlight[:200] + "...")
    
    print("\n\n3. Testing Enhanced Chunking")
    print("-" * 40)
    
    query = "machine learning algorithms"
    
    # Basic chunking
    basic_chunks = highlighter.extract_relevant_chunks(
        sample_text, query, chunk_size=150, use_semantic_chunking=False
    )
    print(f"Basic chunking found {len(basic_chunks)} chunks")
    
    # Semantic chunking
    semantic_chunks = highlighter.extract_relevant_chunks(
        sample_text, query, chunk_size=150, use_semantic_chunking=True
    )
    print(f"Semantic chunking found {len(semantic_chunks)} chunks")
    
    # Compare top chunks
    if basic_chunks and semantic_chunks:
        print(f"\nBasic chunk score: {basic_chunks[0]['relevance_score']:.3f}")
        print(f"Semantic chunk score: {semantic_chunks[0]['relevance_score']:.3f}")
    
    print("\n\n4. Testing Contextual Snippets")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Basic snippet
        basic_snippet = highlighter.create_snippet(sample_text, query, max_length=200)
        print("Basic snippet:")
        print(basic_snippet)
        
        # Contextual snippet
        contextual_snippet = highlighter.create_contextual_snippet(sample_text, query, max_length=200)
        print("\nContextual snippet:")
        print(f"Snippet: {contextual_snippet['snippet']}")
        print(f"Relevance: {contextual_snippet['relevance_score']:.3f}")
        print(f"Keywords found: {contextual_snippet['keyword_count']}")
        print(f"Key sentences: {len(contextual_snippet['key_sentences'])}")

def test_rag_integration():
    """Test enhanced highlighting with RAG agent"""
    
    print("\n\n🤖 Testing RAG Integration")
    print("=" * 50)
    
    # Initialize RAG agent
    agent = RAGAgent()
    
    # Load sample documents if not already loaded
    if len(agent.document_store.documents) == 0:
        print("Loading sample documents...")
        agent.load_documents()
        agent.initialize()
    
    # Test enhanced search
    query = "machine learning algorithms"
    print(f"\nTesting enhanced search with query: '{query}'")
    
    # Basic search
    basic_results = agent.search(query, top_k=2)
    print(f"Basic search found {len(basic_results)} results")
    
    # Enhanced search
    enhanced_results = agent.search_with_highlights(
        query, top_k=2, use_advanced_highlighting=True, include_contextual_info=True
    )
    print(f"Enhanced search found {len(enhanced_results)} results")
    
    # Compare results
    if enhanced_results:
        result = enhanced_results[0]
        print(f"\nEnhanced result for document: {result['document_id']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Snippet info: {result['snippet_info']}")
        print(f"Highlighted snippet: {result['highlighted_snippet'][:200]}...")
        print(f"Relevant sentences: {len(result['relevant_sentences'])}")
        print(f"Relevant chunks: {len(result['relevant_chunks'])}")

def benchmark_performance():
    """Benchmark the performance improvements"""
    
    print("\n\n⚡ Performance Benchmark")
    print("=" * 50)
    
    import time
    
    highlighter = TextHighlighter()
    
    # Large sample text
    large_text = """
    Machine learning is a powerful subset of artificial intelligence that enables computers 
    to learn and improve from experience without being explicitly programmed. The core concept 
    involves algorithms that can identify patterns in data and make predictions or decisions 
    based on those patterns. There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning. Supervised learning uses labeled training 
    data to learn a mapping function from input variables to output variables. Deep learning, 
    a subset of machine learning, uses neural networks with multiple layers to model and understand 
    complex patterns in data. Natural language processing is another important application of 
    machine learning that focuses on the interaction between computers and human language.
    """ * 10  # Make it larger
    
    query = "machine learning deep learning neural networks"
    
    # Benchmark basic highlighting
    start_time = time.time()
    for _ in range(10):
        basic_result = highlighter.highlight_keywords(large_text, query)
    basic_time = time.time() - start_time
    
    # Benchmark advanced highlighting
    start_time = time.time()
    for _ in range(10):
        advanced_result = highlighter.highlight_keywords_advanced(large_text, query)
    advanced_time = time.time() - start_time
    
    print(f"Basic highlighting (10 runs): {basic_time:.3f}s")
    print(f"Advanced highlighting (10 runs): {advanced_time:.3f}s")
    print(f"Performance ratio: {advanced_time/basic_time:.2f}x")
    
    # Benchmark chunking
    start_time = time.time()
    for _ in range(5):
        basic_chunks = highlighter.extract_relevant_chunks(
            large_text, query, use_semantic_chunking=False
        )
    basic_chunk_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(5):
        semantic_chunks = highlighter.extract_relevant_chunks(
            large_text, query, use_semantic_chunking=True
        )
    semantic_chunk_time = time.time() - start_time
    
    print(f"Basic chunking (5 runs): {basic_chunk_time:.3f}s")
    print(f"Semantic chunking (5 runs): {semantic_chunk_time:.3f}s")
    print(f"Chunking performance ratio: {semantic_chunk_time/basic_chunk_time:.2f}x")

if __name__ == "__main__":
    try:
        test_enhanced_highlighting()
        test_rag_integration()
        benchmark_performance()
        
        print("\n\n✅ All tests completed successfully!")
        print("\nKey improvements implemented:")
        print("- Enhanced keyword extraction with POS tagging")
        print("- Advanced highlighting with synonyms and related terms")
        print("- Semantic chunking based on sentence boundaries")
        print("- Contextual snippets with relevance scoring")
        print("- Performance optimizations with caching")
        print("- Multiple highlighting styles")
        print("- Phrase detection and highlighting")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()