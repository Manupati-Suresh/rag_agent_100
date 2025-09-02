#!/usr/bin/env python3
"""
Comprehensive demo of enhanced highlighting features
"""

from text_highlighter import TextHighlighter
from rag_agent import RAGAgent
import time
import json

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f"📋 {title}")
    print(f"{'-'*40}")

def demo_keyword_extraction():
    """Demo enhanced keyword extraction"""
    print_section("Enhanced Keyword Extraction")
    
    highlighter = TextHighlighter()
    
    test_queries = [
        "machine learning algorithms",
        "deep neural networks for computer vision",
        "natural language processing applications",
        "supervised learning with labeled data",
        "artificial intelligence and robotics"
    ]
    
    for query in test_queries:
        print_subsection(f"Query: '{query}'")
        
        # Basic keywords
        basic_keywords = highlighter._extract_keywords(query)
        print(f"📝 Basic keywords: {basic_keywords}")
        
        # Enhanced keywords
        enhanced_keywords = highlighter._extract_enhanced_keywords(query)
        print("🧠 Enhanced keywords:")
        for kw in enhanced_keywords:
            print(f"   • {kw['word']} (importance: {kw['importance']:.2f}, POS: {kw['pos']})")
        
        # Phrases
        phrases = highlighter._extract_phrases(query)
        if phrases:
            print(f"🔍 Detected phrases: {phrases}")
        
        # Expanded keywords (if NLTK available)
        if highlighter.nltk_available:
            try:
                expanded = highlighter._expand_keywords(enhanced_keywords, True, True)
                if expanded:
                    print("🔗 Expanded terms:")
                    for exp in expanded[:3]:  # Show top 3
                        print(f"   • {exp['word']} ({exp['expansion_type']}, {exp['importance']:.2f})")
            except:
                pass

def demo_highlighting_comparison():
    """Demo highlighting comparison"""
    print_section("Highlighting Comparison")
    
    highlighter = TextHighlighter()
    
    sample_text = """
    Machine learning is a powerful subset of artificial intelligence that enables computers 
    to learn and improve from experience without being explicitly programmed. The core concept 
    involves algorithms that can identify patterns in data and make predictions or decisions 
    based on those patterns. There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning. Deep learning, a subset of machine learning, 
    uses neural networks with multiple layers to model and understand complex patterns in data.
    """
    
    queries = [
        "machine learning algorithms",
        "deep learning neural networks",
        "artificial intelligence patterns"
    ]
    
    for query in queries:
        print_subsection(f"Query: '{query}'")
        
        # Basic highlighting
        basic_highlight = highlighter.highlight_keywords(sample_text, query)
        print("📝 Basic highlighting:")
        print(basic_highlight[:300] + "...")
        
        # Advanced highlighting
        advanced_highlight = highlighter.highlight_keywords_advanced(
            sample_text, query, include_synonyms=True, include_related=True
        )
        print("\n🧠 Advanced highlighting:")
        print(advanced_highlight[:300] + "...")

def demo_chunking_comparison():
    """Demo chunking strategies"""
    print_section("Chunking Strategy Comparison")
    
    highlighter = TextHighlighter()
    
    long_text = """
    Machine learning is a powerful subset of artificial intelligence. It enables computers 
    to learn and improve from experience without being explicitly programmed. The core concept 
    involves algorithms that can identify patterns in data. These algorithms make predictions 
    or decisions based on those patterns. There are three main types of machine learning. 
    First is supervised learning, which uses labeled training data. Second is unsupervised 
    learning, which finds hidden patterns in data. Third is reinforcement learning, which 
    learns through interaction with an environment. Deep learning is a subset of machine learning. 
    It uses neural networks with multiple layers. These networks can model and understand 
    complex patterns in data. Natural language processing is another important application. 
    It focuses on the interaction between computers and human language.
    """
    
    query = "machine learning algorithms neural networks"
    
    print_subsection("Chunking Results")
    
    # Basic chunking
    start_time = time.time()
    basic_chunks = highlighter.extract_relevant_chunks(
        long_text, query, chunk_size=200, use_semantic_chunking=False
    )
    basic_time = time.time() - start_time
    
    print(f"📝 Basic chunking: {len(basic_chunks)} chunks in {basic_time:.3f}s")
    if basic_chunks:
        print(f"   Top chunk score: {basic_chunks[0]['relevance_score']:.3f}")
        print(f"   Top chunk: {basic_chunks[0]['text'][:100]}...")
    
    # Semantic chunking
    start_time = time.time()
    semantic_chunks = highlighter.extract_relevant_chunks(
        long_text, query, chunk_size=200, use_semantic_chunking=True
    )
    semantic_time = time.time() - start_time
    
    print(f"\n🧠 Semantic chunking: {len(semantic_chunks)} chunks in {semantic_time:.3f}s")
    if semantic_chunks:
        print(f"   Top chunk score: {semantic_chunks[0]['relevance_score']:.3f}")
        print(f"   Top chunk: {semantic_chunks[0]['text'][:100]}...")
        if 'sentence_count' in semantic_chunks[0]:
            print(f"   Sentences in chunk: {semantic_chunks[0]['sentence_count']}")

def demo_snippet_creation():
    """Demo snippet creation methods"""
    print_section("Snippet Creation Methods")
    
    highlighter = TextHighlighter()
    
    document_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the 
    natural intelligence displayed by humans and animals. Leading AI textbooks define the field 
    as the study of "intelligent agents": any device that perceives its environment and takes 
    actions that maximize its chance of successfully achieving its goals. Machine learning is 
    a subset of AI that provides systems the ability to automatically learn and improve from 
    experience without being explicitly programmed. Deep learning is part of a broader family 
    of machine learning methods based on artificial neural networks with representation learning. 
    Natural language processing (NLP) is a subfield of linguistics, computer science, and 
    artificial intelligence concerned with the interactions between computers and human language.
    """
    
    queries = [
        "machine learning artificial intelligence",
        "deep learning neural networks",
        "natural language processing"
    ]
    
    for query in queries:
        print_subsection(f"Query: '{query}'")
        
        # Basic snippet
        basic_snippet = highlighter.create_snippet(
            document_text, query, max_length=200, use_advanced_highlighting=False
        )
        print("📝 Basic snippet:")
        print(basic_snippet)
        
        # Advanced snippet
        advanced_snippet = highlighter.create_snippet(
            document_text, query, max_length=200, use_advanced_highlighting=True
        )
        print("\n🧠 Advanced snippet:")
        print(advanced_snippet)
        
        # Contextual snippet
        try:
            contextual_snippet = highlighter.create_contextual_snippet(
                document_text, query, max_length=200
            )
            print(f"\n🎯 Contextual snippet (relevance: {contextual_snippet['relevance_score']:.3f}):")
            print(contextual_snippet['snippet'])
            print(f"   Keywords found: {contextual_snippet['keyword_count']}")
            print(f"   Key sentences: {len(contextual_snippet['key_sentences'])}")
        except Exception as e:
            print(f"\n❌ Contextual snippet error: {str(e)}")

def demo_rag_integration():
    """Demo RAG integration with enhanced features"""
    print_section("RAG Integration Demo")
    
    # Initialize RAG agent
    agent = RAGAgent()
    
    # Load sample documents if not already loaded
    if len(agent.document_store.documents) == 0:
        print("📚 Loading sample documents...")
        agent.load_documents()
        agent.initialize()
        print(f"✅ Loaded {len(agent.document_store.documents)} documents")
    else:
        print(f"📚 Using existing {len(agent.document_store.documents)} documents")
    
    test_queries = [
        "machine learning algorithms",
        "python programming",
        "data science techniques"
    ]
    
    for query in test_queries:
        print_subsection(f"Query: '{query}'")
        
        # Basic search
        start_time = time.time()
        basic_results = agent.search(query, top_k=2)
        basic_time = time.time() - start_time
        
        print(f"📝 Basic search: {len(basic_results)} results in {basic_time:.3f}s")
        if basic_results:
            print(f"   Top result: {basic_results[0]['document_id']} (score: {basic_results[0]['score']:.3f})")
        
        # Enhanced search
        start_time = time.time()
        enhanced_results = agent.search_with_highlights(
            query, top_k=2, use_advanced_highlighting=True, include_contextual_info=True
        )
        enhanced_time = time.time() - start_time
        
        print(f"\n🧠 Enhanced search: {len(enhanced_results)} results in {enhanced_time:.3f}s")
        if enhanced_results:
            result = enhanced_results[0]
            print(f"   Top result: {result['document_id']} (score: {result['score']:.3f})")
            print(f"   Snippet preview: {result['highlighted_snippet'][:100]}...")
            if 'snippet_info' in result and result['snippet_info']:
                info = result['snippet_info']
                print(f"   Relevance: {info.get('relevance_score', 0):.3f}")
                print(f"   Keywords: {info.get('keyword_count', 0)}")

def demo_performance_features():
    """Demo performance and caching features"""
    print_section("Performance Features Demo")
    
    highlighter = TextHighlighter()
    
    # Test caching
    query = "machine learning deep learning artificial intelligence"
    
    print_subsection("Caching Performance")
    
    # First call (no cache)
    start_time = time.time()
    keywords1 = highlighter._extract_enhanced_keywords(query)
    first_time = time.time() - start_time
    
    # Second call (cached)
    start_time = time.time()
    keywords2 = highlighter._extract_enhanced_keywords(query)
    cached_time = time.time() - start_time
    
    print(f"📝 First call: {first_time:.4f}s")
    print(f"🚀 Cached call: {cached_time:.4f}s")
    print(f"⚡ Speedup: {first_time/cached_time:.1f}x faster")
    
    # Test batch processing
    print_subsection("Batch Processing")
    
    queries = [
        "machine learning algorithms",
        "deep neural networks",
        "natural language processing",
        "computer vision applications",
        "reinforcement learning methods"
    ]
    
    sample_text = "Machine learning and artificial intelligence are transforming technology."
    
    start_time = time.time()
    for query in queries:
        highlighter.highlight_keywords_advanced(sample_text, query)
    batch_time = time.time() - start_time
    
    print(f"🔄 Processed {len(queries)} queries in {batch_time:.3f}s")
    print(f"⚡ Average per query: {batch_time/len(queries):.3f}s")

def main():
    """Run all demos"""
    print("🎉 Enhanced RAG Highlighting System Demo")
    print("This demo showcases all the new highlighting improvements")
    
    try:
        demo_keyword_extraction()
        demo_highlighting_comparison()
        demo_chunking_comparison()
        demo_snippet_creation()
        demo_rag_integration()
        demo_performance_features()
        
        print_section("Summary of Improvements")
        print("""
✅ Enhanced Keyword Extraction:
   • POS tagging for importance scoring
   • Lemmatization and linguistic analysis
   • Phrase detection for multi-word terms

✅ Advanced Highlighting:
   • Multiple highlighting styles
   • Synonym and related term expansion
   • Context-aware highlighting

✅ Intelligent Chunking:
   • Semantic boundaries (sentences/paragraphs)
   • Enhanced relevance scoring
   • Better context preservation

✅ Smart Snippet Creation:
   • Contextual snippets with metadata
   • Relevance-based selection
   • Smart trimming with highlight preservation

✅ Performance Optimizations:
   • LRU caching for expensive operations
   • Batch processing capabilities
   • Efficient text processing

✅ RAG Integration:
   • Enhanced search results
   • Contextual information
   • Configurable highlighting options
        """)
        
        print("\n🎯 The enhanced highlighting system provides:")
        print("   • More accurate keyword identification")
        print("   • Better visual emphasis of relevant content")
        print("   • Improved user experience with contextual information")
        print("   • Faster performance through intelligent caching")
        print("   • Flexible configuration options")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()