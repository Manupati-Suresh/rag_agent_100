#!/usr/bin/env python3
"""
Quick test of enhanced highlighting features
"""

from text_highlighter import TextHighlighter

def test_basic_functionality():
    """Test basic functionality without complex features"""
    
    print("🧪 Quick Test of Enhanced Highlighting")
    print("=" * 50)
    
    highlighter = TextHighlighter()
    
    sample_text = """
    Machine learning is a powerful subset of artificial intelligence that enables computers 
    to learn and improve from experience. The core concept involves algorithms that can 
    identify patterns in data and make predictions based on those patterns.
    """
    
    query = "machine learning algorithms"
    
    print(f"Query: '{query}'")
    print(f"NLTK Available: {highlighter.nltk_available}")
    
    # Test basic keyword extraction
    print("\n1. Basic Keywords:")
    basic_keywords = highlighter._extract_keywords(query)
    print(basic_keywords)
    
    # Test enhanced keyword extraction
    print("\n2. Enhanced Keywords:")
    try:
        enhanced_keywords = highlighter._extract_enhanced_keywords(query)
        for kw in enhanced_keywords:
            print(f"   {kw['word']} (importance: {kw['importance']:.2f})")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test basic highlighting
    print("\n3. Basic Highlighting:")
    try:
        basic_highlight = highlighter.highlight_keywords(sample_text, query)
        print(basic_highlight[:200] + "...")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test advanced highlighting
    print("\n4. Advanced Highlighting:")
    try:
        advanced_highlight = highlighter.highlight_keywords_advanced(sample_text, query)
        print(advanced_highlight[:200] + "...")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    # Test chunking
    print("\n5. Chunking:")
    try:
        chunks = highlighter.extract_relevant_chunks(sample_text, query, chunk_size=100)
        print(f"   Found {len(chunks)} chunks")
        if chunks:
            print(f"   Top chunk score: {chunks[0]['relevance_score']:.3f}")
    except Exception as e:
        print(f"   Error: {str(e)}")
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    test_basic_functionality()