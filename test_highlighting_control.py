#!/usr/bin/env python3
"""
Test the new highlighting control options
"""

from text_highlighter import TextHighlighter

def test_highlighting_control():
    """Test different highlighting control options"""
    
    print("🎨 Testing Highlighting Control Options")
    print("=" * 50)
    
    highlighter = TextHighlighter()
    
    sample_text = """
    Global challenges affect many aspects of modern society. These challenges include 
    climate change, economic inequality, and social issues that impact communities worldwide.
    Companies must address these provisions and develop sustainable solutions.
    """
    
    query = "global challenges"
    
    print(f"Query: '{query}'")
    print(f"Sample text: {sample_text.strip()}")
    
    print("\n1. Basic Highlighting (exact match only):")
    basic_highlight = highlighter.highlight_keywords(sample_text, query)
    print(basic_highlight)
    
    print("\n2. Advanced Highlighting (with synonyms):")
    advanced_highlight = highlighter.highlight_keywords_advanced(
        sample_text, query, include_synonyms=True, include_related=False
    )
    print(advanced_highlight)
    
    print("\n3. Advanced Highlighting (with synonyms + related terms):")
    full_highlight = highlighter.highlight_keywords_advanced(
        sample_text, query, include_synonyms=True, include_related=True
    )
    print(full_highlight)
    
    print("\n4. Advanced Highlighting (no synonyms, no related):")
    no_expansion_highlight = highlighter.highlight_keywords_advanced(
        sample_text, query, include_synonyms=False, include_related=False
    )
    print(no_expansion_highlight)

if __name__ == "__main__":
    test_highlighting_control()