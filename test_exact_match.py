#!/usr/bin/env python3
"""
Test exact match highlighting
"""

from text_highlighter import TextHighlighter

def test_exact_match():
    """Test exact match highlighting"""
    
    print("🎯 Testing Exact Match Highlighting")
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
    
    print("\n1. Basic Keywords (should be only 'global' and 'challenges'):")
    basic_keywords = highlighter._extract_keywords(query)
    print(basic_keywords)
    
    print("\n2. Exact Match Highlighting (should highlight ONLY 'global' and 'challenges'):")
    exact_highlight = highlighter.highlight_keywords(sample_text, query)
    print(exact_highlight)
    
    print("\n3. Enhanced Highlighting (for comparison - may highlight more words):")
    enhanced_highlight = highlighter.highlight_keywords_enhanced(sample_text, query)
    print(enhanced_highlight)

if __name__ == "__main__":
    test_exact_match()