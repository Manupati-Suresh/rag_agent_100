#!/usr/bin/env python3
"""
Test script for the enhanced RAG agent with text highlighting
"""

from rag_agent import RAGAgent
import json

def main():
    print("🔍 Testing Enhanced RAG Agent with Text Highlighting")
    print("=" * 60)
    
    # Initialize the agent
    print("1. Initializing RAG Agent...")
    agent = RAGAgent(storage_path='test_documents')
    
    # Add sample documents if none exist
    if len(agent.document_store.documents) == 0:
        print("2. Adding sample documents...")
        
        sample_docs = [
            {
                'id': 'machine_learning_guide.txt',
                'content': '''Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The core concept involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common supervised learning algorithms include linear regression, decision trees, random forests, and support vector machines. These algorithms are widely used in applications such as email spam detection, image recognition, and medical diagnosis.''',
                'metadata': {'category': 'AI', 'file_type': '.txt'}
            },
            {
                'id': 'python_programming.txt',
                'content': '''Python is a high-level, interpreted programming language that has gained immense popularity due to its simplicity and versatility. Created by Guido van Rossum in 1991, Python emphasizes code readability and allows programmers to express concepts in fewer lines of code compared to other languages. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. The language features a comprehensive standard library and a vast ecosystem of third-party packages available through PyPI (Python Package Index). Python is extensively used in web development with frameworks like Django and Flask, data science with libraries like NumPy and Pandas, machine learning with TensorFlow and scikit-learn, and automation scripting.''',
                'metadata': {'category': 'Programming', 'file_type': '.txt'}
            },
            {
                'id': 'data_science_overview.txt',
                'content': '''Data science is an interdisciplinary field that combines statistical analysis, machine learning, and domain expertise to extract meaningful insights from structured and unstructured data. The data science process typically involves several key steps: data collection, data cleaning and preprocessing, exploratory data analysis, feature engineering, model building, model evaluation, and deployment. Data scientists use various tools and programming languages, with Python and R being the most popular choices. Essential Python libraries for data science include Pandas for data manipulation, NumPy for numerical computing, Matplotlib and Seaborn for data visualization, and scikit-learn for machine learning. The field has applications across numerous industries including healthcare, finance, marketing, and technology.''',
                'metadata': {'category': 'Data Science', 'file_type': '.txt'}
            }
        ]
        
        for doc in sample_docs:
            agent.document_store.add_document(doc['id'], doc['content'], doc['metadata'])
        
        agent.save_documents()
    
    # Initialize the search index
    print("3. Building search index...")
    if not agent.is_initialized:
        agent.initialize()
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "Python programming language",
        "data science process",
        "supervised learning examples"
    ]
    
    print("\n4. Testing Enhanced Search with Highlighting:")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("=" * 40)
        
        try:
            # Test enhanced search
            results = agent.search_with_highlights(query, top_k=2, snippet_length=200)
            
            for result in results:
                print(f"\n📄 Document: {result['document_id']} (Score: {result['score']:.3f})")
                print(f"🎯 Highlighted Snippet:")
                # Remove HTML tags for console display
                snippet = result['highlighted_snippet'].replace('<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">', '[HIGHLIGHT]').replace('</mark>', '[/HIGHLIGHT]')
                print(f"   {snippet}")
                
                if result['relevant_sentences']:
                    print(f"📝 Key Sentences:")
                    for i, sentence in enumerate(result['relevant_sentences'][:2], 1):
                        clean_sentence = sentence.replace('<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">', '[HIGHLIGHT]').replace('</mark>', '[/HIGHLIGHT]')
                        print(f"   {i}. {clean_sentence}")
                
                print(f"📊 Relevance Chunks: {len(result['relevant_chunks'])}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n5. Testing Enhanced RAG Response:")
    print("-" * 40)
    
    test_query = "What are the main types of machine learning?"
    print(f"\n🤖 Query: '{test_query}'")
    
    try:
        response = agent.generate_enhanced_response(test_query, top_k=2)
        print(f"📋 Summary: {response['summary']}")
        print(f"🎯 Enhanced Context:")
        # Clean HTML for console display
        clean_context = response['context'].replace('<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">', '[HIGHLIGHT]').replace('</mark>', '[/HIGHLIGHT]')
        print(clean_context)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n✅ Enhanced RAG Agent testing completed!")

if __name__ == "__main__":
    main()