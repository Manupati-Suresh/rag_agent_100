#!/usr/bin/env python3
"""
Example usage of the RAG Agent
"""

from rag_agent import RAGAgent
from document_loader import DocumentLoader
import json

def main():
    print("🔍 RAG Agent Example")
    print("=" * 50)
    
    # Initialize the agent
    print("1. Initializing RAG Agent...")
    agent = RAGAgent()
    
    # Load sample documents
    print("2. Loading sample documents...")
    agent.load_documents()  # This loads the sample documents
    
    # Initialize the search index
    print("3. Building search index...")
    agent.initialize()
    
    # Get agent statistics
    stats = agent.get_stats()
    print(f"4. Agent Stats: {json.dumps(stats, indent=2)}")
    
    # Example searches
    queries = [
        "machine learning algorithms",
        "python programming language",
        "natural language processing",
        "data analysis and statistics",
        "neural networks and deep learning"
    ]
    
    print("\n5. Performing example searches:")
    print("-" * 30)
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = agent.search(query, top_k=3)
        
        for result in results:
            print(f"  📄 Rank {result['rank']}: {result['document_id']}")
            print(f"     Similarity: {result['score']:.3f}")
            print(f"     Content: {result['content'][:100]}...")
            print()
    
    # Generate RAG response
    print("\n6. Generating RAG Response:")
    print("-" * 30)
    
    test_query = "What is machine learning?"
    response = agent.generate_response(test_query, top_k=2)
    
    print(f"Query: {response['query']}")
    print(f"Summary: {response['summary']}")
    print(f"Retrieved Documents: {len(response['retrieved_documents'])}")
    print(f"Context Preview: {response['context'][:200]}...")
    
    # Save the agent
    print("\n7. Saving agent to disk...")
    agent.save_agent("rag_agent.pkl")
    print("Agent saved successfully!")
    
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()