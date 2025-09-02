#!/usr/bin/env python3
"""
Enhanced example demonstrating persistent document storage and file browsing
"""

from rag_agent import RAGAgent
from document_loader import DocumentLoader
import json
import os

def main():
    print("🔍 RAG Agent - Persistent Storage Example")
    print("=" * 60)
    
    # Initialize the agent with persistent storage
    print("1. Initializing RAG Agent with persistent storage...")
    agent = RAGAgent(storage_path='my_documents')
    
    # Check if we have existing documents
    stats = agent.get_stats()
    print(f"   Current storage: {stats['total_documents']}/{stats['max_documents']} documents")
    
    if stats['total_documents'] == 0:
        print("\n2. No existing documents found. Adding sample documents...")
        
        # Create some sample documents for demonstration
        sample_docs = [
            {
                'id': 'ai_overview.txt',
                'content': 'Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. It includes machine learning, deep learning, natural language processing, and computer vision.',
                'metadata': {'category': 'AI', 'file_type': '.txt'}
            },
            {
                'id': 'python_guide.txt', 
                'content': 'Python is a high-level, interpreted programming language known for its simplicity and readability. It is widely used in web development, data science, artificial intelligence, and automation.',
                'metadata': {'category': 'Programming', 'file_type': '.txt'}
            },
            {
                'id': 'data_science.txt',
                'content': 'Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.',
                'metadata': {'category': 'Data Science', 'file_type': '.txt'}
            }
        ]
        
        # Add sample documents
        for doc in sample_docs:
            agent.document_store.add_document(doc['id'], doc['content'], doc['metadata'])
        
        # Save documents
        agent.save_documents()
        print(f"   Added {len(sample_docs)} sample documents")
    else:
        print(f"\n2. Found {stats['total_documents']} existing documents in storage")
    
    # Initialize search index
    print("\n3. Initializing search index...")
    if not agent.is_initialized:
        agent.initialize()
    
    # Display document library
    print("\n4. Document Library:")
    print("-" * 40)
    documents = agent.get_document_list()
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc['id']}")
        print(f"      Added: {doc['added_date']}")
        print(f"      Preview: {doc['content_preview']}")
        print()
    
    # Demonstrate file browsing
    print("5. File Browser Demonstration:")
    print("-" * 40)
    
    # Browse current directory for supported files
    current_dir = os.getcwd()
    print(f"   Browsing: {current_dir}")
    
    available_files = DocumentLoader.get_supported_files(current_dir)
    print(f"   Found {len(available_files)} supported files:")
    
    for file_info in available_files[:5]:  # Show first 5 files
        print(f"   - {file_info['filename']} ({file_info['extension']}, {file_info['size_mb']} MB)")
    
    if len(available_files) > 5:
        print(f"   ... and {len(available_files) - 5} more files")
    
    # Demonstrate search functionality
    print("\n6. Search Demonstrations:")
    print("-" * 40)
    
    search_queries = [
        "artificial intelligence and machine learning",
        "python programming language",
        "data analysis and statistics"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Query: '{query}'")
        results = agent.search(query, top_k=2)
        
        for result in results:
            print(f"   📄 {result['document_id']} (Score: {result['score']:.3f})")
            print(f"      {result['content'][:80]}...")
    
    # Demonstrate adding new documents from files
    print("\n7. Adding Documents from Files:")
    print("-" * 40)
    
    # Check if we can add more documents
    remaining_slots = stats['max_documents'] - len(agent.document_store.documents)
    print(f"   Remaining document slots: {remaining_slots}")
    
    if remaining_slots > 0 and available_files:
        print("   You can add more documents using:")
        print("   agent.add_documents_from_files(['/path/to/file1.txt', '/path/to/file2.pdf'])")
        
        # Example of adding a file (if README.md exists)
        readme_files = [f for f in available_files if f['filename'].lower() == 'readme.md']
        if readme_files:
            print(f"\n   Example: Adding README.md...")
            results = agent.add_documents_from_files([readme_files[0]['filepath']])
            print(f"   Result: Added {results['added']}, Skipped {results['skipped']}, Errors {results['errors']}")
    
    # Show final statistics
    print("\n8. Final Statistics:")
    print("-" * 40)
    final_stats = agent.get_stats()
    print(json.dumps(final_stats, indent=2))
    
    # Demonstrate RAG response
    print("\n9. RAG Response Example:")
    print("-" * 40)
    
    test_query = "What is artificial intelligence?"
    print(f"Query: {test_query}")
    
    response = agent.generate_response(test_query, top_k=2)
    print(f"Summary: {response['summary']}")
    print(f"Context: {response['context'][:200]}...")
    
    print("\n✅ Persistent storage example completed!")
    print(f"📁 Documents are saved in: {agent.storage_path}")
    print("🔄 Run this script again to see persistent storage in action!")

if __name__ == "__main__":
    main()