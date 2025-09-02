#!/usr/bin/env python3
"""
Command-line document manager for the RAG agent
"""

import argparse
import os
import sys
from rag_agent import RAGAgent
from document_loader import DocumentLoader
import json

def list_documents(agent):
    """List all stored documents"""
    documents = agent.get_document_list()
    stats = agent.get_stats()
    
    print(f"\n📚 Document Library ({stats['total_documents']}/{stats['max_documents']})")
    print("=" * 60)
    
    if not documents:
        print("No documents stored.")
        return
    
    for i, doc in enumerate(documents, 1):
        print(f"{i:2d}. {doc['id']}")
        print(f"    Added: {doc['added_date']}")
        print(f"    Type: {doc['file_type']}")
        print(f"    Size: {doc['file_size']} bytes")
        print(f"    Preview: {doc['content_preview']}")
        print()

def add_documents(agent, file_paths):
    """Add documents from file paths"""
    print(f"\n📁 Adding {len(file_paths)} documents...")
    
    results = agent.add_documents_from_files(file_paths)
    
    print(f"✅ Added: {results['added']}")
    print(f"⏭️  Skipped: {results['skipped']}")
    print(f"❌ Errors: {results['errors']}")
    
    if results['messages']:
        print("\nDetails:")
        for msg in results['messages']:
            print(f"  {msg}")

def search_documents(agent, query, top_k=5):
    """Search documents"""
    if not agent.is_initialized:
        print("Initializing search index...")
        agent.initialize()
    
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 60)
    
    results = agent.search(query, top_k)
    
    if not results:
        print("No relevant documents found.")
        return
    
    for result in results:
        print(f"📄 Rank {result['rank']}: {result['document_id']}")
        print(f"   Similarity: {result['score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        print()

def browse_directory(directory):
    """Browse directory for supported files"""
    print(f"\n📁 Browsing: {directory}")
    print("=" * 60)
    
    try:
        files = DocumentLoader.get_supported_files(directory)
        
        if not files:
            print("No supported files found.")
            return
        
        print(f"Found {len(files)} supported files:")
        
        for i, file_info in enumerate(files, 1):
            print(f"{i:3d}. {file_info['filename']}")
            print(f"     Path: {file_info['relative_path']}")
            print(f"     Size: {file_info['size_mb']} MB")
            print(f"     Type: {file_info['extension']}")
            print()
            
    except Exception as e:
        print(f"Error browsing directory: {e}")

def remove_document(agent, doc_id):
    """Remove a document"""
    if agent.remove_document(doc_id):
        print(f"✅ Removed document: {doc_id}")
    else:
        print(f"❌ Document not found: {doc_id}")

def clear_all(agent):
    """Clear all documents"""
    stats = agent.get_stats()
    if stats['total_documents'] == 0:
        print("No documents to clear.")
        return
    
    confirm = input(f"Are you sure you want to delete all {stats['total_documents']} documents? (y/N): ")
    if confirm.lower() == 'y':
        agent.clear_all_documents()
        print("✅ All documents cleared.")
    else:
        print("Operation cancelled.")

def main():
    parser = argparse.ArgumentParser(description="RAG Document Manager")
    parser.add_argument("--storage", default="document_storage", help="Storage directory path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    subparsers.add_parser("list", help="List all stored documents")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add documents from files")
    add_parser.add_argument("files", nargs="+", help="File paths to add")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    # Browse command
    browse_parser = subparsers.add_parser("browse", help="Browse directory for files")
    browse_parser.add_argument("directory", help="Directory to browse")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a document")
    remove_parser.add_argument("doc_id", help="Document ID to remove")
    
    # Clear command
    subparsers.add_parser("clear", help="Clear all documents")
    
    # Stats command
    subparsers.add_parser("stats", help="Show storage statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize agent
    print(f"🤖 RAG Document Manager")
    print(f"📁 Storage: {args.storage}")
    
    agent = RAGAgent(storage_path=args.storage)
    
    # Execute command
    if args.command == "list":
        list_documents(agent)
        
    elif args.command == "add":
        # Validate files exist
        valid_files = []
        for file_path in args.files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                print(f"❌ File not found: {file_path}")
        
        if valid_files:
            add_documents(agent, valid_files)
            
    elif args.command == "search":
        search_documents(agent, args.query, args.top_k)
        
    elif args.command == "browse":
        browse_directory(args.directory)
        
    elif args.command == "remove":
        remove_document(agent, args.doc_id)
        
    elif args.command == "clear":
        clear_all(agent)
        
    elif args.command == "stats":
        stats = agent.get_stats()
        print("\n📊 Storage Statistics:")
        print("=" * 30)
        print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()