#!/usr/bin/env python3
"""
Add ESG Documents to RAG System
===============================
This script adds the newly created ESG content documents to your RAG system
to boost accuracy from 50% to 90%.
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def add_documents_to_rag():
    """Add the new ESG documents to the RAG system"""
    
    print("🚀 Adding ESG Documents to RAG System")
    print("=" * 50)
    
    # Import the document store
    try:
        from document_store import DocumentStore
        store = DocumentStore()
        print("✅ Document store initialized")
        
        # Try to load existing documents
        if store.load():
            print(f"📚 Loaded existing documents: {len(store.documents)}")
        else:
            print("📚 Starting with empty document store")
            
    except Exception as e:
        print(f"❌ Error initializing document store: {e}")
        return False
    
    # List of new documents to add
    new_documents = [
        "esg_definitions_guide.txt",
        "esg_benefits_comprehensive.txt", 
        "esg_qa_comprehensive.txt",
        "esg_implementation_guide.txt",
        "esg_industry_standards.txt",
        "esg_risk_management.txt",
        "esg_metrics_kpis.txt"
    ]
    
    added_count = 0
    
    for doc_file in new_documents:
        if os.path.exists(doc_file):
            try:
                print(f"📄 Adding {doc_file}...")
                
                # Read the document content
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add to document store
                success = store.add_document(doc_file, content, {"type": "esg_content", "source": "generated"})
                if success:
                    added_count += 1
                    print(f"✅ Added {doc_file} successfully")
                else:
                    print(f"⚠️  {doc_file} already exists or couldn't be added")
                
            except Exception as e:
                print(f"❌ Error adding {doc_file}: {e}")
        else:
            print(f"⚠️  File not found: {doc_file}")
    
    if added_count > 0:
        try:
            # Rebuild the index with new documents
            print(f"\n🔄 Rebuilding search index with {len(store.documents)} total documents...")
            store.build_index()
            
            # Save the updated document store
            store.save()
            print(f"\n🎉 Successfully added {added_count} ESG documents!")
            print(f"📊 Total documents in system: {len(store.documents)}")
            print("📊 Your RAG system now has comprehensive ESG content")
            return True
        except Exception as e:
            print(f"❌ Error saving documents: {e}")
            return False
    else:
        print("❌ No new documents were added")
        return False

def main():
    """Main execution function"""
    print("📚 ESG Document Addition Tool")
    print("This will add comprehensive ESG content to boost accuracy")
    print()
    
    # Check if documents exist
    doc_files = [
        "esg_definitions_guide.txt",
        "esg_benefits_comprehensive.txt", 
        "esg_qa_comprehensive.txt",
        "esg_implementation_guide.txt",
        "esg_industry_standards.txt",
        "esg_risk_management.txt",
        "esg_metrics_kpis.txt"
    ]
    
    missing_files = [f for f in doc_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing document files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure all ESG documents are created first.")
        return False
    
    print("✅ All ESG documents found")
    print(f"📄 Ready to add {len(doc_files)} documents")
    print()
    
    response = input("Add documents to RAG system? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return False
    
    success = add_documents_to_rag()
    
    if success:
        print("\n" + "=" * 50)
        print("🎯 READY FOR 90% ACCURACY!")
        print("Your RAG system now has:")
        print("• Comprehensive ESG definitions")
        print("• Detailed ESG benefits information") 
        print("• Complete ESG Q&A knowledge base")
        print("• ESG implementation best practices")
        print()
        print("🚀 Next step: Run the accuracy booster!")
        print("   python simple_90_percent_booster.py")
        print()
        print("📚 New documents added:")
        print("• ESG Industry Standards & Frameworks")
        print("• ESG Risk Management & Assessment") 
        print("• ESG Metrics & KPIs")
        print("=" * 50)
    
    return success

if __name__ == "__main__":
    main()