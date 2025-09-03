#!/usr/bin/env python3
"""
Add PDF Documents to RAG System
===============================
This script adds the newly added PDF documents to your RAG system
for even better accuracy and comprehensive ESG knowledge.
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def add_pdf_documents_to_rag():
    """Add the new PDF documents to the RAG system"""
    
    print("🚀 Adding PDF Documents to RAG System")
    print("=" * 50)
    
    # Import the document store and loader
    try:
        from document_store import DocumentStore
        from document_loader import DocumentLoader
        
        store = DocumentStore()
        loader = DocumentLoader()
        print("✅ Document store and loader initialized")
        
        # Try to load existing documents
        if store.load():
            print(f"📚 Loaded existing documents: {len(store.documents)}")
        else:
            print("📚 Starting with empty document store")
            
    except Exception as e:
        print(f"❌ Error initializing document store: {e}")
        return False
    
    # List of new PDF documents to add
    pdf_documents = [
        "esg-brochure.pdf",
        "ESG-Document-online-to-Mastering-ESG.pdf",
        "ifc-esg-guidebook.pdf",
        "Integrating ESG into your strategy formulation- a practical guide for sustainable success.pdf",
        "The-Global-ESG-Handbook.pdf"
    ]
    
    added_count = 0
    
    for pdf_file in pdf_documents:
        if os.path.exists(pdf_file):
            try:
                print(f"📄 Adding {pdf_file}...")
                
                # Load PDF content using DocumentLoader
                content = loader.load_pdf_file(pdf_file)
                
                if content and content.strip():
                    # Add to document store
                    success = store.add_document(
                        pdf_file, 
                        content, 
                        {
                            "type": "esg_pdf", 
                            "source": "external_pdf",
                            "file_type": "pdf",
                            "file_size": os.path.getsize(pdf_file)
                        }
                    )
                    if success:
                        added_count += 1
                        print(f"✅ Added {pdf_file} successfully")
                        print(f"   Content length: {len(content)} characters")
                    else:
                        print(f"⚠️  {pdf_file} already exists or couldn't be added")
                else:
                    print(f"⚠️  {pdf_file} appears to be empty or unreadable")
                
            except Exception as e:
                print(f"❌ Error adding {pdf_file}: {e}")
        else:
            print(f"⚠️  File not found: {pdf_file}")
    
    if added_count > 0:
        try:
            # Rebuild the index with new documents
            print(f"\n🔄 Rebuilding search index with {len(store.documents)} total documents...")
            store.build_index()
            
            # Save the updated document store
            store.save()
            print(f"\n🎉 Successfully added {added_count} PDF documents!")
            print(f"📊 Total documents in system: {len(store.documents)}")
            print("📊 Your RAG system now has even more comprehensive ESG content")
            return True
        except Exception as e:
            print(f"❌ Error saving documents: {e}")
            return False
    else:
        print("❌ No new PDF documents were added")
        return False

def main():
    """Main execution function"""
    print("📚 PDF Document Addition Tool")
    print("This will add your 5 new PDF documents to boost accuracy even further")
    print()
    
    # Check if PDF documents exist
    pdf_files = [
        "esg-brochure.pdf",
        "ESG-Document-online-to-Mastering-ESG.pdf",
        "ifc-esg-guidebook.pdf",
        "Integrating ESG into your strategy formulation- a practical guide for sustainable success.pdf",
        "The-Global-ESG-Handbook.pdf"
    ]
    
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    missing_files = [f for f in pdf_files if not os.path.exists(f)]
    
    if existing_files:
        print(f"✅ Found {len(existing_files)} PDF documents:")
        for f in existing_files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"   - {f} ({size_mb:.1f} MB)")
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} PDF documents:")
        for f in missing_files:
            print(f"   - {f}")
    
    if not existing_files:
        print("\n❌ No PDF documents found to add.")
        return False
    
    print(f"\n📄 Ready to add {len(existing_files)} PDF documents")
    print()
    
    response = input("Add PDF documents to RAG system? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return False
    
    success = add_pdf_documents_to_rag()
    
    if success:
        print("\n" + "=" * 50)
        print("🎯 ENHANCED RAG SYSTEM!")
        print("Your RAG system now includes:")
        print("• Original ESG handbook and guide documents")
        print("• Generated comprehensive ESG content")
        print("• 5 additional professional ESG PDF documents")
        print()
        print("📚 New PDF documents added:")
        for f in existing_files:
            print(f"• {f}")
        print()
        print("🚀 Next step: Test the enhanced accuracy!")
        print("   python simple_90_percent_booster.py")
        print("=" * 50)
    
    return success

if __name__ == "__main__":
    main()