#!/usr/bin/env python3
"""
Add Remaining PDF Document
==========================
This script adds the encrypted esg-brochure.pdf that couldn't be added before.
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def add_remaining_pdf():
    """Add the remaining encrypted PDF document"""
    
    print("🚀 Adding Remaining PDF Document")
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
    
    # The remaining PDF document
    pdf_file = "esg-brochure.pdf"
    
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
                    print(f"✅ Added {pdf_file} successfully")
                    print(f"   Content length: {len(content)} characters")
                    
                    # Rebuild the index with new document
                    print(f"\n🔄 Rebuilding search index with {len(store.documents)} total documents...")
                    store.build_index()
                    
                    # Save the updated document store
                    store.save()
                    print(f"\n🎉 Successfully added {pdf_file}!")
                    print(f"📊 Total documents in system: {len(store.documents)}")
                    print("📊 Your RAG system is now complete with all PDF documents")
                    return True
                else:
                    print(f"⚠️  {pdf_file} already exists or couldn't be added")
                    return False
            else:
                print(f"⚠️  {pdf_file} appears to be empty or unreadable")
                return False
            
        except Exception as e:
            print(f"❌ Error adding {pdf_file}: {e}")
            return False
    else:
        print(f"⚠️  File not found: {pdf_file}")
        return False

def main():
    """Main execution function"""
    print("📚 Remaining PDF Document Addition")
    print("Adding the encrypted esg-brochure.pdf with PyCryptodome support")
    print()
    
    # Check if the PDF document exists
    pdf_file = "esg-brochure.pdf"
    
    if os.path.exists(pdf_file):
        size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
        print(f"✅ Found: {pdf_file} ({size_mb:.1f} MB)")
    else:
        print(f"❌ File not found: {pdf_file}")
        return False
    
    print()
    response = input("Add the remaining PDF document? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return False
    
    success = add_remaining_pdf()
    
    if success:
        print("\n" + "=" * 50)
        print("🎯 COMPLETE RAG SYSTEM!")
        print("Your RAG system now includes ALL documents:")
        print("• Original ESG handbook and guide documents")
        print("• Generated comprehensive ESG content")
        print("• All 5 professional ESG PDF documents")
        print()
        print("🚀 Ready to test maximum accuracy!")
        print("   python simple_90_percent_booster.py")
        print("=" * 50)
    
    return success

if __name__ == "__main__":
    main()