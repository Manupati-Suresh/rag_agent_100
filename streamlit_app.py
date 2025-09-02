import streamlit as st
import json
from rag_agent import RAGAgent
from document_loader import DocumentLoader
import os
from pathlib import Path
import pandas as pd

# Page config
st.set_page_config(
    page_title="RAG Document Search Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = RAGAgent()
if 'initialized' not in st.session_state:
    st.session_state.initialized = st.session_state.agent.is_initialized
if 'selected_directory' not in st.session_state:
    st.session_state.selected_directory = None
if 'available_files' not in st.session_state:
    st.session_state.available_files = []

def initialize_agent():
    """Initialize the RAG agent"""
    if not st.session_state.initialized and len(st.session_state.agent.document_store.documents) > 0:
        with st.spinner("Building search index..."):
            st.session_state.agent.initialize()
            st.session_state.initialized = True
            st.success("RAG Agent initialized!")
    elif len(st.session_state.agent.document_store.documents) == 0:
        st.warning("No documents loaded. Please add documents first.")

def browse_directory():
    """Browse directory for files"""
    directory = st.text_input("Enter directory path:", value=os.getcwd())
    
    if st.button("Browse Directory"):
        if os.path.exists(directory):
            try:
                files = DocumentLoader.get_supported_files(directory)
                st.session_state.available_files = files
                st.session_state.selected_directory = directory
                st.success(f"Found {len(files)} supported files")
            except Exception as e:
                st.error(f"Error browsing directory: {str(e)}")
        else:
            st.error("Directory does not exist")

def display_file_browser():
    """Display file browser and selection interface"""
    if st.session_state.available_files:
        st.subheader("📁 Available Files")
        
        # Create DataFrame for better display
        df_files = pd.DataFrame(st.session_state.available_files)
        df_files['select'] = False
        
        # File selection interface
        selected_files = []
        
        with st.expander(f"Select files to add (Max: {st.session_state.agent.document_store.max_documents - len(st.session_state.agent.document_store.documents)} remaining)", expanded=True):
            for idx, file_info in enumerate(st.session_state.available_files):
                col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
                
                with col1:
                    if st.checkbox("", key=f"file_{idx}"):
                        selected_files.append(file_info['filepath'])
                
                with col2:
                    st.write(f"**{file_info['filename']}**")
                    st.caption(file_info['relative_path'])
                
                with col3:
                    st.write(f"{file_info['size_mb']} MB")
                
                with col4:
                    st.write(file_info['extension'])
        
        # Add selected files
        if selected_files:
            if st.button(f"Add {len(selected_files)} Selected Files"):
                with st.spinner("Adding documents..."):
                    results = st.session_state.agent.add_documents_from_files(selected_files)
                    
                    if results['added'] > 0:
                        st.success(f"Added {results['added']} documents!")
                        st.session_state.initialized = False  # Need to rebuild index
                        
                    if results['skipped'] > 0:
                        st.warning(f"Skipped {results['skipped']} duplicate documents")
                        
                    if results['errors'] > 0:
                        st.error(f"Failed to add {results['errors']} documents")
                    
                    # Show detailed messages
                    with st.expander("Detailed Results"):
                        for msg in results['messages']:
                            st.write(msg)

def main():
    st.title("🔍 RAG Document Search Agent")
    st.markdown("**Permanent Storage for 100 Documents with Semantic Search**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Document Storage")
        
        # Agent stats
        stats = st.session_state.agent.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats['total_documents'])
        with col2:
            st.metric("Remaining", stats['remaining_slots'])
        
        # Progress bar
        progress = stats['total_documents'] / stats['max_documents']
        st.progress(progress)
        st.caption(f"{stats['total_documents']}/{stats['max_documents']} documents stored")
        
        st.markdown("---")
        
        # Initialize/Rebuild Index
        if not st.session_state.initialized and stats['total_documents'] > 0:
            if st.button("🚀 Initialize Search Index", type="primary"):
                initialize_agent()
        elif st.session_state.initialized:
            st.success("✅ Search Ready")
            if st.button("🔄 Rebuild Index"):
                initialize_agent()
        
        st.markdown("---")
        
        # Document Management
        st.header("📁 Document Management")
        
        # Clear all documents
        if stats['total_documents'] > 0:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    st.session_state.agent.clear_all_documents()
                    st.session_state.initialized = False
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm deletion")
        
        # Storage info
        st.markdown("---")
        st.caption(f"Storage: {stats['storage_path']}")
        st.caption("Documents are automatically saved")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📁 Browse & Add", "📚 Document Library", "ℹ️ Help"])
    
    with tab1:
        # Search interface
        if not st.session_state.initialized:
            st.info("Please add documents and initialize the search index first")
        else:
            st.subheader("🔍 Semantic Search")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input(
                    "Enter your search query:",
                    placeholder="e.g., 'machine learning algorithms' or 'python programming'"
                )
            with col2:
                top_k = st.selectbox("Results to show:", [3, 5, 10], index=1)
            
            if query:
                with st.spinner("Searching documents..."):
                    try:
                        # Use enhanced search with highlighting
                        results = st.session_state.agent.search_with_highlights(query, top_k, snippet_length=400)
                        
                        if results:
                            st.success(f"Found {len(results)} relevant document excerpts")
                            
                            # Display enhanced results
                            for result in results:
                                with st.expander(
                                    f"📄 Rank {result['rank']}: {result['document_id']} "
                                    f"(Similarity: {result['score']:.3f})"
                                ):
                                    # Show highlighted snippet
                                    st.markdown("**🎯 Relevant Excerpt:**")
                                    st.markdown(result['highlighted_snippet'], unsafe_allow_html=True)
                                    
                                    # Show relevant sentences if available
                                    if result['relevant_sentences']:
                                        st.markdown("**📝 Key Sentences:**")
                                        for i, sentence in enumerate(result['relevant_sentences'], 1):
                                            highlighted_sentence = st.session_state.agent.text_highlighter.highlight_keywords(sentence, query)
                                            st.markdown(f"{i}. {highlighted_sentence}", unsafe_allow_html=True)
                                    
                                    # Show relevant chunks info
                                    if result['relevant_chunks']:
                                        st.markdown("**📊 Relevance Scores:**")
                                        for i, chunk in enumerate(result['relevant_chunks'], 1):
                                            st.write(f"Chunk {i}: Relevance {chunk['relevance_score']:.3f}")
                                    
                                    # Metadata
                                    if result['metadata']:
                                        st.markdown("**📁 File Info:**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"**File:** {result['metadata'].get('filename', 'N/A')}")
                                        with col2:
                                            st.write(f"**Type:** {result['metadata'].get('file_type', 'N/A')}")
                                    
                                    # Option to view full document
                                    if st.button(f"View Full Document", key=f"full_{result['rank']}"):
                                        st.text_area("Full Content", result['full_content'], height=200)
                        else:
                            st.warning("No relevant documents found")
                            
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                
                # RAG Response
                st.markdown("---")
                st.subheader("🤖 RAG Response")
                
                if st.button("Generate Enhanced RAG Response"):
                    with st.spinner("Generating enhanced response..."):
                        try:
                            response = st.session_state.agent.generate_enhanced_response(query, top_k=3)
                            
                            st.write("**Query:**", response['query'])
                            st.write("**Summary:**", response['summary'])
                            
                            st.markdown("**🎯 Context from Relevant Excerpts:**")
                            st.markdown(response['context'], unsafe_allow_html=True)
                            
                            # Show detailed breakdown
                            with st.expander("📊 Detailed Source Breakdown"):
                                for result in response['enhanced_results']:
                                    st.markdown(f"**{result['document_id']}** (Score: {result['score']:.3f})")
                                    st.markdown(result['highlighted_snippet'], unsafe_allow_html=True)
                                    st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"Response generation error: {str(e)}")
    
    with tab2:
        st.subheader("📁 Browse and Add Documents")
        
        # Directory browser
        browse_directory()
        
        # File selection interface
        display_file_browser()
        
        # Upload interface
        st.markdown("---")
        st.subheader("📤 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload documents directly",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'json', 'md']
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                with st.spinner("Processing uploaded files..."):
                    # Save uploaded files temporarily and process
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(temp_path)
                    
                    results = st.session_state.agent.add_documents_from_files(file_paths)
                    
                    # Clean up temp files
                    for temp_path in file_paths:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    
                    if results['added'] > 0:
                        st.success(f"Added {results['added']} documents!")
                        st.session_state.initialized = False
                    
                    if results['errors'] > 0:
                        st.error(f"Failed to process {results['errors']} files")
    
    with tab3:
        st.subheader("📚 Document Library")
        
        if stats['total_documents'] > 0:
            documents = st.session_state.agent.get_document_list()
            
            # Display documents in a table
            df = pd.DataFrame(documents)
            st.dataframe(df, use_container_width=True)
            
            # Individual document management
            st.markdown("---")
            st.subheader("Document Details")
            
            selected_doc = st.selectbox(
                "Select document to view/remove:",
                options=[doc['id'] for doc in documents],
                index=0 if documents else None
            )
            
            if selected_doc:
                doc_info = next(doc for doc in documents if doc['id'] == selected_doc)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**File:** {selected_doc}")
                    st.write(f"**Added:** {doc_info['added_date']}")
                    st.write(f"**Type:** {doc_info['file_type']}")
                    st.write(f"**Size:** {doc_info['file_size']} bytes")
                    
                with col2:
                    if st.button("🗑️ Remove Document"):
                        if st.session_state.agent.remove_document(selected_doc):
                            st.success(f"Removed {selected_doc}")
                            st.rerun()
                        else:
                            st.error("Failed to remove document")
                
                st.write("**Content Preview:**")
                st.text_area("", doc_info['content_preview'], height=150, disabled=True)
        else:
            st.info("No documents stored yet. Use the 'Browse & Add' tab to add documents.")
    
    with tab4:
        st.subheader("ℹ️ Help & Information")
        
        st.markdown("""
        ### How to Use the RAG Document Search Agent
        
        1. **Add Documents**: Use the 'Browse & Add' tab to:
           - Browse your computer for supported files (TXT, PDF, DOCX, JSON, MD)
           - Select up to 100 documents to store permanently
           - Upload files directly through the interface
        
        2. **Initialize Search**: Once documents are added:
           - Click 'Initialize Search Index' in the sidebar
           - This builds the semantic search index
        
        3. **Search Documents**: Use the 'Search' tab to:
           - Enter natural language queries
           - Get semantically similar documents ranked by relevance
           - Generate RAG responses using retrieved context
        
        4. **Manage Documents**: Use the 'Document Library' tab to:
           - View all stored documents
           - Remove individual documents
           - See document metadata and previews
        
        ### Features
        - **Permanent Storage**: Documents are saved automatically
        - **Semantic Search**: Uses AI embeddings for intelligent search
        - **100 Document Limit**: Optimized for performance
        - **Multiple Formats**: Supports various document types
        - **Duplicate Detection**: Prevents storing identical content
        
        ### Tips
        - Use descriptive, specific search queries for better results
        - Documents are automatically saved to the `document_storage` folder
        - The search index is rebuilt automatically when needed
        - Clear all documents if you want to start fresh
        """)
        
        # System info
        st.markdown("---")
        st.subheader("System Information")
        st.json(stats)

if __name__ == "__main__":
    main()