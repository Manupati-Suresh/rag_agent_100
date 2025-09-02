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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Search", "🤖 AI Chat", "📁 Browse & Add", "📚 Document Library", "🎨 Highlighting Demo", "ℹ️ Help"])
    
    with tab1:
        # Search interface
        if not st.session_state.initialized:
            st.info("Please add documents and initialize the search index first")
        else:
            st.subheader("🔍 Semantic Search")
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                query = st.text_input(
                    "Enter your search query:",
                    placeholder="e.g., 'machine learning algorithms' or 'python programming'"
                )
            with col2:
                top_k = st.selectbox("Results to show:", [3, 5, 10], index=1)
            with col3:
                snippet_length = st.selectbox("Snippet length:", [200, 300, 400], index=1)
            
            if query:
                with st.spinner("Searching documents..."):
                    try:
                        # Add highlighting options
                        with st.expander("🎨 Highlighting Options", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                exact_match_only = st.checkbox("Exact Match Only", value=True,
                                                             help="✅ RECOMMENDED: Only highlight exact query terms")
                                use_advanced = st.checkbox("Advanced Highlighting", value=False, 
                                                         help="Use NLP-based highlighting with synonyms")
                                include_context = st.checkbox("Contextual Info", value=True,
                                                            help="Include relevance scores and metadata")
                            with col2:
                                semantic_chunking = st.checkbox("Semantic Chunking", value=True,
                                                              help="Use sentence boundaries for chunking")
                                include_synonyms = st.checkbox("Include Synonyms", value=False,
                                                             help="Highlight related terms and synonyms")
                                include_related = st.checkbox("Include Related Terms", value=False,
                                                            help="Highlight semantically related words")
                        
                        # Override advanced settings if exact match is selected
                        if exact_match_only:
                            use_advanced = False
                            include_synonyms = False
                            include_related = False
                        
                        # Use enhanced search with highlighting
                        results = st.session_state.agent.search_with_highlights(
                            query, top_k, snippet_length=snippet_length,
                            use_advanced_highlighting=use_advanced,
                            include_contextual_info=include_context,
                            exact_match_only=exact_match_only,
                            include_synonyms=include_synonyms,
                            include_related=include_related
                        )
                        
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
                            # Add highlighting options
                            col1, col2 = st.columns(2)
                            with col1:
                                use_advanced = st.checkbox("Use Advanced Highlighting", value=True)
                            with col2:
                                include_context = st.checkbox("Include Contextual Info", value=True)
                            
                            response = st.session_state.agent.generate_enhanced_response(query, top_k=3)
                            
                            st.write("**Query:**", response['query'])
                            st.write("**Summary:**", response['summary'])
                            
                            st.markdown("**🎯 Context from Relevant Excerpts:**")
                            st.markdown(response['context'], unsafe_allow_html=True)
                            
                            # Show detailed breakdown with enhanced features
                            with st.expander("📊 Detailed Source Breakdown"):
                                for result in response['enhanced_results']:
                                    st.markdown(f"**{result['document_id']}** (Score: {result['score']:.3f})")
                                    
                                    # Show snippet info if available
                                    if 'snippet_info' in result and result['snippet_info']:
                                        info = result['snippet_info']
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Relevance", f"{info.get('relevance_score', 0):.3f}")
                                        with col2:
                                            st.metric("Keywords", info.get('keyword_count', 0))
                                        with col3:
                                            st.write("More content" if info.get('has_more_content', False) else "Complete")
                                    
                                    st.markdown(result['highlighted_snippet'], unsafe_allow_html=True)
                                    st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"Response generation error: {str(e)}")
    
    with tab2:
        # AI Chat Tab - Gemini Integration
        st.subheader("🤖 AI-Powered Document Chat")
        
        # Check Gemini status
        model_status = st.session_state.agent.get_model_status()
        
        if not model_status['gemini_available']:
            st.error("🚫 Gemini AI is not available")
            if not model_status['api_key_configured']:
                st.info("Please add your GOOGLE_API_KEY to the .env file")
            else:
                st.info("There was an issue initializing Gemini. Check your API key.")
            return
        
        # Show Gemini status
        with st.expander("🔧 AI Model Status", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.success("✅ Gemini Available")
                st.info(f"Model: {model_status['model_name']}")
            with col2:
                st.success("✅ API Key Configured")
        
        if not st.session_state.initialized:
            st.info("Please add documents and initialize the search index first")
        else:
            # Chat interface
            st.markdown("### 💬 Chat with Your Documents")
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat['user'])
                with st.chat_message("assistant"):
                    st.write(chat['assistant'])
            
            # Chat input
            user_message = st.chat_input("Ask a question about your documents...")
            
            if user_message:
                # Add user message to chat
                with st.chat_message("user"):
                    st.write(user_message)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        chat_response = st.session_state.agent.chat_with_documents(
                            user_message, 
                            st.session_state.chat_history[-3:] if st.session_state.chat_history else None
                        )
                        
                        if chat_response.get('success'):
                            st.write(chat_response['assistant_response'])
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'user': user_message,
                                'assistant': chat_response['assistant_response']
                            })
                            
                            # Show sources used
                            if chat_response.get('sources_used', 0) > 0:
                                st.caption(f"📚 Based on {chat_response['sources_used']} relevant documents")
                        else:
                            st.error(f"Error: {chat_response.get('error', 'Unknown error')}")
            
            # Clear chat button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.markdown("---")
            
            # Question Answering Section
            st.markdown("### ❓ Ask Specific Questions")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                question = st.text_input(
                    "Ask a specific question:",
                    placeholder="What are the main benefits of renewable energy?"
                )
            with col2:
                response_style = st.selectbox(
                    "Response Style:",
                    ["comprehensive", "concise", "analytical", "practical"]
                )
            
            if question:
                with st.spinner("Generating answer..."):
                    answer = st.session_state.agent.ask_question(
                        question, 
                        top_k=3, 
                        response_style=response_style
                    )
                    
                    if answer.get('success'):
                        st.markdown("#### 💡 Answer:")
                        st.write(answer['answer'])
                        
                        # Show sources
                        with st.expander(f"📖 Sources ({answer['source_count']} documents)", expanded=False):
                            for source in answer['sources']:
                                st.write(f"• **Document {source['rank']}**: {source['document_id']} (Score: {source['score']:.3f})")
                    else:
                        st.error(f"Error: {answer.get('error', 'Unknown error')}")
            
            st.markdown("---")
            
            # Document Summary Section
            st.markdown("### 📄 Generate Document Summaries")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                summary_topic = st.text_input(
                    "Topic for summary:",
                    placeholder="artificial intelligence applications"
                )
            with col2:
                summary_docs = st.selectbox("Documents to analyze:", [3, 5, 7, 10], index=1)
            
            if summary_topic:
                if st.button("📋 Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.agent.generate_summary(summary_topic, top_k=summary_docs)
                        
                        if summary.get('success'):
                            st.markdown("#### 📋 Summary:")
                            st.write(summary['summary'])
                            st.caption(f"📚 Based on {summary['source_documents']} documents")
                        else:
                            st.error(f"Error: {summary.get('error', 'Unknown error')}")
    
    with tab3:
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
    
    with tab4:
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
    
    with tab5:
        st.subheader("🎨 Highlighting Demo & Settings")
        
        # Demo section
        st.markdown("### Interactive Highlighting Demo")
        
        # Sample text for demo
        demo_text = st.text_area(
            "Sample text for highlighting demo:",
            value="""Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The core concept involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns in data.""",
            height=150
        )
        
        demo_query = st.text_input(
            "Enter query to highlight:",
            value="machine learning algorithms neural networks",
            placeholder="e.g., 'deep learning patterns' or 'artificial intelligence'"
        )
        
        if demo_text and demo_query:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Highlighting:**")
                try:
                    basic_highlight = st.session_state.agent.text_highlighter.highlight_keywords(demo_text, demo_query)
                    st.markdown(basic_highlight, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            with col2:
                st.markdown("**Advanced Highlighting:**")
                try:
                    advanced_highlight = st.session_state.agent.text_highlighter.highlight_keywords_advanced(
                        demo_text, demo_query, include_synonyms=True, include_related=True
                    )
                    st.markdown(advanced_highlight, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Keyword analysis
        st.markdown("---")
        st.markdown("### Keyword Analysis")
        
        if demo_query:
            try:
                # Basic keywords
                basic_keywords = st.session_state.agent.text_highlighter._extract_keywords(demo_query)
                
                # Enhanced keywords
                enhanced_keywords = st.session_state.agent.text_highlighter._extract_enhanced_keywords(demo_query)
                
                # Phrases
                phrases = st.session_state.agent.text_highlighter._extract_phrases(demo_query)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Basic Keywords:**")
                    for kw in basic_keywords:
                        st.write(f"• {kw}")
                
                with col2:
                    st.markdown("**Enhanced Keywords:**")
                    for kw in enhanced_keywords:
                        st.write(f"• {kw['word']} ({kw['importance']:.2f})")
                
                with col3:
                    st.markdown("**Detected Phrases:**")
                    for phrase in phrases:
                        st.write(f"• {phrase}")
                        
            except Exception as e:
                st.error(f"Keyword analysis error: {str(e)}")
        
        # Chunking comparison
        st.markdown("---")
        st.markdown("### Chunking Comparison")
        
        if demo_text and demo_query:
            try:
                # Basic chunking
                basic_chunks = st.session_state.agent.text_highlighter.extract_relevant_chunks(
                    demo_text, demo_query, chunk_size=150, use_semantic_chunking=False
                )
                
                # Semantic chunking
                semantic_chunks = st.session_state.agent.text_highlighter.extract_relevant_chunks(
                    demo_text, demo_query, chunk_size=150, use_semantic_chunking=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Chunking:**")
                    st.write(f"Found {len(basic_chunks)} chunks")
                    if basic_chunks:
                        st.write(f"Top chunk score: {basic_chunks[0]['relevance_score']:.3f}")
                        st.text_area("Top chunk:", basic_chunks[0]['text'], height=100)
                
                with col2:
                    st.markdown("**Semantic Chunking:**")
                    st.write(f"Found {len(semantic_chunks)} chunks")
                    if semantic_chunks:
                        st.write(f"Top chunk score: {semantic_chunks[0]['relevance_score']:.3f}")
                        st.text_area("Top chunk:", semantic_chunks[0]['text'], height=100)
                        
            except Exception as e:
                st.error(f"Chunking comparison error: {str(e)}")
        
        # Performance info
        st.markdown("---")
        st.markdown("### Performance Features")
        
        st.markdown("""
        **Enhanced Highlighting Features:**
        - 🧠 **NLP-based keyword extraction** with POS tagging
        - 🔍 **Phrase detection** for multi-word terms
        - 📊 **Importance scoring** based on linguistic analysis
        - 🎯 **Semantic chunking** using sentence boundaries
        - 🚀 **Performance caching** for repeated queries
        - 🎨 **Multiple highlighting styles** for different term types
        - 🔗 **Synonym and related term** expansion (when NLTK available)
        - 📝 **Contextual snippets** with relevance metadata
        """)
    
    with tab6:
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