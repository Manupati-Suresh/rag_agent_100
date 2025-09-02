#!/usr/bin/env python3
"""
Streamlit App with Telegram Document Storage
Enhanced RAG interface using Telegram as document storage backend
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
import os
from typing import Dict, List

# Import our components
from rag_agent_telegram import TelegramRAGAgent, create_telegram_rag_agent
from telegram_config import TelegramConfig

# Page configuration
st.set_page_config(
    page_title="RAG Agent with Telegram Storage",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .storage-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .document-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #ffeb3b;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'telegram_initialized' not in st.session_state:
    st.session_state.telegram_initialized = False

async def initialize_agent():
    """Initialize the RAG agent with Telegram storage"""
    if st.session_state.agent is None:
        with st.spinner("🔄 Initializing RAG Agent with Telegram storage..."):
            try:
                agent = await create_telegram_rag_agent(use_telegram=True)
                st.session_state.agent = agent
                st.session_state.telegram_initialized = True
                st.success("✅ RAG Agent with Telegram storage initialized!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {e}")
                # Try fallback to local storage
                try:
                    agent = await create_telegram_rag_agent(use_telegram=False)
                    st.session_state.agent = agent
                    st.session_state.telegram_initialized = False
                    st.warning("⚠️ Using local storage as fallback")
                    return True
                except Exception as e2:
                    st.error(f"❌ Complete initialization failure: {e2}")
                    return False
    return True

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Agent with Telegram Storage</h1>', unsafe_allow_html=True)
    
    # Sidebar - Configuration and Stats
    with st.sidebar:
        st.header("📊 Configuration & Stats")
        
        # Telegram Configuration Status
        st.subheader("🔧 Telegram Setup")
        try:
            TelegramConfig.validate_config()
            st.success("✅ Telegram configured")
            
            if st.session_state.telegram_initialized:
                st.info("📱 Using Telegram storage")
            else:
                st.warning("📁 Using local storage")
                
        except ValueError as e:
            st.error("❌ Telegram not configured")
            with st.expander("Setup Instructions"):
                st.code(TelegramConfig.get_setup_instructions())
        
        # Initialize agent button
        if st.button("🚀 Initialize/Refresh Agent"):
            st.session_state.agent = None
            st.rerun()
        
        # Storage stats
        if st.session_state.agent:
            st.subheader("📈 Storage Statistics")
            
            # Get stats asynchronously
            async def get_stats():
                return await st.session_state.agent.get_storage_stats()
            
            try:
                stats = asyncio.run(get_stats())
                
                if 'telegram' in stats:
                    st.markdown("**Telegram Storage:**")
                    st.write(f"📄 Documents: {stats['telegram']['total_documents']}")
                    st.write(f"📱 Channel: {stats['telegram']['channel']}")
                    st.write(f"📞 Phone: {stats['telegram']['phone_number']}")
                
                if 'local_cache' in stats:
                    st.markdown("**Local Cache:**")
                    st.write(f"📄 Documents: {stats['local_cache']['total_documents']}")
                    st.write(f"🔍 Index: {'✅' if stats['local_cache']['has_index'] else '❌'}")
                
                if 'local' in stats:
                    st.markdown("**Local Storage:**")
                    st.write(f"📄 Documents: {stats['local']['total_documents']}")
                    st.write(f"🔍 Index: {'✅' if stats['local']['has_index'] else '❌'}")
                    
            except Exception as e:
                st.error(f"Failed to get stats: {e}")
    
    # Initialize agent
    if not asyncio.run(initialize_agent()):
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 AI Chat", 
        "❓ Q&A", 
        "📄 Summaries", 
        "🔍 Search", 
        "📚 Document Management"
    ])
    
    # Tab 1: AI Chat
    with tab1:
        st.header("🤖 Chat with Your Documents")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
                    if 'sources' in message and message['sources']:
                        with st.expander("📚 Sources"):
                            for source in message['sources']:
                                st.write(f"- Document: {source['document_id']} (Score: {source['score']:.3f})")
                st.divider()
        
        # Chat input
        user_input = st.text_input("💬 Ask anything about your documents:", key="chat_input")
        
        if st.button("Send", key="send_chat") and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Get AI response
            with st.spinner("🤔 AI is thinking..."):
                response = st.session_state.agent.chat_with_documents(user_input)
                
                if response['success']:
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response['response'],
                        'sources': response.get('sources', []),
                        'timestamp': datetime.now()
                    })
                else:
                    st.error(f"❌ Error: {response.get('error', 'Unknown error')}")
            
            st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Tab 2: Q&A
    with tab2:
        st.header("❓ Question & Answer")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_area("🤔 Ask a specific question:", height=100)
        
        with col2:
            response_style = st.selectbox(
                "📝 Response Style:",
                ["comprehensive", "brief", "bullet_points", "step_by_step"]
            )
        
        if st.button("🔍 Get Answer", key="qa_button") and question:
            with st.spinner("🔍 Searching for answer..."):
                result = st.session_state.agent.ask_question(question, response_style)
                
                if result['success']:
                    st.success("✅ Answer found!")
                    st.markdown("### 📝 Answer:")
                    st.write(result['answer'])
                    
                    if result.get('sources'):
                        st.markdown("### 📚 Sources:")
                        for source in result['sources']:
                            st.write(f"- Document: {source['document_id']} (Relevance: {source['relevance_score']:.3f})")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Tab 3: Summaries
    with tab3:
        st.header("📄 Document Summaries")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            focus_topic = st.text_input("🎯 Focus on specific topic (optional):")
        
        with col2:
            st.write("")  # Spacing
            generate_summary = st.button("📝 Generate Summary", key="summary_button")
        
        if generate_summary:
            with st.spinner("📝 Generating summary..."):
                result = st.session_state.agent.generate_summary(focus_topic if focus_topic else None)
                
                if result['success']:
                    st.success("✅ Summary generated!")
                    
                    if result.get('focus_topic'):
                        st.markdown(f"### 🎯 Summary: {result['focus_topic']}")
                    else:
                        st.markdown("### 📄 Complete Document Summary")
                    
                    st.write(result['summary'])
                    st.info(f"📊 Based on {result['documents_count']} documents")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Tab 4: Search
    with tab4:
        st.header("🔍 Document Search")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search documents:")
        
        with col2:
            top_k = st.number_input("📊 Results:", min_value=1, max_value=20, value=5)
        
        if st.button("🔍 Search", key="search_button") and search_query:
            with st.spinner("🔍 Searching documents..."):
                results = st.session_state.agent.search_documents(search_query, top_k)
                
                if results:
                    st.success(f"✅ Found {len(results)} results")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 Result {i}: {result['document_id']} (Score: {result['score']:.3f})"):
                            if 'highlighted_content' in result:
                                st.markdown("**Highlighted Content:**")
                                st.markdown(result['highlighted_content'], unsafe_allow_html=True)
                            else:
                                st.write(result['content'][:500] + "...")
                            
                            if result.get('metadata'):
                                st.json(result['metadata'])
                else:
                    st.warning("❌ No results found")
    
    # Tab 5: Document Management
    with tab5:
        st.header("📚 Document Management")
        
        # Add new document
        st.subheader("➕ Add New Document")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_doc_id = st.text_input("📄 Document ID:")
            new_doc_content = st.text_area("📝 Document Content:", height=200)
        
        with col2:
            new_doc_title = st.text_input("📋 Title (optional):")
            new_doc_category = st.text_input("🏷️ Category (optional):")
            new_doc_tags = st.text_input("🏷️ Tags (comma-separated):")
        
        if st.button("➕ Add Document") and new_doc_id and new_doc_content:
            metadata = {}
            if new_doc_title:
                metadata['title'] = new_doc_title
            if new_doc_category:
                metadata['category'] = new_doc_category
            if new_doc_tags:
                metadata['tags'] = [tag.strip() for tag in new_doc_tags.split(',')]
            
            async def add_doc():
                return await st.session_state.agent.add_document(new_doc_id, new_doc_content, metadata)
            
            with st.spinner("📤 Adding document..."):
                success = asyncio.run(add_doc())
                
                if success:
                    st.success(f"✅ Document '{new_doc_id}' added successfully!")
                    # Rebuild index
                    st.session_state.agent.initialize_search_index()
                else:
                    st.error("❌ Failed to add document")
        
        st.divider()
        
        # List existing documents
        st.subheader("📋 Existing Documents")
        
        if st.session_state.agent and st.session_state.agent.document_store.documents:
            for doc in st.session_state.agent.document_store.documents:
                with st.expander(f"📄 {doc['id']}"):
                    st.write(f"**Content Preview:** {doc['content'][:200]}...")
                    if doc.get('metadata'):
                        st.write("**Metadata:**")
                        st.json(doc['metadata'])
                    
                    # Delete button
                    if st.button(f"🗑️ Delete", key=f"delete_{doc['id']}"):
                        async def delete_doc():
                            return await st.session_state.agent.remove_document(doc['id'])
                        
                        with st.spinner("🗑️ Deleting document..."):
                            success = asyncio.run(delete_doc())
                            
                            if success:
                                st.success(f"✅ Document '{doc['id']}' deleted!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete document")
        else:
            st.info("📭 No documents found. Add some documents to get started!")

if __name__ == "__main__":
    main()