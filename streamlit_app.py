import streamlit as st
import json
from rag_agent import RAGAgent
from document_loader import DocumentLoader
import os

# Page config
st.set_page_config(
    page_title="RAG Document Search Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_agent():
    """Initialize the RAG agent"""
    with st.spinner("Initializing RAG Agent..."):
        agent = RAGAgent()
        
        # Load sample documents for demo
        agent.load_documents()
        agent.initialize()
        
        st.session_state.agent = agent
        st.session_state.initialized = True
        st.success("RAG Agent initialized with sample documents!")

def main():
    st.title("🔍 RAG Document Search Agent")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Agent Configuration")
        
        if not st.session_state.initialized:
            if st.button("Initialize Agent", type="primary"):
                initialize_agent()
        else:
            st.success("✅ Agent Ready")
            
            # Agent stats
            if st.session_state.agent:
                stats = st.session_state.agent.get_stats()
                st.metric("Total Documents", stats['total_documents'])
                st.metric("Embedding Dimension", stats['embedding_dimension'])
        
        st.markdown("---")
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'json', 'md']
        )
        
        if uploaded_files and st.button("Process Uploaded Files"):
            # Process uploaded files (simplified for demo)
            st.info("File upload processing would be implemented here")
    
    # Main content
    if not st.session_state.initialized:
        st.info("👈 Please initialize the agent from the sidebar to get started")
        
        # Show sample documents
        st.subheader("Sample Documents Preview")
        sample_docs = DocumentLoader.create_sample_documents()
        for i, doc in enumerate(sample_docs, 1):
            with st.expander(f"Document {i}: {doc['id']}"):
                st.write(f"**Category:** {doc['metadata']['category']}")
                st.write(f"**Topic:** {doc['metadata']['topic']}")
                st.write(f"**Content:** {doc['content']}")
        
    else:
        # Search interface
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
                    results = st.session_state.agent.search(query, top_k)
                    
                    if results:
                        st.success(f"Found {len(results)} relevant documents")
                        
                        # Display results
                        for result in results:
                            with st.expander(
                                f"📄 Rank {result['rank']}: {result['document_id']} "
                                f"(Similarity: {result['score']:.3f})"
                            ):
                                st.write(f"**Content:** {result['content']}")
                                if result['metadata']:
                                    st.write(f"**Metadata:** {json.dumps(result['metadata'], indent=2)}")
                    else:
                        st.warning("No relevant documents found")
                        
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
        
        # RAG Response
        st.markdown("---")
        st.subheader("🤖 RAG Response")
        
        if query:
            if st.button("Generate RAG Response"):
                with st.spinner("Generating response..."):
                    try:
                        response = st.session_state.agent.generate_response(query, top_k=3)
                        
                        st.write("**Query:**", response['query'])
                        st.write("**Summary:**", response['summary'])
                        
                        st.write("**Context from Retrieved Documents:**")
                        st.text_area("Context", response['context'], height=200)
                        
                    except Exception as e:
                        st.error(f"Response generation error: {str(e)}")

if __name__ == "__main__":
    main()