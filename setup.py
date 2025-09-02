#!/usr/bin/env python3
"""
Setup script for RAG Document Search Agent
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    try:
        from rag_agent import RAGAgent
        agent = RAGAgent()
        print("✅ RAG Agent imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("🚀 RAG Document Search Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run example: python example_usage.py")
    print("2. Launch web app: streamlit run streamlit_app.py")
    print("3. Check README.md for detailed usage instructions")

if __name__ == "__main__":
    main()