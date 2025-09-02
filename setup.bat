@echo off
echo 🚀 RAG Document Search Agent Setup
echo ==================================================

echo 🔧 Installing dependencies...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Error installing dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully!
echo.
echo 🧪 Testing installation...
python -c "from rag_agent import RAGAgent; print('✅ RAG Agent imported successfully!')"

if %errorlevel% neq 0 (
    echo ❌ Import test failed
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📋 Next steps:
echo 1. Run example: python example_usage.py
echo 2. Launch web app: streamlit run streamlit_app.py
echo 3. Check README.md for detailed usage instructions
echo.
pause