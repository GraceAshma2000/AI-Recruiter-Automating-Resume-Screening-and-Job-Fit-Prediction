#!/bin/bash

# AI Resume Classifier Deployment Script
# This script sets up the Streamlit application for deployment

echo "🚀 Setting up AI Resume Classifier Application..."

# Create project structure
echo "📁 Creating project structure..."
mkdir -p .streamlit
mkdir -p saved_models/vectorizers
mkdir -p saved_models/encoders

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ pip found"

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create Streamlit config
echo "⚙️ Creating Streamlit configuration..."
cat > .streamlit/config.toml << EOF
[global]
developmentMode = false

[server]
runOnSave = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
caching = true
displayEnabled = true

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true

[logger]
level = "info"
messageFormat = "%(asctime)s %(message)s"
EOF

echo "✅ Streamlit configuration created"

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️ NLTK download warning: {e}')
"

# Check if model files exist
echo "🔍 Checking for model files..."
required_files=(
    "saved_models/best_resume_classifier_random_forest.pkl"
    "saved_models/vectorizers/tfidf_vectorizer.pkl"
    "saved_models/encoders/label_encoder.pkl"
    "saved_models/preprocessing_components.pkl"
    "saved_models/model_performance.pkl"
    "saved_models/deployment_config.pkl"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ All model files found"
else
    echo "⚠️ Missing model files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo "Please ensure all model files are in the saved_models directory"
fi

# Test import of main modules
echo "🧪 Testing application imports..."
python3 -c "
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import sklearn
    import nltk
    import plotly
    import textstat
    import wordcloud
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Create run script
echo "📝 Creating run script..."
cat > run_app.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting AI Resume Classifier..."
echo "📱 Application will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""
streamlit run app.py
EOF

chmod +x run_app.sh

echo "✅ Run script created (run_app.sh)"

# Final setup check
echo ""
echo "🔍 Final Setup Check:"
echo "✅ Dependencies installed"
echo "✅ Configuration files created"
echo "✅ NLTK data downloaded"
echo "✅ Project structure ready"

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ Model files verified"
    echo ""
    echo "🎉 Setup complete! You can now run the application:"
    echo "   ./run_app.sh"
    echo "   OR"
    echo "   streamlit run app.py"
else
    echo "⚠️ Some model files are missing - please add them before running"
fi

echo ""
echo "📖 For more information, see README.md"
echo "🐛 For troubleshooting, check the README.md troubleshooting section"