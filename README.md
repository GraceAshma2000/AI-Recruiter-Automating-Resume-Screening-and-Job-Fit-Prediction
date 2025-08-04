# 🤖 AI Resume Classifier - Streamlit Application

A professional web application for automatic resume classification using advanced machine learning and natural language processing techniques.

## 🚀 Features

- **Advanced Text Analysis**: NLP preprocessing with skill extraction and experience detection
- **Multiple Input Methods**: Text paste or file upload support
- **Real-time Predictions**: Instant classification with confidence scores
- **Visual Analytics**: Interactive charts, probability distributions, and word clouds
- **Comprehensive Metrics**: Resume quality assessment and recommendations
- **Professional UI**: Modern, responsive design with intuitive navigation

## 📋 Supported Categories

- Data Science
- Software Engineer
- Product Manager
- UX/UI Designer
- Marketing Manager

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download the Project

Ensure you have the following project structure:
```
resume_classifier/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── saved_models/              # Pre-trained models directory
│   ├── best_resume_classifier_random_forest.pkl
│   ├── vectorizers/
│   │   ├── tfidf_vectorizer.pkl
│   │   └── count_vectorizer.pkl
│   ├── encoders/
│   │   └── label_encoder.pkl
│   ├── preprocessing_components.pkl
│   ├── model_performance.pkl
│   └── deployment_config.pkl
└── README.md

```

### Step 2 : open Anaconda Promt cmd

### Step 3 : Change the path to
cd C:\Users\grace\Downloads\AI Resume Classification\AI Resume Classification

### Step 3: Install Dependencies-> for first time

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data -> for First time

run -> setup.bat

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Resume Analysis Tab

- **Text Input**: Paste resume text directly or upload a TXT file
- **Analysis**: Click "Analyze Resume" to get predictions
- **Results**: View predicted category, confidence score, and detailed analytics

### 2. Model Performance Tab

- View model comparison metrics
- Check training results and validation scores
- Analyze model accuracy and performance statistics

### 3. Analytics Dashboard Tab

- System usage statistics
- Prediction distribution charts
- Feature importance analysis
- Confidence score distributions

### 4. About Tab

- Project overview and technical details
- Supported categories and use cases
- System information and specifications

## 🔧 Technical Details

### Machine Learning Pipeline

1. **Text Preprocessing**:
   - Lowercasing and tokenization
   - Stop word removal and stemming
   - Special character handling
   - Technical term preservation

2. **Feature Extraction**:
   - TF-IDF vectorization (5000 features)
   - Technical skill pattern matching
   - Experience years extraction
   - Text complexity metrics
   - Readability scores

3. **Model Architecture**:
   - Random Forest Classifier (optimized)
   - Hyperparameter tuning with GridSearch
   - Cross-validation for robustness
   - Feature importance analysis

### Performance Metrics

- **Accuracy**: 95%+
- **Processing Time**: <1 second
- **Confidence**: High reliability scores
- **Categories**: 5 professional categories

## 🎯 Model Features

- **Text Features**: 5000+ TF-IDF features
- **Numerical Features**: 9 engineered metrics
- **Skill Categories**: 6 technical skill domains
- **Experience Analysis**: Automatic years extraction
- **Quality Scoring**: Composite resume assessment

## 📊 Input Requirements

- **Text Length**: 100-50,000 characters
- **Format**: Plain text (UTF-8 encoding)
- **Content**: Professional resume content
- **Language**: English

## 🔒 Privacy & Security

- **No Data Storage**: Text is processed in memory only
- **Local Processing**: All analysis happens locally
- **No Logging**: Personal information is not retained
- **Secure**: No external API calls for sensitive data

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

1. **Streamlit Cloud**: Upload to GitHub and deploy via Streamlit Cloud
2. **Docker**: Create containerized deployment
3. **Cloud Platforms**: Deploy on AWS, GCP, or Azure
4. **Heroku**: Simple cloud deployment

### Docker Deployment (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t resume-classifier .
docker run -p 8501:8501 resume-classifier
```

## 🔄 Updates & Maintenance

### Model Updates
- Replace model files in `saved_models/` directory
- Update `deployment_config.pkl` with new model information
- Restart the application

### Feature Updates
- Modify `app.py` for UI changes
- Update preprocessing functions for new features
- Test thoroughly before deployment

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**:
   - Ensure all model files are in correct directories
   - Check file permissions and paths
   - Verify pickle file compatibility

2. **NLTK Data Error**:
   - Run NLTK downloads manually
   - Check internet connection for downloads
   - Verify NLTK installation

3. **Memory Issues**:
   - Large files may cause memory problems
   - Increase system memory or optimize model size
   - Use text length limits

4. **Prediction Errors**:
   - Verify input text quality
   - Check for special characters or encoding issues
   - Ensure text meets minimum length requirements

### Performance Optimization

- **Caching**: Streamlit caches models automatically
- **Memory**: Monitor memory usage with large texts
- **Speed**: Preprocessing is the main bottleneck
- **Scaling**: Consider API deployment for high volume

## 📞 Support

For technical issues or questions:
- Check the troubleshooting section
- Review error messages carefully
- Ensure all dependencies are installed correctly
- Verify model files are present and accessible

## 📈 Future Enhancements

- [ ] PDF and DOCX file support
- [ ] Multi-language processing
- [ ] Batch processing capabilities
- [ ] API endpoint creation
- [ ] Industry-specific models
- [ ] Advanced visualization features
- [ ] Export functionality
- [ ] User feedback integration

## 📄 License

This project is for educational and professional use. Please ensure compliance with your organization's policies when deploying.

---

**Built with ❤️ using Streamlit, scikit-learn, and advanced NLP techniques**