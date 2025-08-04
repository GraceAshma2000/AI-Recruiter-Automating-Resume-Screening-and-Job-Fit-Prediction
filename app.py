import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Additional imports for PDF support
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    try:
        import pdfplumber
        PDF_SUPPORT = True
    except ImportError:
        try:
            import fitz  # PyMuPDF
            PDF_SUPPORT = True
        except ImportError:
            PDF_SUPPORT = False

# Additional imports for DOCX support
try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    try:
        from docx import Document
        DOCX_SUPPORT = True
    except ImportError:
        DOCX_SUPPORT = False

# Page config
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #2c3e50;
    }
    
    .prediction-result {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
        text-align: center;
    }
    
    .score-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models and components"""
    try:
        model = joblib.load('C:/Users/grace/Downloads/AI Resume Classification/saved_models/best_resume_classifier_random_forest.pkl')
        tfidf_vectorizer = joblib.load('C:/Users/grace/Downloads/AI Resume Classification/saved_models/vectorizers/tfidf_vectorizer.pkl')
        label_encoder = joblib.load('C:/Users/grace/Downloads/AI Resume Classification/saved_models/encoders/label_encoder.pkl')
        preprocessing_components = joblib.load('C:/Users/grace/Downloads/AI Resume Classification/saved_models/preprocessing_components.pkl')
        
        return {
            'model': model,
            'tfidf_vectorizer': tfidf_vectorizer,
            'label_encoder': label_encoder,
            'preprocessing_components': preprocessing_components
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

@st.cache_data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using multiple methods"""
    text = ""
    
    try:
        if PDF_SUPPORT:
            # Try PyPDF2 first
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                if text.strip():
                    return text
            except:
                pass
            
            # Try pdfplumber
            try:
                import pdfplumber
                pdf_file.seek(0)
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text
            except:
                pass
            
            # Try PyMuPDF
            try:
                import fitz
                pdf_file.seek(0)
                pdf_bytes = pdf_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text += page.get_text() + "\n"
                pdf_document.close()
                if text.strip():
                    return text
            except:
                pass
        
        if not text.strip():
            st.error("Could not extract text from PDF. Please try a different file.")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None
    
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        if DOCX_SUPPORT:
            import docx
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        else:
            st.error("DOCX support not available. Please install python-docx package.")
            return None
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return None

def advanced_text_preprocessing(text):
    """Advanced text preprocessing pipeline (same as original)"""
    if pd.isna(text) or not text.strip():
        return ""
    
    # Initialize stemmer and stop words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Add domain-specific stop words
    domain_stop_words = {
        'experience', 'year', 'years', 'month', 'months', 'work', 'working', 'company',
        'role', 'position', 'job', 'team', 'project', 'projects', 'client', 'clients',
        'skill', 'skills', 'knowledge', 'understanding', 'ability', 'experience',
        'strong', 'good', 'excellent', 'proficient', 'expert', 'advanced', 'basic'
    }
    stop_words.update(domain_stop_words)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    
    # Handle programming languages and technical terms
    text = re.sub(r'\bc\+\+\b', 'cplusplus', text)
    text = re.sub(r'\bc#\b', 'csharp', text)
    text = re.sub(r'\.net\b', 'dotnet', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Remove tokens that are just numbers
    tokens = [token for token in tokens if not token.isdigit()]
    
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

def extract_text_features(text):
    """Extract comprehensive text features (same as original)"""
    if pd.isna(text) or not text.strip():
        return {}
    
    # Basic counts
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len([s for s in text.split('.') if s.strip()])
    
    # Character analysis
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    special_char_count = sum(1 for c in text if c in string.punctuation)
    
    # Advanced metrics
    avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Readability metrics
    try:
        import textstat
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
    except:
        flesch_reading_ease = flesch_kincaid_grade = gunning_fog = 0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'uppercase_count': uppercase_count,
        'digit_count': digit_count,
        'special_char_count': special_char_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'gunning_fog': gunning_fog
    }

def extract_technical_skills(text):
    """Extract technical skills and keywords (same as original)"""
    if pd.isna(text) or not text.strip():
        return {}
    
    # Define skill patterns
    skill_patterns = {
        'programming_languages': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'cplusplus', 'c#', 'csharp',
            'php', 'ruby', 'go', 'rust', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'sas'
        ],
        'web_technologies': [
            'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask',
            'spring', 'laravel', 'bootstrap', 'jquery', 'webpack', 'sass', 'less'
        ],
        'databases': [
            'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlserver', 'sqlite',
            'cassandra', 'elasticsearch', 'dynamodb', 'firebase', 'hbase'
        ],
        'cloud_platforms': [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
            'jenkins', 'gitlab', 'circleci', 'heroku', 'digitalocean'
        ],
        'data_science': [
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            'matplotlib', 'seaborn', 'plotly', 'tableau', 'powerbi', 'spark', 'hadoop'
        ],
        'tools': [
            'git', 'github', 'jira', 'confluence', 'slack', 'trello', 'figma', 'sketch',
            'photoshop', 'illustrator', 'excel', 'powerpoint', 'word'
        ]
    }
    
    text_lower = text.lower()
    found_skills = {}
    
    for category, skills in skill_patterns.items():
        found_skills[category] = [skill for skill in skills if skill in text_lower]
    
    return found_skills

def extract_experience_years(text):
    """Extract years of experience using multiple patterns (same as original)"""
    if pd.isna(text) or not text.strip():
        return 0
    
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'(?:experience|exp).*?(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*(?:years?|yrs?)',
        r'over\s*(\d+)\s*(?:years?|yrs?)',
        r'more\s*than\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\+\s*(?:years?|yrs?)'
    ]
    
    text_lower = text.lower()
    max_experience = 0
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                experience = max([int(match) for match in matches if match.isdigit()])
                max_experience = max(max_experience, experience)
            except:
                continue
    
    return min(max_experience, 50)  # Cap at 50 years

def create_advanced_features(df):
    """Create advanced features for machine learning (same as original)"""
    
    # Text complexity features
    df['text_complexity_score'] = (
        df['avg_word_length'] * 0.3 +
        df['avg_sentence_length'] * 0.3 +
        (100 - df['flesch_reading_ease']) * 0.4
    )
    
    # Professional experience level
    def categorize_experience(years):
        if years <= 1:
            return 'Entry Level'
        elif years <= 3:
            return 'Junior'
        elif years <= 7:
            return 'Mid Level'
        elif years <= 12:
            return 'Senior'
        else:
            return 'Expert'
    
    df['experience_level'] = df['experience_years'].apply(categorize_experience)
    
    # Skills diversity score
    df['skills_diversity'] = (
        (df['programming_languages_count'] > 0).astype(int) +
        (df['web_technologies_count'] > 0).astype(int) +
        (df['databases_count'] > 0).astype(int) +
        (df['cloud_platforms_count'] > 0).astype(int) +
        (df['data_science_count'] > 0).astype(int) +
        (df['tools_count'] > 0).astype(int)
    )
    
    # Resume quality score (composite metric)
    df['resume_quality_score'] = (
        np.log1p(df['word_count']) * 0.2 +
        df['total_skills'] * 0.3 +
        df['experience_years'] * 0.2 +
        df['skills_diversity'] * 0.3
    )
    
    # Technical depth score
    df['technical_depth'] = (
        df['programming_languages_count'] * 2 +
        df['web_technologies_count'] * 1.5 +
        df['databases_count'] * 1.5 +
        df['cloud_platforms_count'] * 2 +
        df['data_science_count'] * 2.5 +
        df['tools_count'] * 1
    )
    
    return df

def predict_category(resume_text, components):
    """Make prediction on resume category (using original feature extraction)"""
    try:
        model = components['model']
        tfidf_vectorizer = components['tfidf_vectorizer']
        label_encoder = components['label_encoder']
        numerical_features = components['preprocessing_components']['numerical_features']
        
        # Preprocess text
        processed_text = advanced_text_preprocessing(resume_text)
        
        if not processed_text.strip():
            raise ValueError("Processed text is empty")
        
        # Extract features using original methods
        features_dict = extract_text_features(resume_text)
        skills_dict = extract_technical_skills(resume_text)
        experience_years = extract_experience_years(resume_text)
        
        # Create enhanced dataframe for feature engineering (same as original)
        temp_df = pd.DataFrame([{
            'Resume': resume_text,
            'processed_resume': processed_text,
            'experience_years': experience_years,
            **features_dict,
            'technical_skills': skills_dict
        }])
        
        # Add skill counts (same as original)
        for skill_category in ['programming_languages', 'web_technologies', 'databases',
                              'cloud_platforms', 'data_science', 'tools']:
            temp_df[f'{skill_category}_count'] = temp_df['technical_skills'].apply(
                lambda x: len(x.get(skill_category, []))
            )
        
        temp_df['total_skills'] = temp_df[['programming_languages_count', 'web_technologies_count',
                                         'databases_count', 'cloud_platforms_count',
                                         'data_science_count', 'tools_count']].sum(axis=1)
        
        # Create advanced features (same as original)
        temp_df = create_advanced_features(temp_df)
        
        # Vectorize text
        text_features = tfidf_vectorizer.transform([processed_text])
        
        # Numerical features
        numerical_vals = temp_df[numerical_features].values
        
        # Combine features
        combined_features = hstack([text_features, numerical_vals])
        
        # Make prediction
        prediction_encoded = model.predict(combined_features)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(combined_features)[0]
            confidence = np.max(probabilities) * 100
        else:
            confidence = 50.0
        
        return {
            'prediction': prediction,
            'confidence': confidence
        }
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

def main():
    # Download NLTK data
    download_nltk_data()
    
    # Load models
    components = load_models()
    
    if components is None:
        st.error("❌ Failed to load models.")
        return
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>📄 Resume Classifier</h1>
        <p>Upload or paste your resume to get category prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input options
    input_method = st.radio("Choose input method:", ["📝 Paste Text", "📁 Upload File"], horizontal=True)
    
    resume_text = ""
    
    if input_method == "📝 Paste Text":
        resume_text = st.text_area(
            "Paste your resume text:",
            height=200,
            placeholder="Copy and paste your resume content here..."
        )
    else:
        # File upload with PDF/DOCX support
        supported_types = ['txt']
        if PDF_SUPPORT:
            supported_types.append('pdf')
        if DOCX_SUPPORT:
            supported_types.append('docx')
        
        uploaded_file = st.file_uploader(
            "Upload resume file:",
            type=supported_types
        )
        
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.type
                file_name = uploaded_file.name.lower()
                
                with st.spinner("Extracting text from file..."):
                    if file_type == "text/plain" or file_name.endswith('.txt'):
                        resume_text = str(uploaded_file.read(), "utf-8")
                        
                    elif file_type == "application/pdf" or file_name.endswith('.pdf'):
                        if PDF_SUPPORT:
                            resume_text = extract_text_from_pdf(uploaded_file)
                            if resume_text is None:
                                resume_text = ""
                        else:
                            st.error("PDF support not available.")
                            resume_text = ""
                            
                    elif (file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" 
                          or file_name.endswith('.docx')):
                        if DOCX_SUPPORT:
                            resume_text = extract_text_from_docx(uploaded_file)
                            if resume_text is None:
                                resume_text = ""
                        else:
                            st.error("DOCX support not available.")
                            resume_text = ""
                    else:
                        st.error(f"Unsupported file type: {file_type}")
                        resume_text = ""
                
                if resume_text.strip():
                    st.success(f"✅ File processed successfully!")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                resume_text = ""
    
    # Analyze button
    if st.button("🚀 Predict Category", type="primary", use_container_width=True):
        if resume_text.strip():
            if len(resume_text) < 50:
                st.warning("⚠️ Resume text is too short.")
            else:
                with st.spinner("Analyzing resume..."):
                    try:
                        result = predict_category(resume_text, components)
                        
                        # Show result
                        st.markdown(f"""
                        <div class='prediction-result'>
                            <h2>Predicted Category</h2>
                            <h1 style='color: #007bff; margin: 0.5rem 0;'>{result['prediction']}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show score
                        st.markdown(f"""
                        <div class='score-box'>
                            <h3>Confidence Score</h3>
                            <h2 style='color: #28a745; margin: 0;'>{result['confidence']:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please provide resume text.")

if __name__ == "__main__":
    main()