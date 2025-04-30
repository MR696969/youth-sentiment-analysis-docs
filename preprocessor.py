import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Ensure NLTK data directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Force download all required NLTK data at startup
def download_nltk_data():
    required_packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger'
    ]
    
    for package in required_packages:
        print(f"Downloading {package}...")
        try:
            nltk.download(package, quiet=False, raise_on_error=True)
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
            # Second attempt with different download method
            try:
                nltk.download(package, download_dir=nltk_data_dir)
            except Exception as e:
                print(f"Failed to download {package}: {str(e)}")

# Download required data at module import
print("Initializing NLTK downloads...")
download_nltk_data()
print("NLTK downloads completed.")

class TextPreprocessor:
    def __init__(self):
        # Ensure all required NLTK data is available
        download_nltk_data()
        
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError as e:
            print(f"Error loading stopwords: {str(e)}")
            nltk.download('stopwords', quiet=False)
            self.stop_words = set(stopwords.words('english'))
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            strip_accents='unicode',
            lowercase=True
        )
    
    def clean_text(self, text):
        """Clean and preprocess text data."""
        if not text:
            return ""
            
        # Convert to lowercase and ensure text is string
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        try:
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words]
            
            return ' '.join(tokens)
        except LookupError:
            # If tokenization fails, try downloading required data again
            print("Tokenization failed. Attempting to download required data...")
            download_nltk_data()
            try:
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                         if token not in self.stop_words]
                return ' '.join(tokens)
            except Exception as e:
                print(f"Error in text processing: {str(e)}")
                return text  # Return original text if processing fails
    
    def prepare_data(self, texts, labels=None, train=True):
        """Prepare data for model training or prediction."""
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        if train:
            # Fit and transform for training data
            X = self.vectorizer.fit_transform(cleaned_texts)
            return X, labels
        else:
            # Transform only for test data
            try:
                X = self.vectorizer.transform(cleaned_texts)
            except:
                # If vectorizer hasn't been fit yet, fit and transform
                X = self.vectorizer.fit_transform(cleaned_texts)
            return X 