import numpy as np
import torch
import torch.nn as nn
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger(__name__)

class NaiveBayesClassifier:
    def __init__(self):
        self.model = MultinomialNB(alpha=1.0)
        self.is_trained = False
        self.vectorizer = TfidfVectorizer(max_features=100)  # Fixed number of features
        
        # Sample training data for basic initialization
        self.sample_texts = [
            "This is great!", "Amazing product", "Love it",
            "Excellent service", "Very good", "Outstanding",
            "Terrible service", "Awful product", "Hate it",
            "Very bad", "Disappointed", "Waste of money",
            "The quality is excellent", "Best purchase ever",
            "Highly recommended", "Don't buy this", "Poor quality",
            "Not worth the money", "Save your money", "Fantastic",
            "Absolutely wonderful", "Terrible experience", "Worst ever",
            "Complete waste", "Perfect!", "Just what I needed"
        ]
        self.sample_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
        
        # Initialize with sample data
        self._initialize_with_samples()
        
    def _initialize_with_samples(self):
        # Fit vectorizer and transform sample texts
        X_sample = self.vectorizer.fit_transform(self.sample_texts)
        # Train the model with sample data
        self.model.fit(X_sample, self.sample_labels)
        self.is_trained = True
        
    def train(self, X, y=None):
        if y is not None:
            # If new training data is provided, update the model
            self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, texts):
        if isinstance(texts, np.ndarray) or isinstance(texts, list):
            # Transform text using the same vectorizer
            X = self.vectorizer.transform(texts)
            return self.model.predict(X)
        return self.model.predict(texts)
    
    def predict_proba(self, texts):
        if isinstance(texts, np.ndarray) or isinstance(texts, list):
            # Transform text using the same vectorizer
            X = self.vectorizer.transform(texts)
            return self.model.predict_proba(X)
        return self.model.predict_proba(texts)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :]
        output = self.fc(hidden)
        return self.sigmoid(output)

class BERTClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use sentiment analysis pipeline from transformers
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1  # Use CPU
        )
        
    def predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
            
        results = self.sentiment_pipeline(texts)
        
        # Convert 5-class sentiment to binary (positive/negative)
        predictions = []
        for result in results:
            # Get sentiment score (1-5) and normalize to 0-1
            score = int(result['label'][0])  # Extract first character (1-5)
            normalized_score = (score - 1) / 4  # Convert 1-5 to 0-1
            predictions.append(normalized_score)
            
        return np.array(predictions)

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item 

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize transformers model
        try:
            model_name = "finiteautomata/bertweet-base-sentiment-analysis"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1 if not torch.cuda.is_available() else 0
            )
        except Exception as e:
            logger.error(f"Error loading transformer model: {str(e)}")
            self.transformer_pipeline = None

        # Custom sentiment lexicon for social media and YouTube comments
        self.custom_lexicon = {
            # Positive sentiment
            'like': 2.0,
            'love': 2.0,
            'awesome': 2.0,
            'amazing': 2.0,
            'great': 1.5,
            'good': 1.0,
            'nice': 1.0,
            'thanks': 1.0,
            'helpful': 1.5,
            'subscribe': 1.0,
            'subscribed': 1.0,
            'interesting': 1.0,
            'beautiful': 1.5,
            'perfect': 2.0,
            'lit': 1.5,
            'fire': 1.5,
            'goat': 2.0,
            
            # Negative sentiment
            'dislike': -2.0,
            'hate': -2.0,
            'terrible': -2.0,
            'awful': -2.0,
            'bad': -1.5,
            'worst': -2.0,
            'boring': -1.0,
            'waste': -1.5,
            'disappointed': -1.5,
            'clickbait': -1.5,
            'cringe': -1.0,
            'trash': -2.0,
            'garbage': -2.0,
            
            # Social media specific
            'W': 1.5,      # Win
            'L': -1.5,     # Loss
            'mid': -1.0,   # Mediocre
            'ratio': -1.0,
            'based': 1.0,
            'cap': -0.5,   # Lie
            'no cap': 1.0, # Truth
            'fr': 0.5,     # For real
            'ngl': 0.5,    # Not gonna lie
            
            # Emojis
            '❤': 2.0,
            '👍': 1.0,
            '👎': -1.0,
            '🔥': 1.5,
            '💯': 2.0,
            '😊': 1.0,
            '😢': -1.0,
            '🙏': 1.0,
            '👏': 1.0,
            '🎉': 1.5,
            '😍': 2.0,
            '🤮': -2.0,
            '💪': 1.0,
            '🐐': 2.0,     # GOAT
        }
        
        # Update VADER lexicon with custom entries
        self.vader.lexicon.update(self.custom_lexicon)

    def analyze(self, text):
        try:
            # Get VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            
            # Get transformer sentiment if available
            transformer_sentiment = None
            if self.transformer_pipeline:
                try:
                    result = self.transformer_pipeline(text[:512])[0]  # Limit text length
                    transformer_sentiment = {
                        'label': result['label'],
                        'score': result['score']
                    }
                except Exception as e:
                    logger.error(f"Transformer analysis error: {str(e)}")
            
            # Combine scores
            if transformer_sentiment:
                # Convert transformer score to compound-like score (-1 to 1)
                if transformer_sentiment['label'] == 'POS':
                    trans_compound = transformer_sentiment['score']
                elif transformer_sentiment['label'] == 'NEG':
                    trans_compound = -transformer_sentiment['score']
                else:
                    trans_compound = 0
                
                # Weighted average of VADER and transformer scores
                compound = (vader_scores['compound'] + trans_compound) / 2
            else:
                compound = vader_scores['compound']
            
            # Get sentiment label
            if compound >= 0.05:
                sentiment_label = 'Positive'
            elif compound <= -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
            
            return {
                'compound': compound,
                'pos': vader_scores['pos'],
                'neu': vader_scores['neu'],
                'neg': vader_scores['neg'],
                'label': sentiment_label,
                'transformer': transformer_sentiment
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return None 