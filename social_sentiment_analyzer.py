import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import re
from typing import Dict, Any, List, Tuple
import logging

class SocialSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        try:
            self.transformer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
            )
        except Exception as e:
            logging.error(f"Error initializing transformer model: {str(e)}")
            self.transformer = None
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess social media text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags but keep the text
        text = re.sub(r'#', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple models and return combined results"""
        if not text or text.isspace():
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'overall_sentiment': 'neutral'
            }

        cleaned_text = self.preprocess_text(text)
        
        # VADER analysis
        vader_scores = self.vader.polarity_scores(cleaned_text)
        
        # TextBlob analysis
        blob = TextBlob(cleaned_text)
        blob_sentiment = blob.sentiment.polarity
        
        # Transformer model analysis
        transformer_label = 'neutral'
        transformer_score = 0.5
        if self.transformer:
            try:
                transformer_result = self.transformer(cleaned_text, truncation=True, max_length=512)[0]
                transformer_label = transformer_result['label']
                transformer_score = transformer_result['score']
            except Exception as e:
                logging.error(f"Error in transformer analysis: {str(e)}")

        # Combine scores
        compound_score = (vader_scores['compound'] + blob_sentiment + 
                        (transformer_score if transformer_label == 'POSITIVE' else -transformer_score)) / 3

        return {
            'compound': compound_score,
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'overall_sentiment': self._get_sentiment_label(compound_score),
            'confidence': self._calculate_confidence([
                abs(vader_scores['compound']),
                abs(blob_sentiment),
                transformer_score
            ])
        }

    def analyze_social_media_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of a social media post with context"""
        text = post.get('text', '')
        if post.get('platform') == 'reddit' and post.get('title'):
            text = f"{post['title']} {text}"
        
        sentiment_results = self.analyze_sentiment(text)
        
        # Add engagement metrics if available
        engagement_score = self._calculate_engagement_score(post)
        if engagement_score is not None:
            sentiment_results['engagement_score'] = engagement_score
        
        return {
            **sentiment_results,
            'platform': post.get('platform', 'unknown'),
            'created_at': post.get('created_at'),
            'author': post.get('author'),
            'original_text': text
        }

    def analyze_batch(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of social media posts"""
        results = []
        for post in posts:
            try:
                analysis = self.analyze_social_media_post(post)
                results.append(analysis)
            except Exception as e:
                logging.error(f"Error analyzing post: {str(e)}")
                continue
        return results

    def _get_sentiment_label(self, compound_score: float) -> str:
        """Convert compound score to sentiment label"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence score based on agreement between different models"""
        return float(np.mean([abs(score) for score in scores]))

    def _calculate_engagement_score(self, post: Dict[str, Any]) -> float:
        """Calculate normalized engagement score based on platform-specific metrics"""
        try:
            if post.get('platform') == 'twitter':
                return (post.get('likes', 0) + post.get('retweets', 0) * 2) / 100
            elif post.get('platform') == 'reddit':
                return (post.get('score', 0) + post.get('comments', 0) * 2) / 100
            elif post.get('platform') == 'facebook':
                return post.get('likes', 0) / 100
            elif post.get('platform') == 'instagram':
                return (post.get('likes', 0) + post.get('comments', 0) * 2) / 100
            elif post.get('platform') == 'youtube':
                return post.get('likes', 0) / 100
        except Exception:
            pass
        return None 