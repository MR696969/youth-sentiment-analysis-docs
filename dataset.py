import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
from typing import List, Tuple, Dict, Union
import torch
from torch.utils.data import Dataset, DataLoader
import requests
import gzip
import shutil
from tqdm import tqdm
import tarfile
import zipfile
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
import re

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 512):
        """
        Initialize the sentiment dataset
        
        Args:
            texts (List[str]): List of text samples
            labels (List[int]): List of corresponding labels (0 for negative, 1 for positive)
            tokenizer: Tokenizer for text preprocessing (optional)
            max_length (int): Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }

class SentimentDataLoader:
    def __init__(self, data_path: str = None):
        """
        Initialize the data loader
        
        Args:
            data_path (str): Path to the data file (CSV, JSON, or TXT)
        """
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.data_dir = 'data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def download_file(self, url: str, filename: str) -> str:
        """
        Download a file from a URL with progress bar
        
        Args:
            url (str): URL to download from
            filename (str): Name to save the file as
            
        Returns:
            str: Path to the downloaded file
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File already exists: {filepath}")
            return filepath
            
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
                
        return filepath

    def load_imdb_dataset(self) -> Tuple[List[str], List[int]]:
        """
        Load the IMDB Movie Reviews dataset
        
        Returns:
            Tuple[List[str], List[int]]: Texts and labels
        """
        print("Loading IMDB dataset...")
        dataset = load_dataset("imdb", trust_remote_code=True)
        
        texts = []
        labels = []
        
        # Only use a subset of the data to avoid memory issues
        max_samples = 10000  # Limit to 10k samples
        sample_count = 0
        
        # Combine train and test sets
        for split in ['train', 'test']:
            for item in dataset[split]:
                if sample_count >= max_samples:
                    break
                    
                texts.append(item['text'])
                labels.append(item['label'])
                sample_count += 1
                
            if sample_count >= max_samples:
                break
                
        return texts, labels

    def load_amazon_reviews(self, language: str = 'en') -> Tuple[List[str], List[int]]:
        """
        Load Amazon Reviews dataset for a specific language
        
        Args:
            language (str): Language code ('en', 'de', 'es', 'fr', 'ja', 'zh')
            
        Returns:
            Tuple[List[str], List[int]]: Texts and labels
        """
        print(f"Loading Amazon Reviews dataset for language: {language}")
        dataset = load_dataset("amazon_reviews_multi", language, trust_remote_code=True)
        
        texts = []
        labels = []
        
        # Only use a subset of the data to avoid memory issues
        max_samples = 50000  # Limit to 50k samples
        sample_count = 0
        
        for split in ['train', 'test']:
            for item in dataset[split]:
                if sample_count >= max_samples:
                    break
                    
                texts.append(item['review_body'])
                # Convert 1-5 stars to binary (1-2 negative, 4-5 positive)
                # Ignore neutral reviews (3 stars)
                if item['stars'] <= 2:
                    labels.append(0)  # Negative
                elif item['stars'] >= 4:
                    labels.append(1)  # Positive
                else:
                    continue  # Skip neutral reviews
                    
                sample_count += 1
                
            if sample_count >= max_samples:
                break
                
        return texts, labels

    def load_twitter_sentiment(self) -> Tuple[List[str], List[int]]:
        """
        Load Twitter Sentiment dataset
        
        Returns:
            Tuple[List[str], List[int]]: Texts and labels
        """
        print("Loading Twitter Sentiment dataset...")
        try:
            # Try loading from Hugging Face
            dataset = load_dataset("sentiment140", trust_remote_code=True)
            
            texts = []
            labels = []
            
            # Only use a subset of the data to avoid memory issues
            max_samples = 10000  # Limit to 10k samples
            sample_count = 0
            
            for item in dataset['train']:
                if sample_count >= max_samples:
                    break
                    
                texts.append(item['text'])
                # Convert sentiment to binary (0 for negative, 1 for positive)
                labels.append(1 if item['sentiment'] == 4 else 0)
                sample_count += 1
                
            return texts, labels
            
        except Exception as e:
            print(f"Error loading Twitter dataset: {str(e)}")
            return [], []

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing special characters, URLs, etc.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def load_csv_data(self, file_path: str, text_column: str, label_column: str) -> Tuple[List[str], List[int]]:
        """
        Load data from a CSV file
        
        Args:
            file_path (str): Path to the CSV file
            text_column (str): Name of the column containing text data
            label_column (str): Name of the column containing labels
            
        Returns:
            Tuple[List[str], List[int]]: Texts and labels
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in CSV")
            
            # Extract texts and labels
            texts = df[text_column].astype(str).tolist()
            
            # Convert text labels to numeric (0 for negative, 1 for positive)
            label_map = {'negative': 0, 'positive': 1}
            labels = df[label_column].map(label_map).tolist()
            
            # Preprocess texts
            texts = [self.preprocess_text(text) for text in texts]
            
            return texts, labels
            
        except Exception as e:
            print(f"Error loading CSV data: {str(e)}")
            return [], []

    def load_data(self, file_path: str = None, dataset_name: str = None, 
                 text_column: str = None, label_column: str = None) -> Tuple[List[str], List[int]]:
        """
        Load data from various sources
        
        Args:
            file_path (str): Path to the data file
            dataset_name (str): Name of the dataset to load ('imdb', 'amazon', 'twitter')
            text_column (str): Name of the text column in CSV file
            label_column (str): Name of the label column in CSV file
            
        Returns:
            Tuple[List[str], List[int]]: Texts and labels
        """
        if file_path and file_path.endswith('.csv'):
            if not text_column or not label_column:
                raise ValueError("text_column and label_column are required for CSV files")
            return self.load_csv_data(file_path, text_column, label_column)
            
        if dataset_name:
            if dataset_name.lower() == 'imdb':
                texts, labels = self.load_imdb_dataset()
            elif dataset_name.lower() == 'amazon':
                texts, labels = self.load_amazon_reviews()
            elif dataset_name.lower() == 'twitter':
                texts, labels = self.load_twitter_sentiment()
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
            # Preprocess texts
            texts = [self.preprocess_text(text) for text in texts]
            return texts, labels
            
        if file_path is None:
            file_path = self.data_path
            
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            texts = df['text'].tolist()
            labels = self.label_encoder.fit_transform(df['label'].tolist())
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts = [item['text'] for item in data]
            labels = self.label_encoder.fit_transform([item['label'] for item in data])
        elif file_path.endswith('.txt'):
            texts = []
            labels = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    text, label = line.strip().split('\t')
                    texts.append(text)
                    labels.append(int(label))
            labels = self.label_encoder.fit_transform(labels)
        else:
            raise ValueError("Unsupported file format. Use CSV, JSON, or TXT.")
            
        return texts, labels
    
    def create_data_loaders(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer=None,
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders
        
        Args:
            texts (List[str]): List of text samples
            labels (List[int]): List of labels
            tokenizer: Tokenizer for text preprocessing
            batch_size (int): Batch size for data loaders
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
        """
        # Split data into train and test sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        
        # Split train data into train and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=val_size, random_state=random_state
        )
        
        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader

    def save_sample_data(self, output_path: str = 'sample_data.csv'):
        """
        Create and save a sample dataset for testing
        
        Args:
            output_path (str): Path to save the sample data
        """
        sample_data = {
            'text': [
                "This product is amazing! I love it!",
                "Terrible service, would not recommend.",
                "Great experience, will buy again.",
                "The quality is poor and it broke easily.",
                "Excellent customer service and fast delivery.",
                "Waste of money, very disappointed.",
                "Best purchase I've made this year!",
                "Not worth the price at all.",
                "Highly recommended, exceeded my expectations.",
                "The product arrived damaged."
            ],
            'label': [
                'positive',
                'negative',
                'positive',
                'negative',
                'positive',
                'negative',
                'positive',
                'negative',
                'positive',
                'negative'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}") 