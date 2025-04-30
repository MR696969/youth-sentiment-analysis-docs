from dataset import SentimentDataLoader
import os
from transformers import BertTokenizer
import torch

def main():
    # Initialize data loader
    data_loader = SentimentDataLoader()
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load different datasets
    print("\n=== Loading IMDB Movie Reviews Dataset ===")
    try:
        texts, labels = data_loader.load_data(dataset_name='imdb')
        print(f"Total samples: {len(texts)}")
        print(f"Positive samples: {sum(labels)}")
        print(f"Negative samples: {len(labels) - sum(labels)}")
        print("\nSample texts:")
        for i in range(min(3, len(texts))):
            print(f"Text: {texts[i][:100]}...")
            print(f"Label: {'Positive' if labels[i] == 1 else 'Negative'}")
            print("---")
    except Exception as e:
        print(f"Error loading IMDB dataset: {str(e)}")
    
    print("\n=== Loading Amazon Reviews Dataset ===")
    try:
        texts, labels = data_loader.load_amazon_reviews(language='en')
        print(f"Total samples: {len(texts)}")
        print(f"Positive samples: {sum(labels)}")
        print(f"Negative samples: {len(labels) - sum(labels)}")
        print("\nSample texts:")
        for i in range(min(3, len(texts))):
            print(f"Text: {texts[i][:100]}...")
            print(f"Label: {'Positive' if labels[i] == 1 else 'Negative'}")
            print("---")
    except Exception as e:
        print(f"Error loading Amazon Reviews dataset: {str(e)}")
    
    print("\n=== Loading Twitter Sentiment Dataset ===")
    try:
        texts, labels = data_loader.load_twitter_sentiment()
        print(f"Total samples: {len(texts)}")
        print(f"Positive samples: {sum(labels)}")
        print(f"Negative samples: {len(labels) - sum(labels)}")
        print("\nSample texts:")
        for i in range(min(3, len(texts))):
            print(f"Text: {texts[i][:100]}...")
            print(f"Label: {'Positive' if labels[i] == 1 else 'Negative'}")
            print("---")
    except Exception as e:
        print(f"Error loading Twitter Sentiment dataset: {str(e)}")

if __name__ == "__main__":
    main() 