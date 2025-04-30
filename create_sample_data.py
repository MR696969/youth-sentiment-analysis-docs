from dataset import SentimentDataLoader
import os

def main():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Initialize data loader
    data_loader = SentimentDataLoader()
    
    # Save sample data
    data_loader.save_sample_data('data/sample_data.csv')
    
    # Load the data to verify
    texts, labels = data_loader.load_data('data/sample_data.csv')
    print(f"Loaded {len(texts)} samples")
    print("\nSample data:")
    for text, label in zip(texts[:5], labels[:5]):
        print(f"Text: {text}")
        print(f"Label: {label}")
        print("---")

if __name__ == "__main__":
    main() 