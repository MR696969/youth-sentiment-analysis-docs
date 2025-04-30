import pandas as pd
import os
import zipfile
from typing import Tuple, List
import re
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

class OdiaDatasetLoader:
    def __init__(self, zip_path: str):
        """
        Initialize the Odia dataset loader
        
        Args:
            zip_path (str): Path to the Odia dataset zip file
        """
        self.zip_path = zip_path
        self.data_dir = 'data/odia'
        self.label_encoder = LabelEncoder()
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def extract_dataset(self) -> str:
        """
        Extract the dataset from the zip file
        
        Returns:
            str: Path to the extracted dataset
        """
        print(f"Extracting dataset from {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        return self.data_dir
    
    def preprocess_odia_text(self, text: str) -> str:
        """
        Preprocess Odia text
        
        Args:
            text (str): Raw Odia text
            
        Returns:
            str: Preprocessed text
        """
        # Remove special characters and extra spaces
        text = re.sub(r'[^\u0B00-\u0B7F\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def load_dataset(self) -> Tuple[List[str], List[int]]:
        """
        Load and preprocess the Odia dataset
        
        Returns:
            Tuple[List[str], List[int]]: Texts and encoded labels
        """
        # Extract the dataset
        data_dir = self.extract_dataset()
        
        # Find the CSV file in the extracted directory
        csv_file = None
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                csv_file = os.path.join(data_dir, file)
                break
        
        if not csv_file:
            raise FileNotFoundError("No CSV file found in the dataset")
        
        # Load the dataset
        df = pd.read_csv(csv_file)
        
        # Print dataset information
        print("\nDataset Information:")
        print(f"Total samples: {len(df)}")
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        
        # Process texts and labels
        texts = df['headings'].apply(self.preprocess_odia_text).values
        
        # Encode labels to integers
        labels = self.label_encoder.fit_transform(df['label'])
        
        print("\nLabel encoding mapping:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"{label} -> {i}")
        
        return texts, labels

def main():
    # Initialize the loader with your zip file path
    loader = OdiaDatasetLoader("C:/Users/Maheswar/Downloads/archive (2).zip")
    
    # Load the dataset
    texts, labels = loader.load_dataset()
    
    print(f"\nLoaded {len(texts)} samples")
    
    # Print some sample texts
    print("\nSample texts:")
    for i in range(min(3, len(texts))):
        print(f"Text {i+1}: {texts[i]}")
        print(f"Label: {loader.label_encoder.inverse_transform([labels[i]])[0]}\n")

if __name__ == "__main__":
    main() 