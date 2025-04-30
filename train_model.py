import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from dataset import SentimentDataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import argparse
import pandas as pd
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import logging
from torch.cuda.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate(model, val_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_metrics(train_metrics, val_metrics, save_path=None):
    """Plot training and validation metrics"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(train_metrics[metric], label='Train')
        ax.plot(val_metrics[metric], label='Validation')
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    logger.info(f"Loading dataset from {file_path}")
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Assuming the dataset has 'text' and 'label' columns
    # If column names are different, adjust accordingly
    texts = df['text'].values
    labels = df['label'].values
    
    # Convert labels to 0 and 1 if they're not already
    if labels.dtype != np.int64:
        labels = (labels == 'positive').astype(int)
    
    return texts, labels

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    """Train the model with mixed precision"""
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    accumulation_steps = 2
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        val_preds = []
        val_labels_all = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                val_steps += 1
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels_all.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / val_steps
        val_accuracy = accuracy_score(val_labels_all, val_preds)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            logger.info("Saved best model")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    file_path = "data/IMDB Dataset.csv"
    logger.info(f"Loading dataset from {file_path}")
    
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully with {len(df)} samples")
        
        texts = df['review'].values
        labels = df['sentiment'].values
        labels = (labels == 'positive').astype(int)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.1, random_state=42
        )
        logger.info(f"Split data into {len(train_texts)} training and {len(val_texts)} validation samples")
        
        # Initialize tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        model.to(device)
        logger.info("Model and tokenizer initialized")
        
        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
        
        # Create dataloaders with larger batch size
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
        logger.info("Data loaders created")
        
        # Train model
        logger.info("Starting model training...")
        train_model(model, train_loader, val_loader, device)
        
        # Save final model
        torch.save(model.state_dict(), 'final_model.pt')
        logger.info("Training completed and model saved")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 