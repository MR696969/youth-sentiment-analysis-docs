import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from odia_dataset import OdiaDatasetLoader
from dataset import SentimentDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from torch.cuda.amp import autocast, GradScaler
import os
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.parallel import DataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.quantization import quantize_dynamic
import time
from early_stopping import EarlyStopping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, optimizer, scheduler, device, scaler=None):
    """Train for one epoch with optimized processing"""
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def optimize_model(model):
    """Optimize model for inference"""
    # Quantize model
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def main():
    # Set device
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed()
    
    # Load dataset
    loader = OdiaDatasetLoader("C:/Users/Maheswar/Downloads/archive (2).zip")
    texts, labels = loader.load_dataset()
    
    # Use a smaller subset of data for faster training (30% of original data)
    subset_size = int(0.3 * len(texts))
    indices = np.random.permutation(len(texts))[:subset_size]
    texts = [texts[i] for i in indices]
    labels = labels[indices]
    
    num_labels = len(loader.label_encoder.classes_)
    
    # Initialize tokenizer and model (using a smaller model)
    model_name = "prajjwal1/bert-tiny"  # Much smaller model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = model.to(device)
    
    # Create datasets
    train_size = int(0.8 * len(texts))
    val_size = len(texts) - train_size
    
    train_texts = texts[:train_size]
    train_labels = labels[:train_size]
    val_texts = texts[train_size:]
    val_labels = labels[train_size:]
    
    # Create datasets with smaller max length for faster processing
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length=128)
    
    # Optimize batch size and workers for CPU
    batch_size = 32
    num_workers = 0  # Better for CPU training
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize optimizer with a larger learning rate for faster convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Use a simple learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # Initialize early stopping with shorter patience
    early_stopping = EarlyStopping(patience=2, min_delta=0.01)
    
    # Training loop with fewer epochs
    num_epochs = 3
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_results = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Train Loss: {train_results['loss']:.4f}, Accuracy: {train_results['accuracy']:.4f}")
        
        # Evaluate
        val_results = evaluate(model, val_loader, device)
        logger.info(f"Val Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy']:.4f}")
        
        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': loader.label_encoder,
                'val_loss': val_results['loss'],
                'val_accuracy': val_results['accuracy']
            }, 'best_odia_model_fast.pt')
        
        # Early stopping check
        if early_stopping(val_results['loss']):
            logger.info("Early stopping triggered")
            break
        
        scheduler.step()
    
    training_time = time.time() - start_time
    logger.info(f"Total training time: {training_time:.2f} seconds")

if __name__ == "__main__":
    main() 