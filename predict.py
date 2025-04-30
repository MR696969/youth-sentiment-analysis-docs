import torch
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
from dataset import SentimentDataLoader
import os
import json
from tqdm import tqdm

def load_model(model_path, model_name='bert-base-uncased', device=None):
    """Load a trained model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict_sentiment(model, tokenizer, text, device=None):
    """Predict sentiment for a single text"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tokenize text
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # Map prediction to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': {
            'negative': probabilities[0][0].item(),
            'positive': probabilities[0][1].item()
        }
    }

def batch_predict(model, tokenizer, texts, batch_size=32, device=None):
    """Predict sentiment for a batch of texts"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            confidences = probabilities.max(dim=1)[0].cpu().numpy()
        
        # Process results
        for j, (text, pred, conf) in enumerate(zip(batch_texts, predictions, confidences)):
            sentiment = "Positive" if pred == 1 else "Negative"
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': conf,
                'probabilities': {
                    'negative': probabilities[j][0].item(),
                    'positive': probabilities[j][1].item()
                }
            })
    
    return results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Predict sentiment for text')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pre-trained model name')
    parser.add_argument('--input', type=str, required=True,
                        help='Input text or path to file with texts')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.model_name, device)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Load input
    if os.path.isfile(args.input):
        # Check file extension
        if args.input.endswith('.txt'):
            with open(args.input, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        elif args.input.endswith('.json'):
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item['text'] if isinstance(item, dict) else item for item in data]
                else:
                    texts = [data['text']]
        elif args.input.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(args.input)
            texts = df['text'].tolist()
        else:
            raise ValueError(f"Unsupported file format: {args.input}")
    else:
        # Treat input as a single text
        texts = [args.input]
    
    print(f"Processing {len(texts)} texts...")
    
    # Predict sentiment
    if len(texts) == 1:
        result = predict_sentiment(model, tokenizer, texts[0], device)
        print("\nPrediction:")
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: Negative: {result['probabilities']['negative']:.4f}, Positive: {result['probabilities']['positive']:.4f}")
    else:
        results = batch_predict(model, tokenizer, texts, args.batch_size, device)
        
        # Print summary
        positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
        negative_count = sum(1 for r in results if r['sentiment'] == 'Negative')
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print("\nSummary:")
        print(f"Total texts: {len(results)}")
        print(f"Positive: {positive_count} ({positive_count/len(results)*100:.2f}%)")
        print(f"Negative: {negative_count} ({negative_count/len(results)*100:.2f}%)")
        print(f"Average confidence: {avg_confidence:.4f}")
        
        # Print a few examples
        print("\nExamples:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. Text: {result['text'][:100]}...")
            print(f"   Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.4f}")
    
    # Save results if output path is provided
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            if len(texts) == 1:
                json.dump(result, f, indent=2)
            else:
                json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 