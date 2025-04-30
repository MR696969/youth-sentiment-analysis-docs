from flask import Flask, render_template, request, jsonify, redirect
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import json
from dataset import SentimentDataLoader
import argparse
import tweepy
import praw
from twitter_config import get_twitter_config
from reddit_config import get_reddit_config
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.middleware.shared_data import SharedDataMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Global variables
model = None
tokenizer = None
device = None
twitter_client = None
reddit_client = None

@app.before_request
def before_request():
    """Handle pre-request processing"""
    if request.method == 'OPTIONS':
        return '', 200
    # Force HTTP if HTTPS is not available
    if request.url.startswith('https://'):
        return redirect(request.url.replace('https://', 'http://', 1))

@app.after_request
def after_request(response):
    """Handle post-request processing"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.errorhandler(400)
def bad_request(error):
    """Handle bad requests"""
    logger.error(f"Bad request: {error}")
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    logger.error(f"Not found: {error}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def initialize_twitter():
    """Initialize Twitter API client"""
    global twitter_client
    try:
        config = get_twitter_config()
        auth = tweepy.OAuthHandler(config['api_key'], config['api_secret'])
        auth.set_access_token(config['access_token'], config['access_token_secret'])
        twitter_client = tweepy.API(auth)
        return True
    except Exception as e:
        print(f"Error initializing Twitter client: {str(e)}")
        return False

def initialize_reddit():
    """Initialize Reddit API client"""
    global reddit_client
    try:
        config = get_reddit_config()
        reddit_client = praw.Reddit(
            client_id=config['client_id'],
            client_secret=config['client_secret'],
            user_agent=config['user_agent']
        )
        return True
    except Exception as e:
        print(f"Error initializing Reddit client: {str(e)}")
        return False

def load_model(model_path=None, model_name='bert-base-uncased'):
    """Load a trained model"""
    global model, tokenizer, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize model and tokenizer
        logger.info("Loading model and tokenizer...")
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Move model to device
        model.to(device)
        model.eval()
        
        logger.info("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_sentiment(text):
    """Predict sentiment for a single text"""
    global model, tokenizer, device
    
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

@app.route('/')
def index():
    try:
        return render_template('sentiment.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return "Error loading page. Please check the server logs.", 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = predict_sentiment(text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    data = request.json
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    results = []
    for text in texts:
        result = predict_sentiment(text)
        results.append(result)
    
    # Calculate summary
    positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
    negative_count = sum(1 for r in results if r['sentiment'] == 'Negative')
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    return jsonify({
        'results': results,
        'summary': {
            'total': len(results),
            'positive': positive_count,
            'negative': negative_count,
            'positive_percentage': positive_count / len(results) * 100,
            'negative_percentage': negative_count / len(results) * 100,
            'average_confidence': avg_confidence
        }
    })

@app.route('/analyze_twitter', methods=['POST'])
def analyze_twitter():
    try:
        data = request.json
        query = data.get('query', '')
        count = data.get('count', 10)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        if not twitter_client:
            return jsonify({'error': 'Twitter client not initialized'}), 500
        
        # Search tweets
        tweets = twitter_client.search_tweets(q=query, count=count, tweet_mode='extended')
        
        results = []
        for tweet in tweets:
            text = tweet.full_text
            result = predict_sentiment(text)
            result['tweet_id'] = tweet.id_str
            result['created_at'] = tweet.created_at.isoformat()
            result['user'] = tweet.user.screen_name
            results.append(result)
        
        # Calculate summary
        positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
        negative_count = sum(1 for r in results if r['sentiment'] == 'Negative')
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'positive': positive_count,
                'negative': negative_count,
                'positive_percentage': positive_count / len(results) * 100 if results else 0,
                'negative_percentage': negative_count / len(results) * 100 if results else 0,
                'average_confidence': avg_confidence
            }
        })
    except Exception as e:
        logger.error(f"Error in analyze_twitter endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_reddit', methods=['POST'])
def analyze_reddit():
    try:
        data = request.json
        subreddit_name = data.get('subreddit', '')
        limit = data.get('limit', 10)
        
        if not subreddit_name:
            return jsonify({'error': 'No subreddit provided'}), 400
        
        if not reddit_client:
            return jsonify({'error': 'Reddit client not initialized'}), 500
        
        # Get subreddit posts
        subreddit = reddit_client.subreddit(subreddit_name)
        posts = subreddit.hot(limit=limit)
        
        results = []
        for post in posts:
            text = post.title + " " + post.selftext
            result = predict_sentiment(text)
            result['post_id'] = post.id
            result['created_at'] = post.created_utc
            result['author'] = post.author.name if post.author else '[deleted]'
            result['title'] = post.title
            results.append(result)
        
        # Calculate summary
        positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
        negative_count = sum(1 for r in results if r['sentiment'] == 'Negative')
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'positive': positive_count,
                'negative': negative_count,
                'positive_percentage': positive_count / len(results) * 100 if results else 0,
                'negative_percentage': negative_count / len(results) * 100 if results else 0,
                'average_confidence': avg_confidence
            }
        })
    except Exception as e:
        logger.error(f"Error in analyze_reddit endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Run sentiment analysis web app')
        parser.add_argument('--model_path', type=str, default=None,
                            help='Path to the trained model')
        parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                            help='Pre-trained model name')
        parser.add_argument('--port', type=int, default=5000,
                            help='Port to run the web app on')
        parser.add_argument('--host', type=str, default='127.0.0.1',
                            help='Host to run the web app on')
        args = parser.parse_args()
        
        # Load model
        logger.info("Loading model...")
        load_model(args.model_path, args.model_name)
        
        # Initialize Twitter client
        logger.info("Initializing Twitter client...")
        if not initialize_twitter():
            logger.warning("Twitter client initialization failed. Twitter features will be disabled.")
        
        # Initialize Reddit client
        logger.info("Initializing Reddit client...")
        if not initialize_reddit():
            logger.warning("Reddit client initialization failed. Reddit features will be disabled.")
        
        # Create templates directory if it doesn't exist
        if not os.path.exists('templates'):
            os.makedirs('templates')
        
        # Create HTML template
        with open('templates/sentiment.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .result-card {
            margin-top: 1rem;
            display: none;
        }
        .sentiment-positive {
            color: green;
        }
        .sentiment-negative {
            color: red;
        }
        .confidence-bar {
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .confidence-fill {
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.5s;
        }
        .twitter-results, .reddit-results {
            margin-top: 2rem;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Sentiment Analysis</h1>
        
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Single Text Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="text-input" class="form-label">Enter text:</label>
                            <textarea class="form-control" id="text-input" rows="4"></textarea>
                        </div>
                        <button class="btn btn-primary" id="analyze-btn">Analyze</button>
                        
                        <div class="card result-card" id="result-card">
                            <div class="card-body">
                                <h5 class="card-title">Result</h5>
                                <p class="card-text">
                                    <strong>Sentiment:</strong> <span id="sentiment-result"></span>
                                </p>
                                <p class="card-text">
                                    <strong>Confidence:</strong> <span id="confidence-result"></span>
                                </p>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" id="confidence-bar-fill"></div>
                                </div>
                                <p class="card-text mt-3">
                                    <strong>Probabilities:</strong>
                                </p>
                                <p class="card-text">
                                    Negative: <span id="negative-prob"></span>
                                </p>
                                <p class="card-text">
                                    Positive: <span id="positive-prob"></span>
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Twitter Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="twitter-query" class="form-label">Search query:</label>
                            <input type="text" class="form-control" id="twitter-query" placeholder="Enter search query">
                        </div>
                        <div class="mb-3">
                            <label for="tweet-count" class="form-label">Number of tweets:</label>
                            <input type="number" class="form-control" id="tweet-count" value="10" min="1" max="100">
                        </div>
                        <button class="btn btn-primary" id="twitter-analyze-btn">Analyze Tweets</button>
                        
                        <div class="twitter-results" id="twitter-results">
                            <h5 class="mt-4">Summary</h5>
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Total tweets:</strong> <span id="twitter-total"></span></p>
                                    <p><strong>Positive:</strong> <span id="twitter-positive"></span> (<span id="twitter-positive-percentage"></span>%)</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Negative:</strong> <span id="twitter-negative"></span> (<span id="twitter-negative-percentage"></span>%)</p>
                                    <p><strong>Average confidence:</strong> <span id="twitter-avg-confidence"></span></p>
                                </div>
                            </div>
                            
                            <h5 class="mt-4">Results</h5>
                            <div id="twitter-results-list"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Reddit Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="subreddit" class="form-label">Subreddit name:</label>
                            <input type="text" class="form-control" id="subreddit" placeholder="Enter subreddit name">
                        </div>
                        <div class="mb-3">
                            <label for="post-count" class="form-label">Number of posts:</label>
                            <input type="number" class="form-control" id="post-count" value="10" min="1" max="100">
                        </div>
                        <button class="btn btn-primary" id="reddit-analyze-btn">Analyze Posts</button>
                        
                        <div class="reddit-results" id="reddit-results">
                            <h5 class="mt-4">Summary</h5>
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Total posts:</strong> <span id="reddit-total"></span></p>
                                    <p><strong>Positive:</strong> <span id="reddit-positive"></span> (<span id="reddit-positive-percentage"></span>%)</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Negative:</strong> <span id="reddit-negative"></span> (<span id="reddit-negative-percentage"></span>%)</p>
                                    <p><strong>Average confidence:</strong> <span id="reddit-avg-confidence"></span></p>
                                </div>
                            </div>
                            
                            <h5 class="mt-4">Results</h5>
                            <div id="reddit-results-list"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('analyze-btn').addEventListener('click', function() {
            const text = document.getElementById('text-input').value.trim();
            if (!text) {
                alert('Please enter some text');
                return;
            }
            
            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                const resultCard = document.getElementById('result-card');
                const sentimentResult = document.getElementById('sentiment-result');
                const confidenceResult = document.getElementById('confidence-result');
                const confidenceBarFill = document.getElementById('confidence-bar-fill');
                const negativeProb = document.getElementById('negative-prob');
                const positiveProb = document.getElementById('positive-prob');
                
                sentimentResult.textContent = data.sentiment;
                sentimentResult.className = data.sentiment === 'Positive' ? 'sentiment-positive' : 'sentiment-negative';
                
                const confidence = (data.confidence * 100).toFixed(2);
                confidenceResult.textContent = confidence + '%';
                confidenceBarFill.style.width = confidence + '%';
                
                negativeProb.textContent = (data.probabilities.negative * 100).toFixed(2) + '%';
                positiveProb.textContent = (data.probabilities.positive * 100).toFixed(2) + '%';
                
                resultCard.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while analyzing the text');
            });
        });
        
        document.getElementById('twitter-analyze-btn').addEventListener('click', function() {
            const query = document.getElementById('twitter-query').value.trim();
            const count = parseInt(document.getElementById('tweet-count').value);
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            fetch('/analyze_twitter', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query, count: count })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                const twitterResults = document.getElementById('twitter-results');
                const twitterTotal = document.getElementById('twitter-total');
                const twitterPositive = document.getElementById('twitter-positive');
                const twitterNegative = document.getElementById('twitter-negative');
                const twitterPositivePercentage = document.getElementById('twitter-positive-percentage');
                const twitterNegativePercentage = document.getElementById('twitter-negative-percentage');
                const twitterAvgConfidence = document.getElementById('twitter-avg-confidence');
                const twitterResultsList = document.getElementById('twitter-results-list');
                
                twitterTotal.textContent = data.summary.total;
                twitterPositive.textContent = data.summary.positive;
                twitterNegative.textContent = data.summary.negative;
                twitterPositivePercentage.textContent = data.summary.positive_percentage.toFixed(2);
                twitterNegativePercentage.textContent = data.summary.negative_percentage.toFixed(2);
                twitterAvgConfidence.textContent = (data.summary.average_confidence * 100).toFixed(2) + '%';
                
                twitterResultsList.innerHTML = '';
                data.results.forEach((result, index) => {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'card mb-2';
                    resultItem.innerHTML = `
                        <div class="card-body">
                            <h6 class="card-title">Tweet ${index + 1} by @${result.user}:</h6>
                            <p class="card-text">${result.text}</p>
                            <p class="card-text">
                                <strong>Sentiment:</strong> 
                                <span class="${result.sentiment === 'Positive' ? 'sentiment-positive' : 'sentiment-negative'}">
                                    ${result.sentiment}
                                </span>
                                (${(result.confidence * 100).toFixed(2)}%)
                            </p>
                            <p class="card-text">
                                <small class="text-muted">Posted on ${new Date(result.created_at).toLocaleString()}</small>
                            </p>
                        </div>
                    `;
                    twitterResultsList.appendChild(resultItem);
                });
                
                twitterResults.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while analyzing tweets');
            });
        });
        
        document.getElementById('reddit-analyze-btn').addEventListener('click', function() {
            const subreddit = document.getElementById('subreddit').value.trim();
            const count = parseInt(document.getElementById('post-count').value);
            
            if (!subreddit) {
                alert('Please enter a subreddit name');
                return;
            }
            
            fetch('/analyze_reddit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ subreddit: subreddit, limit: count })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                const redditResults = document.getElementById('reddit-results');
                const redditTotal = document.getElementById('reddit-total');
                const redditPositive = document.getElementById('reddit-positive');
                const redditNegative = document.getElementById('reddit-negative');
                const redditPositivePercentage = document.getElementById('reddit-positive-percentage');
                const redditNegativePercentage = document.getElementById('reddit-negative-percentage');
                const redditAvgConfidence = document.getElementById('reddit-avg-confidence');
                const redditResultsList = document.getElementById('reddit-results-list');
                
                redditTotal.textContent = data.summary.total;
                redditPositive.textContent = data.summary.positive;
                redditNegative.textContent = data.summary.negative;
                redditPositivePercentage.textContent = data.summary.positive_percentage.toFixed(2);
                redditNegativePercentage.textContent = data.summary.negative_percentage.toFixed(2);
                redditAvgConfidence.textContent = (data.summary.average_confidence * 100).toFixed(2) + '%';
                
                redditResultsList.innerHTML = '';
                data.results.forEach((result, index) => {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'card mb-2';
                    resultItem.innerHTML = `
                        <div class="card-body">
                            <h6 class="card-title">Post ${index + 1} by u/${result.author}:</h6>
                            <p class="card-text"><strong>Title:</strong> ${result.title}</p>
                            <p class="card-text">
                                <strong>Sentiment:</strong> 
                                <span class="${result.sentiment === 'Positive' ? 'sentiment-positive' : 'sentiment-negative'}">
                                    ${result.sentiment}
                                </span>
                                (${(result.confidence * 100).toFixed(2)}%)
                            </p>
                            <p class="card-text">
                                <small class="text-muted">Posted on ${new Date(result.created_at * 1000).toLocaleString()}</small>
                            </p>
                        </div>
                    `;
                    redditResultsList.appendChild(resultItem);
                });
                
                redditResults.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while analyzing Reddit posts');
            });
        });
    </script>
</body>
</html>
        ''')
        
        # Run the app
        logger.info(f"Starting web app on {args.host}:{args.port}...")
        app.run(host=args.host, port=args.port, debug=True, use_reloader=True)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 