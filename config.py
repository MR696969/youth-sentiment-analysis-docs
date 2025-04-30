import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Twitter API Configuration
TWITTER_CONFIG = {
    'api_key': os.getenv('TWITTER_API_KEY'),
    'api_secret': os.getenv('TWITTER_API_SECRET'),
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
}

def get_twitter_config():
    """Get Twitter configuration with validation"""
    required_keys = ['api_key', 'api_secret', 'bearer_token', 'access_token', 'access_token_secret']
    
    # Check if all required keys are present
    missing_keys = [key for key in required_keys if not TWITTER_CONFIG.get(key)]
    if missing_keys:
        raise ValueError(f"Missing required Twitter configuration keys: {', '.join(missing_keys)}")
    
    return TWITTER_CONFIG 