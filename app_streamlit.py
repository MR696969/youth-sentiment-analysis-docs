import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tweepy
import praw
from twitter_config import get_twitter_config
from reddit_config import get_reddit_config
from facebook_config import get_facebook_config
import logging
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from auth import login_user, register_user, is_logged_in, logout_user, facebook_login
import googleapiclient.discovery
import re
import requests
import facebook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables
model = None
tokenizer = None
device = None
twitter_client = None
reddit_client = None
youtube_client = None
facebook_client = None

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Youth Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/youth-sentiment-analysis',
        'Report a bug': 'https://github.com/yourusername/youth-sentiment-analysis/issues',
        'About': 'Youth Sentiment Analysis Dashboard'
    }
)

# Custom CSS for light theme
st.markdown("""
    <style>
        /* Base styles */
        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF;
        }

        [data-testid="stSidebar"] {
            background-color: #F0F2F6;
        }

        [data-testid="stHeader"] {
            background-color: #FFFFFF;
        }

        /* Text colors */
        .st-emotion-cache-ue6h4q {
            color: #262730 !important;
        }

        .st-emotion-cache-16idsys p {
            color: #262730 !important;
        }

        .st-emotion-cache-1788y8l {
            color: #262730 !important;
        }

        /* Sidebar navigation */
        .st-emotion-cache-16txtl3 {
            color: #262730 !important;
        }

        /* Form elements */
        .stTextInput input {
            background-color: #FFFFFF !important;
            color: #262730 !important;
            border-color: #E0E0E0 !important;
        }

        .stTextInput label {
            color: #262730 !important;
        }

        /* Login button specific styling */
        div[data-testid="stForm"] button[data-testid="baseButton-secondary"],
        div[data-testid="stForm"] button[data-testid="baseButton-secondary"]:hover,
        div[data-testid="stForm"] button[data-testid="baseButton-secondary"]:focus,
        div[data-testid="stForm"] button[data-testid="baseButton-secondary"]:active {
            background-color: #DC3545 !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }

        /* Other buttons */
        .stButton button:not([data-testid="baseButton-secondary"]) {
            background-color: #FFFFFF !important;
            color: #262730 !important;
            border: 1px solid #E0E0E0 !important;
        }

        .stButton button:not([data-testid="baseButton-secondary"]):hover {
            background-color: #F0F2F6 !important;
            border: 1px solid #E0E0E0 !important;
        }

        /* Headers and text */
        h1, h2, h3, h4, h5, h6 {
            color: #262730 !important;
        }

        p {
            color: #262730 !important;
        }

        /* Radio buttons */
        .st-emotion-cache-1qg05tj {
            color: #262730 !important;
        }

        /* Selectbox */
        .stSelectbox label {
            color: #262730 !important;
        }

        .stSelectbox select {
            background-color: #FFFFFF !important;
            color: #262730 !important;
            border-color: #E0E0E0 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            color: #262730 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #F0F2F6 !important;
        }

        /* Success/Error messages */
        .stSuccess {
            background-color: #E6F4EA !important;
            color: #1E4620 !important;
        }

        .stError {
            background-color: #FCE8E6 !important;
            color: #C5221F !important;
        }

        .stWarning {
            background-color: #FFF3E0 !important;
            color: #E65100 !important;
        }

        .stInfo {
            background-color: #E8F0FE !important;
            color: #1A73E8 !important;
        }

        /* Force all text to be dark */
        div {
            color: #262730 !important;
        }

        /* Markdown text */
        .element-container {
            color: #262730 !important;
        }

        /* Additional sidebar elements */
        [data-testid="stSidebarNav"] {
            color: #262730 !important;
        }

        .st-emotion-cache-16idsys {
            color: #262730 !important;
        }

        .st-emotion-cache-pkbazv {
            color: #262730 !important;
        }

        /* Radio button text */
        .st-emotion-cache-1umgz6j {
            color: #262730 !important;
        }

        .st-emotion-cache-1umgz6j span {
            color: #262730 !important;
        }
    </style>
""", unsafe_allow_html=True)

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
        logger.error(f"Error initializing Twitter client: {str(e)}")
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
        logger.error(f"Error initializing Reddit client: {str(e)}")
        return False

def initialize_youtube():
    """Initialize YouTube API client"""
    global youtube_client
    try:
        api_key = "AIzaSyAvR1Smdb1N2BvrRY9Hy9qJ9VOHlJlb0Xw"
        youtube_client = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
        return True
    except Exception as e:
        logger.error(f"Error initializing YouTube client: {str(e)}")
        return False

def initialize_facebook():
    """Initialize Facebook API client"""
    global facebook_client
    try:
        config = get_facebook_config()
        st.write("Debug: Facebook Config:", config)  # Debug line
        facebook_client = facebook.GraphAPI(access_token=config['access_token'])
        
        # Test the API connection with basic profile
        try:
            test_response = facebook_client.get_object('me', fields='id,name')
            st.write("Debug: API Test Response:", test_response)  # Debug line
        except Exception as e:
            st.error(f"Debug: Basic API test failed: {str(e)}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error initializing Facebook client: {str(e)}")
        st.error(f"Debug: Facebook initialization error: {str(e)}")  # Debug line
        return False

def predict_sentiment(text):
    """Predict sentiment for a single text"""
    global model, tokenizer, device
    
    # Preprocess text
    text = text.lower().strip()
    
    # Handle emojis and special characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Check for positive indicators
    positive_indicators = [
        'nice', 'good', 'great', 'excellent', 'amazing', 'wonderful', 'beautiful',
        'love', 'favorite', 'best', 'awesome', 'fantastic', 'perfect', 'wow',
        'melodious', 'unique', 'underrated', 'hats off', 'thank', 'thanks',
        'evergreen', 'like', 'listening', 'watching', 'enjoy', 'enjoying'
    ]
    
    # Check for negative indicators
    negative_indicators = [
        'bad', 'worst', 'terrible', 'awful', 'hate', 'dislike', 'poor',
        'boring', 'waste', 'useless', 'wrong', 'false', 'fake', 'stupid',
        'not', 'no', 'never', 'nothing', 'nobody', 'none', 'neither', 'nor'
    ]
    
    # Count positive and negative indicators
    positive_count = sum(1 for word in positive_indicators if word in text)
    negative_count = sum(1 for word in negative_indicators if word in text)
    
    # If there are clear indicators, use them
    if positive_count > negative_count:
        return {
            'text': text,
            'sentiment': "Positive",
            'confidence': 0.8,
            'probabilities': {
                'negative': 0.2,
                'positive': 0.8
            }
        }
    elif negative_count > positive_count:
        return {
            'text': text,
            'sentiment': "Negative",
            'confidence': 0.8,
            'probabilities': {
                'negative': 0.8,
                'positive': 0.2
            }
        }
    
    # If no clear indicators, use the model
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
    
    # Adjust sentiment threshold
    positive_prob = probabilities[0][1].item()
    negative_prob = probabilities[0][0].item()
    
    # Use a more lenient threshold for positive sentiment
    if positive_prob > 0.35:  # Lowered threshold for positive sentiment
        sentiment = "Positive"
        confidence = positive_prob
    else:
        sentiment = "Negative"
        confidence = negative_prob
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': {
            'negative': negative_prob,
            'positive': positive_prob
        }
    }

def show_privacy_policy():
    st.title("Privacy Policy")
    st.markdown("Last updated: March 2024")
    
    st.header("1. Introduction")
    st.write("Welcome to Youth Sentiment Analysis. We respect your privacy and are committed to protecting your personal data. This privacy policy will inform you about how we handle your data when you use our application.")
    
    st.header("2. Data Collection")
    st.write("Our application only collects and processes public data for sentiment analysis purposes. This includes:")
    st.markdown("""
    - Public social media posts and comments
    - Public page metadata
    - Text content for sentiment analysis
    """)
    
    st.header("3. Data Usage")
    st.write("We use the collected data solely for:")
    st.markdown("""
    - Performing sentiment analysis
    - Generating insights and reports
    - Improving our analysis algorithms
    """)
    
    st.header("4. Data Storage and Security")
    st.write("We do not store, share, or sell any personal data. All analysis is performed in real-time, and no personal information is retained after processing.")
    
    st.header("5. Your Rights")
    st.write("You have the right to:")
    st.markdown("""
    - Request access to your data
    - Request deletion of your data
    - Opt-out of data collection
    """)
    
    st.header("Contact Us")
    st.info("If you have any questions about this Privacy Policy or would like to exercise your rights, please contact us at: maheswarrana49@gmail.com")

def show_data_deletion():
    st.title("Data Deletion Instructions")
    
    st.header("How to Request Data Deletion")
    st.write("We respect your right to have your data deleted. Here's how you can request data deletion:")
    
    st.subheader("1. Email Request")
    st.write("Send an email to maheswarrana49@gmail.com with the subject line 'Data Deletion Request' and include:")
    st.markdown("""
    - Your name
    - The email address associated with your account
    - The social media platforms you want your data deleted from
    - Any specific data you want deleted
    """)
    
    st.subheader("2. Processing Time")
    st.write("We will process your request within 30 days and send you a confirmation email once completed.")
    
    st.subheader("3. What Gets Deleted")
    st.write("When you request data deletion, we will:")
    st.markdown("""
    - Remove all stored data associated with your account
    - Delete any analysis results linked to your data
    - Remove your account information from our systems
    """)
    
    st.subheader("4. Data Retention")
    st.write("Please note that we do not store any personal data after processing. All analysis is performed in real-time, and no personal information is retained.")
    
    st.info("For immediate assistance, please contact us at: maheswarrana49@gmail.com")

def main():
    # Login/Register section
    if not is_logged_in():
        st.sidebar.title("Login/Register")
        tab1, tab2 = st.sidebar.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    if login_user(email, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
        
        with tab2:
            with st.form("register_form"):
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register = st.form_submit_button("Register")
                
                if register:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif register_user(new_email, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed. Email might already be in use.")
        
        # Show privacy policy and data deletion pages in sidebar when not logged in
        st.sidebar.markdown("---")
        if st.sidebar.button("Privacy Policy"):
            show_privacy_policy()
        if st.sidebar.button("Data Deletion"):
            show_data_deletion()
        return
    
    # Main app content (only shown when logged in)
    st.title("Youth Sentiment Analysis Dashboard")
    st.markdown(f"Welcome, {st.session_state['username']}!")
    
    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        logout_user()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Text Analysis", "Social Media Analysis", "Batch Analysis", "About"])
    
    # Initialize model and clients
    if model is None:
        with st.spinner("Loading model..."):
            load_model()
    
    if twitter_client is None:
        with st.spinner("Initializing Twitter client..."):
            if not initialize_twitter():
                st.warning("Twitter client initialization failed. Twitter features will be disabled.")
    
    if reddit_client is None:
        with st.spinner("Initializing Reddit client..."):
            if not initialize_reddit():
                st.warning("Reddit client initialization failed. Reddit features will be disabled.")
    
    if youtube_client is None:
        with st.spinner("Initializing YouTube client..."):
            if not initialize_youtube():
                st.warning("YouTube client initialization failed. YouTube features will be disabled.")
    
    if facebook_client is None:
        with st.spinner("Initializing Facebook client..."):
            if not initialize_facebook():
                st.warning("Facebook client initialization failed. Facebook features will be disabled.")
    
    if page == "Text Analysis":
        st.header("Text Sentiment Analysis")
        
        # Text input
        text_input = st.text_area("Enter text to analyze:", height=150)
        
        if st.button("Analyze Text"):
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    result = predict_sentiment(text_input)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Sentiment", result['sentiment'].upper())
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    with col3:
                        st.metric("Compound Score", f"{result['confidence']:.2f}")
                    
                    # Sentiment distribution
                    sentiment_data = {
                        'Sentiment': ['Positive', 'Negative'],
                        'Score': [result['probabilities']['positive'], result['probabilities']['negative']]
                    }
                    df = pd.DataFrame(sentiment_data)
                    
                    fig = px.bar(df, x='Sentiment', y='Score', 
                               title='Sentiment Distribution',
                               color='Sentiment',
                               color_discrete_map={
                                   'Positive': 'green',
                                   'Negative': 'red'
                               })
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze.")
    
    elif page == "Social Media Analysis":
        st.header("Social Media Analysis")
        
        # Platform selection
        platform = st.selectbox(
            "Select Platform",
            ["Twitter", "Reddit", "YouTube", "Facebook"]
        )
        
        if platform == "Twitter":
            query = st.text_input("Enter search query:")
            limit = st.slider("Number of tweets to analyze", 10, 100, 50)
            
            if st.button("Analyze Twitter"):
                if query:
                    with st.spinner("Collecting and analyzing tweets..."):
                        try:
                            # Search tweets
                            tweets = twitter_client.search_tweets(q=query, count=limit, tweet_mode='extended')
                            
                            results = []
                            for tweet in tweets:
                                text = tweet.full_text
                                result = predict_sentiment(text)
                                result['tweet_id'] = tweet.id_str
                                result['created_at'] = tweet.created_at.isoformat()
                                result['user'] = tweet.user.screen_name
                                results.append(result)
                            
                            # Display results
                            st.subheader("Analysis Results")
                            
                            # Sentiment distribution
                            sentiment_counts = pd.DataFrame(results)['sentiment'].value_counts()
                            fig = px.pie(values=sentiment_counts.values, 
                                       names=sentiment_counts.index,
                                       title='Sentiment Distribution')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display tweets
                            st.subheader("Analyzed Tweets")
                            for result in results:
                                st.write(f"**Tweet by @{result['user']}:**")
                                st.write(result['text'])
                                st.write(f"**Sentiment:** {result['sentiment']} ({result['confidence']*100:.1f}%)")
                                st.write(f"**Posted on:** {result['created_at']}")
                                st.write("---")
                        except Exception as e:
                            st.error(f"Error analyzing tweets: {str(e)}")
                else:
                    st.warning("Please enter a search query.")
        
        elif platform == "Reddit":
            subreddit = st.text_input("Enter subreddit name (without r/):")
            limit = st.slider("Number of posts to analyze", 10, 100, 50)
            
            if st.button("Analyze Reddit"):
                if subreddit:
                    with st.spinner("Collecting and analyzing Reddit posts..."):
                        try:
                            # Get subreddit posts
                            subreddit_obj = reddit_client.subreddit(subreddit)
                            posts = subreddit_obj.hot(limit=limit)
                            
                            results = []
                            for post in posts:
                                text = post.title + " " + post.selftext
                                result = predict_sentiment(text)
                                result['post_id'] = post.id
                                result['created_at'] = post.created_utc
                                result['author'] = post.author.name if post.author else '[deleted]'
                                result['title'] = post.title
                                results.append(result)
                            
                            # Display results
                            st.subheader("Analysis Results")
                            
                            # Sentiment distribution
                            sentiment_counts = pd.DataFrame(results)['sentiment'].value_counts()
                            fig = px.pie(values=sentiment_counts.values, 
                                       names=sentiment_counts.index,
                                       title='Sentiment Distribution')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display posts
                            st.subheader("Analyzed Posts")
                            for result in results:
                                st.write(f"**Post by u/{result['author']}:**")
                                st.write(f"**Title:** {result['title']}")
                                st.write(f"**Sentiment:** {result['sentiment']} ({result['confidence']*100:.1f}%)")
                                st.write(f"**Posted on:** {datetime.fromtimestamp(result['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
                                st.write("---")
                        except Exception as e:
                            st.error(f"Error analyzing Reddit posts: {str(e)}")
                else:
                    st.warning("Please enter a subreddit name.")
        
        elif platform == "YouTube":
            video_url = st.text_input("Enter YouTube video URL:")
            limit = st.slider("Number of comments to analyze", 10, 100, 50)
            
            if st.button("Analyze YouTube"):
                if video_url:
                    with st.spinner("Collecting and analyzing YouTube comments..."):
                        try:
                            # Extract video ID from URL
                            if "youtu.be" in video_url:
                                video_id = video_url.split("/")[-1].split("?")[0]
                            elif "youtube.com" in video_url:
                                video_id = video_url.split("v=")[-1].split("&")[0]
                            else:
                                st.error("Invalid YouTube URL format. Please use a valid YouTube video URL.")
                                st.stop()
                            
                            # Get video comments
                            comments = []
                            for comment in youtube_client.commentThreads().list(
                                part="snippet",
                                videoId=video_id,
                                maxResults=limit,
                                textFormat="plainText"
                            ).execute()["items"]:
                                comment_text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                                comment_author = comment["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                                comment_date = comment["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
                                comments.append({
                                    "text": comment_text,
                                    "author": comment_author,
                                    "date": comment_date
                                })
                            
                            if not comments:
                                st.warning("No comments found for this video.")
                                st.stop()
                            
                            results = []
                            for comment in comments:
                                result = predict_sentiment(comment["text"])
                                result['comment_id'] = comment["text"][:50] + "..."  # Use first 50 chars as ID
                                result['created_at'] = comment["date"]
                                result['user'] = comment["author"]
                                results.append(result)
                            
                            # Display results
                            st.subheader("Analysis Results")
                            
                            # Sentiment distribution
                            sentiment_counts = pd.DataFrame(results)['sentiment'].value_counts()
                            fig = px.pie(values=sentiment_counts.values, 
                                       names=sentiment_counts.index,
                                       title='Sentiment Distribution')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display comments
                            st.subheader("Analyzed Comments")
                            for result in results:
                                st.write(f"**Comment by {result['user']}:**")
                                st.write(result['text'])
                                st.write(f"**Sentiment:** {result['sentiment']} ({result['confidence']*100:.1f}%)")
                                st.write(f"**Posted on:** {result['created_at']}")
                                st.write("---")
                        except Exception as e:
                            st.error(f"Error analyzing YouTube comments: {str(e)}")
                else:
                    st.warning("Please enter a YouTube video URL.")
        
        elif platform == "Facebook":
            page_url = st.text_input("Enter Facebook Page URL:")
            limit = st.slider("Number of posts to analyze", 10, 100, 50)
            
            if st.button("Analyze Facebook"):
                if page_url:
                    try:
                        # Check if Facebook client is properly initialized
                        if facebook_client is None:
                            st.error("Facebook API is not properly configured. Please check your access token and permissions.")
                            st.stop()
                        
                        # Debug: Show the input URL
                        st.write("Debug: Input URL:", page_url)  # Debug line
                        
                        # Extract page ID or username from URL
                        if "facebook.com/" in page_url:
                            # Extract the identifier from the URL
                            identifier = page_url.split("facebook.com/")[-1].split("?")[0].split("/")[0]
                            st.write("Debug: Extracted identifier:", identifier)  # Debug line
                            
                            # Try to get page information
                            try:
                                st.write("Debug: Attempting to get page info...")  # Debug line
                                
                                # First try to get basic page info
                                page = facebook_client.get_object(
                                    identifier,
                                    fields='id,name,username'
                                )
                                st.write("Debug: Basic page info response:", page)  # Debug line
                                
                                # If basic info succeeds, try to get more detailed info
                                try:
                                    detailed_page = facebook_client.get_object(
                                        page['id'],
                                        fields='id,name,username,link,fan_count'
                                    )
                                    st.write("Debug: Detailed page info response:", detailed_page)  # Debug line
                                    
                                    page_id = detailed_page['id']
                                    page_name = detailed_page.get('name', identifier)
                                    page_link = detailed_page.get('link', f"https://www.facebook.com/{identifier}")
                                    
                                    st.success(f"Found page: {page_name}")
                                    st.info(f"Page URL: {page_link}")
                                    
                                except Exception as e:
                                    st.error(f"Debug: Detailed page info error: {str(e)}")
                                    # Fall back to basic info
                                    page_id = page['id']
                                    page_name = page.get('name', identifier)
                                    page_link = f"https://www.facebook.com/{identifier}"
                                    
                                    st.success(f"Found page: {page_name}")
                                    st.info(f"Page URL: {page_link}")
                                
                            except Exception as e:
                                st.error(f"Debug: Page info error: {str(e)}")
                                st.error("Could not find the Facebook page. Please make sure the page is public and you have the necessary permissions.")
                                st.stop()
                        else:
                            st.error("Invalid Facebook URL format. Please use a valid Facebook page URL.")
                            st.stop()
                        
                        # Get page posts
                        with st.spinner("Collecting and analyzing Facebook posts..."):
                            try:
                                posts = facebook_client.get_connections(
                                    page_id,
                                    'posts',
                                    fields='message,created_time,id,permalink_url',
                                    limit=limit
                                )
                            except Exception as e:
                                st.error("Could not access page posts.")
                                st.stop()
                            
                            if not posts or 'data' not in posts:
                                st.error("No posts found or unable to access posts.")
                                st.stop()
                            
                            results = []
                            for post in posts['data']:
                                if 'message' in post and post['message']:
                                    text = post['message']
                                    result = predict_sentiment(text)
                                    result['post_id'] = post['id']
                                    result['created_at'] = post['created_time']
                                    result['user'] = page_name
                                    result['post_url'] = post.get('permalink_url', '')
                                    results.append(result)
                            
                            if not results:
                                st.warning("No posts with text content found to analyze.")
                                st.stop()
                            
                            # Display results
                            st.subheader("Analysis Results")
                            
                            # Sentiment distribution
                            sentiment_counts = pd.DataFrame(results)['sentiment'].value_counts()
                            fig = px.pie(values=sentiment_counts.values, 
                                       names=sentiment_counts.index,
                                       title='Sentiment Distribution')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display posts
                            st.subheader("Analyzed Posts")
                            for result in results:
                                st.write(f"**Post by {result['user']}:**")
                                st.write(result['text'])
                                st.write(f"**Sentiment:** {result['sentiment']} ({result['confidence']*100:.1f}%)")
                                st.write(f"**Posted on:** {result['created_at']}")
                                if result['post_url']:
                                    st.write(f"**Post URL:** {result['post_url']}")
                                st.write("---")
                    except Exception as e:
                        st.error(f"Error analyzing Facebook posts: {str(e)}")
                else:
                    st.warning("Please enter a Facebook page URL.")
    
    elif page == "Batch Analysis":
        st.header("Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file with text data", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if 'text' in df.columns:
                if st.button("Analyze Batch"):
                    with st.spinner("Analyzing texts..."):
                        results = []
                        for text in df['text']:
                            result = predict_sentiment(text)
                            results.append(result)
                        
                        # Add results to dataframe
                        df['sentiment'] = [r['sentiment'] for r in results]
                        df['confidence'] = [r['confidence'] for r in results]
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Sentiment distribution
                        sentiment_counts = df['sentiment'].value_counts()
                        fig = px.pie(values=sentiment_counts.values, 
                                   names=sentiment_counts.index,
                                   title='Sentiment Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display dataframe
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("The uploaded file must contain a 'text' column.")
    
    else:  # About page
        st.header("About")
        st.markdown("""
        ### Youth Sentiment Analysis Dashboard
        
        This dashboard provides sentiment analysis capabilities across various social media platforms.
        
        **Features:**
        - Text sentiment analysis
        - Social media content analysis
        - Batch processing of text data
        - Real-time sentiment visualization
        
        **Supported Platforms:**
        - Twitter
        - Reddit
        - YouTube
        - Facebook
        
        **Technologies Used:**
        - Streamlit
        - Transformers
        - Plotly
        - Social Media APIs
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main() 