import os
import tweepy
import praw
import facebook
import instaloader
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging
import mimetypes

# Custom imghdr implementation for Python 3.13+
def what(file, h=None):
    if h is None:
        if isinstance(file, str):
            with open(file, 'rb') as f:
                h = f.read(32)
        else:
            location = file.tell()
            h = file.read(32)
            file.seek(location)
    
    if h.startswith(b'\xff\xd8'):
        return 'jpeg'
    elif h.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    elif h.startswith(b'GIF87a') or h.startswith(b'GIF89a'):
        return 'gif'
    elif h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'
    return None

# Monkey patch tweepy's imghdr
import sys
sys.modules['imghdr'] = type('imghdr', (), {'what': what})

# Load environment variables
load_dotenv()

class SocialMediaCollector:
    def __init__(self):
        self.twitter_api = self._init_twitter()
        self.reddit_api = self._init_reddit()
        self.facebook_api = self._init_facebook()
        self.instagram_api = self._init_instagram()
        
    def _init_twitter(self) -> tweepy.API:
        try:
            auth = tweepy.OAuthHandler(
                os.getenv('TWITTER_API_KEY'),
                os.getenv('TWITTER_API_SECRET')
            )
            auth.set_access_token(
                os.getenv('TWITTER_ACCESS_TOKEN'),
                os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            )
            return tweepy.API(auth)
        except Exception as e:
            logging.error(f"Failed to initialize Twitter API: {str(e)}")
            return None

    def _init_reddit(self) -> praw.Reddit:
        try:
            return praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'YouthSentimentBot 1.0')
            )
        except Exception as e:
            logging.error(f"Failed to initialize Reddit API: {str(e)}")
            return None

    def _init_facebook(self) -> facebook.GraphAPI:
        try:
            return facebook.GraphAPI(access_token=os.getenv('FACEBOOK_ACCESS_TOKEN'))
        except Exception as e:
            logging.error(f"Failed to initialize Facebook API: {str(e)}")
            return None

    def _init_instagram(self) -> instaloader.Instaloader:
        try:
            L = instaloader.Instaloader()
            if os.getenv('INSTAGRAM_USERNAME') and os.getenv('INSTAGRAM_PASSWORD'):
                L.login(os.getenv('INSTAGRAM_USERNAME'), os.getenv('INSTAGRAM_PASSWORD'))
            return L
        except Exception as e:
            logging.error(f"Failed to initialize Instagram API: {str(e)}")
            return None

    def collect_twitter_data(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Collect tweets based on query"""
        if not self.twitter_api:
            return []
        
        tweets = []
        try:
            for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=query, tweet_mode="extended").items(limit):
                tweets.append({
                    'platform': 'twitter',
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user': tweet.user.screen_name,
                    'likes': tweet.favorite_count,
                    'retweets': tweet.retweet_count
                })
        except Exception as e:
            logging.error(f"Error collecting Twitter data: {str(e)}")
        return tweets

    def collect_reddit_data(self, subreddit: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Collect Reddit posts and comments from a subreddit"""
        if not self.reddit_api:
            return []
        
        posts = []
        try:
            subreddit = self.reddit_api.subreddit(subreddit)
            for post in subreddit.hot(limit=limit):
                posts.append({
                    'platform': 'reddit',
                    'text': post.selftext,
                    'title': post.title,
                    'created_at': post.created_utc,
                    'author': str(post.author),
                    'score': post.score,
                    'comments': post.num_comments
                })
        except Exception as e:
            logging.error(f"Error collecting Reddit data: {str(e)}")
        return posts

    def collect_facebook_data(self, page_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Collect Facebook posts from a page"""
        if not self.facebook_api:
            return []
        
        posts = []
        try:
            feed = self.facebook_api.get_connections(page_id, 'feed', limit=limit)
            for post in feed['data']:
                posts.append({
                    'platform': 'facebook',
                    'text': post.get('message', ''),
                    'created_at': post.get('created_time'),
                    'id': post.get('id'),
                    'type': post.get('type')
                })
        except Exception as e:
            logging.error(f"Error collecting Facebook data: {str(e)}")
        return posts

    def collect_instagram_data(self, hashtag: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Collect Instagram posts by hashtag"""
        if not self.instagram_api:
            return []
        
        posts = []
        try:
            hashtag_posts = self.instagram_api.get_hashtag_posts(hashtag)
            count = 0
            for post in hashtag_posts:
                if count >= limit:
                    break
                posts.append({
                    'platform': 'instagram',
                    'text': post.caption if post.caption else '',
                    'created_at': post.date_local,
                    'likes': post.likes,
                    'comments': post.comments,
                    'author': post.owner_username
                })
                count += 1
        except Exception as e:
            logging.error(f"Error collecting Instagram data: {str(e)}")
        return posts

    def collect_youtube_comments(self, video_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Collect YouTube comments from a video"""
        from googleapiclient.discovery import build
        
        comments = []
        try:
            youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
            
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=limit
            )
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'platform': 'youtube',
                    'text': comment['textDisplay'],
                    'created_at': comment['publishedAt'],
                    'author': comment['authorDisplayName'],
                    'likes': comment['likeCount']
                })
        except Exception as e:
            logging.error(f"Error collecting YouTube data: {str(e)}")
        return comments 