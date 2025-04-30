from flask import Flask, render_template, request, jsonify
from models import EnhancedSentimentAnalyzer
from social_media_collector import SocialMediaCollector
from social_sentiment_analyzer import SocialSentimentAnalyzer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import re
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
analyzer = EnhancedSentimentAnalyzer()
social_collector = SocialMediaCollector()
social_analyzer = SocialSentimentAnalyzer()

# YouTube API setup
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'AIzaSyAvR1Smdb1N2BvrRY9Hy9qJ9VOHlJlb0Xw')

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return "Error loading page. Check if templates/index.html exists.", 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        text = data.get('text', '')
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        logger.debug(f"Analyzing text: {text[:100]}...")  # Log first 100 chars
        sentiment_result = analyzer.analyze(text)
        
        if not sentiment_result:
            return jsonify({'success': False, 'error': 'Error analyzing text'})
        
        return jsonify({
            'success': True,
            'text_length': len(text.split()),
            'sentiment': {
                'pos': sentiment_result['pos'],
                'neu': sentiment_result['neu'],
                'neg': sentiment_result['neg'],
                'compound': sentiment_result['compound']
            },
            'sentiment_label': sentiment_result['label'],
            'transformer_result': sentiment_result.get('transformer')
        })
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_document', methods=['POST'])
def analyze_document():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({'success': False, 'error': 'File encoding not supported. Please use UTF-8 encoded text files.'})
        
        logger.debug(f"Analyzing document: {file.filename}")
        sentiment_result = analyzer.analyze(content)
        
        if not sentiment_result:
            return jsonify({'success': False, 'error': 'Error analyzing document'})
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'text_length': len(content.split()),
            'sentiment': {
                'pos': sentiment_result['pos'],
                'neu': sentiment_result['neu'],
                'neg': sentiment_result['neg'],
                'compound': sentiment_result['compound']
            },
            'sentiment_label': sentiment_result['label'],
            'transformer_result': sentiment_result.get('transformer')
        })
    except Exception as e:
        logger.error(f"Error in analyze_document: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def get_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:be\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_sentiment_label(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    return 'Neutral'

@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        video_url = data.get('video_url', '')
        if not video_url:
            return jsonify({'success': False, 'error': 'No video URL provided'})
        
        try:
            comment_limit = int(data.get('comment_limit', 100))
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid comment limit'})
        
        video_id = get_video_id(video_url)
        if not video_id:
            return jsonify({'success': False, 'error': 'Invalid YouTube URL'})
        
        logger.debug(f"Analyzing video ID: {video_id}")
        
        try:
            # Initialize YouTube API client
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            
            # Get video details
            video_response = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if not video_response.get('items'):
                return jsonify({'success': False, 'error': 'Video not found'})
            
            video_title = video_response['items'][0]['snippet']['title']
            logger.debug(f"Video title: {video_title}")
            
            # Get video comments
            comments = []
            next_page_token = None
            
            while len(comments) < comment_limit:
                try:
                    comments_request = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=min(100, comment_limit - len(comments)),
                        pageToken=next_page_token,
                        textFormat='plainText'
                    )
                    response = comments_request.execute()
                    
                    if not response.get('items'):
                        break
                    
                    for item in response['items']:
                        comment = item['snippet']['topLevelComment']['snippet']
                        comments.append({
                            'text': comment['textDisplay'],
                            'author': comment['authorDisplayName'],
                            'likes': comment['likeCount']
                        })
                    
                    next_page_token = response.get('nextPageToken')
                    if not next_page_token or len(comments) >= comment_limit:
                        break
                    
                except HttpError as e:
                    if e.resp.status == 403:
                        return jsonify({'success': False, 'error': 'Comments are disabled for this video'})
                    raise
            
            if not comments:
                return jsonify({'success': False, 'error': 'No comments found for this video'})
            
            logger.debug(f"Retrieved {len(comments)} comments")
            
            # Analyze sentiment for each comment
            positive_comments = []
            neutral_comments = []
            negative_comments = []
            total_compound = 0
            
            for comment in comments:
                sentiment_result = analyzer.analyze(comment['text'])
                if not sentiment_result:
                    continue
                
                total_compound += sentiment_result['compound']
                comment['sentiment'] = sentiment_result
                
                if sentiment_result['label'] == 'Positive':
                    positive_comments.append(comment)
                elif sentiment_result['label'] == 'Negative':
                    negative_comments.append(comment)
                else:
                    neutral_comments.append(comment)
            
            # Calculate sentiment distribution
            total_comments = len(comments)
            sentiment_distribution = {
                'positive': round(len(positive_comments) / total_comments * 100, 1),
                'neutral': round(len(neutral_comments) / total_comments * 100, 1),
                'negative': round(len(negative_comments) / total_comments * 100, 1)
            }
            
            # Calculate average sentiment
            average_compound = total_compound / total_comments if total_comments > 0 else 0
            
            return jsonify({
                'success': True,
                'video_title': video_title,
                'total_comments': total_comments,
                'average_sentiment': {'compound': average_compound},
                'sentiment_distribution': sentiment_distribution,
                'comments': {
                    'positive': sorted(positive_comments, key=lambda x: x['likes'], reverse=True)[:5],
                    'neutral': sorted(neutral_comments, key=lambda x: x['likes'], reverse=True)[:5],
                    'negative': sorted(negative_comments, key=lambda x: x['likes'], reverse=True)[:5]
                }
            })
            
        except HttpError as e:
            error_message = e.content.decode('utf-8') if hasattr(e, 'content') else str(e)
            logger.error(f"YouTube API error: {error_message}")
            if 'quota' in error_message.lower():
                return jsonify({'success': False, 'error': 'YouTube API quota exceeded. Please try again tomorrow.'})
            return jsonify({'success': False, 'error': 'Error accessing YouTube API. Please try again later.'})
            
    except Exception as e:
        logger.error(f"Error in analyze_youtube: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_social', methods=['POST'])
def analyze_social():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        platform = data.get('platform', '').lower()
        query = data.get('query', '')
        limit = int(data.get('limit', 100))
        
        if not platform or not query:
            return jsonify({'success': False, 'error': 'Platform and query are required'})
        
        # Collect social media data based on platform
        posts = []
        if platform == 'twitter':
            posts = social_collector.collect_twitter_data(query, limit)
        elif platform == 'reddit':
            posts = social_collector.collect_reddit_data(query, limit)
        elif platform == 'facebook':
            posts = social_collector.collect_facebook_data(query, limit)
        elif platform == 'instagram':
            posts = social_collector.collect_instagram_data(query, limit)
        elif platform == 'youtube':
            video_id = get_video_id(query) if 'youtube.com' in query else query
            if video_id:
                posts = social_collector.collect_youtube_comments(video_id, limit)
        else:
            return jsonify({'success': False, 'error': 'Unsupported platform'})
        
        if not posts:
            return jsonify({'success': False, 'error': f'No data found for the given query on {platform}'})
        
        # Analyze collected posts
        analysis_results = social_analyzer.analyze_batch(posts)
        
        # Aggregate results
        total_posts = len(analysis_results)
        positive_posts = sum(1 for post in analysis_results if post['overall_sentiment'] == 'positive')
        negative_posts = sum(1 for post in analysis_results if post['overall_sentiment'] == 'negative')
        neutral_posts = sum(1 for post in analysis_results if post['overall_sentiment'] == 'neutral')
        
        avg_compound = sum(post['compound'] for post in analysis_results) / total_posts
        avg_confidence = sum(post['confidence'] for post in analysis_results) / total_posts
        
        # Calculate engagement metrics if available
        engagement_scores = [post.get('engagement_score', 0) for post in analysis_results if 'engagement_score' in post]
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else None
        
        return jsonify({
            'success': True,
            'platform': platform,
            'query': query,
            'total_posts': total_posts,
            'sentiment_distribution': {
                'positive': positive_posts,
                'negative': negative_posts,
                'neutral': neutral_posts
            },
            'sentiment_percentages': {
                'positive': (positive_posts / total_posts) * 100,
                'negative': (negative_posts / total_posts) * 100,
                'neutral': (neutral_posts / total_posts) * 100
            },
            'average_sentiment': avg_compound,
            'average_confidence': avg_confidence,
            'average_engagement': avg_engagement,
            'detailed_results': analysis_results[:10],  # Return first 10 detailed results
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in analyze_social: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_multi_platform', methods=['POST'])
def analyze_multi_platform():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        query = data.get('query', '')
        platforms = data.get('platforms', [])
        limit_per_platform = int(data.get('limit', 50))
        
        if not query or not platforms:
            return jsonify({'success': False, 'error': 'Query and platforms are required'})
        
        all_results = {}
        for platform in platforms:
            platform = platform.lower()
            try:
                # Collect and analyze data for each platform
                posts = []
                if platform == 'twitter':
                    posts = social_collector.collect_twitter_data(query, limit_per_platform)
                elif platform == 'reddit':
                    posts = social_collector.collect_reddit_data(query, limit_per_platform)
                elif platform == 'facebook':
                    posts = social_collector.collect_facebook_data(query, limit_per_platform)
                elif platform == 'instagram':
                    posts = social_collector.collect_instagram_data(query, limit_per_platform)
                elif platform == 'youtube':
                    if 'youtube.com' in query:
                        video_id = get_video_id(query)
                        if video_id:
                            posts = social_collector.collect_youtube_comments(video_id, limit_per_platform)
                
                if posts:
                    analysis_results = social_analyzer.analyze_batch(posts)
                    all_results[platform] = {
                        'total_posts': len(analysis_results),
                        'sentiment_distribution': {
                            'positive': sum(1 for post in analysis_results if post['overall_sentiment'] == 'positive'),
                            'negative': sum(1 for post in analysis_results if post['overall_sentiment'] == 'negative'),
                            'neutral': sum(1 for post in analysis_results if post['overall_sentiment'] == 'neutral')
                        },
                        'average_sentiment': sum(post['compound'] for post in analysis_results) / len(analysis_results),
                        'average_confidence': sum(post['confidence'] for post in analysis_results) / len(analysis_results),
                        'sample_posts': analysis_results[:5]  # Include 5 sample posts
                    }
            except Exception as e:
                logger.error(f"Error analyzing {platform}: {str(e)}")
                all_results[platform] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        })
    except Exception as e:
        logger.error(f"Error in analyze_multi_platform: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Check if templates directory exists
    if not os.path.exists('templates'):
        logger.error("Templates directory not found!")
        print("Error: 'templates' directory not found!")
        exit(1)
        
    # Check if index.html exists
    if not os.path.exists('templates/index.html'):
        logger.error("index.html not found in templates directory!")
        print("Error: 'templates/index.html' not found!")
        exit(1)
        
    app.run(debug=True, host='0.0.0.0', port=5000) 