# Youth Sentiment Analysis

A Streamlit-based web application for analyzing sentiment across various social media platforms.

## Features

- Text sentiment analysis
- Social media content analysis (Twitter, Reddit, YouTube, Facebook)
- Batch processing of text data
- Real-time sentiment visualization
- Privacy policy and data deletion information

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MR696969/youth-sentiment-analysis.git
cd youth-sentiment-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with your API keys:
```
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret

REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent

YOUTUBE_API_KEY=your_youtube_api_key

FACEBOOK_ACCESS_TOKEN=your_facebook_access_token
```

5. Run the application:
```bash
streamlit run app_streamlit.py
```

## Privacy Policy and Data Deletion

- Privacy Policy: [Privacy Policy](https://mr696969.github.io/youth-sentiment-analysis/privacy.html)
- Data Deletion Instructions: [Data Deletion](https://mr696969.github.io/youth-sentiment-analysis/data_deletion.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or concerns, please contact: maheswarrana49@gmail.com 