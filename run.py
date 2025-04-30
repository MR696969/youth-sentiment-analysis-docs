from app_sentiment import app
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        logger.info("Starting web app on http://127.0.0.1:5000/")
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=True,
            ssl_context=None  # Disable SSL
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise 