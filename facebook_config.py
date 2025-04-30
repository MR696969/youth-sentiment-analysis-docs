# Facebook API Configuration
FACEBOOK_CONFIG = {
    'app_id': '1375689360124693',
    'app_secret': 'ba4024ff4293ec45dd192a915fe56c53',
    'access_token': 'EAATjLqOhqxUBO3n8YU1wG43WRCozWqafiYRR9Qn9p4bixRdPYk4W66a55g1PEFCLchRGtVZBaWwMKOaY389krIzlBWlWZAUOa341gNdZA4u1Ywr3pBIFc9cmXCaq7pwWMFQZBEKKqxN9xIZAoy9JVtOrZBwqonr4gw7jEi6bm6PLu8KEYH0ZCGFM2crxVXqib8UKwfnXPMNHHDg0Vsf6AZDZD',
    'permissions': [
        'pages_read_engagement',
        'pages_show_list',
        'pages_read_user_content',
        'public_profile'
    ]
}

def get_facebook_config():
    """Get Facebook configuration"""
    return FACEBOOK_CONFIG 