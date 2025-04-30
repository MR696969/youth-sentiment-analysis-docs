import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime
import requests
import json

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  email TEXT,
                  created_at TIMESTAMP,
                  auth_provider TEXT,
                  social_id TEXT)''')
    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Register new user
def register_user(username, password, email):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check if username already exists
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        if c.fetchone():
            return False, "Username already exists"
        
        # Hash password and store user
        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, password, email, created_at) VALUES (?, ?, ?, ?)",
                 (username, hashed_password, email, datetime.now()))
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except Exception as e:
        return False, str(e)

# Login user
def login_user(username, password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check credentials
        hashed_password = hash_password(password)
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                 (username, hashed_password))
        user = c.fetchone()
        conn.close()
        
        if user:
            return True, "Login successful"
        return False, "Invalid username or password"
    except Exception as e:
        return False, str(e)

# Facebook login
def facebook_login(access_token):
    try:
        # Verify token with Facebook
        response = requests.get(
            f'https://graph.facebook.com/me?access_token={access_token}&fields=id,name,email'
        )
        if response.status_code != 200:
            return False, "Invalid Facebook token"
        
        user_data = response.json()
        facebook_id = user_data.get('id')
        email = user_data.get('email')
        name = user_data.get('name')
        
        # Check if user exists
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE social_id = ? AND auth_provider = 'facebook'", (facebook_id,))
        user = c.fetchone()
        
        if not user:
            # Create new user
            c.execute("INSERT INTO users (username, email, created_at, auth_provider, social_id) VALUES (?, ?, ?, ?, ?)",
                     (name, email, datetime.now(), 'facebook', facebook_id))
            conn.commit()
        
        conn.close()
        return True, "Login successful"
    except Exception as e:
        return False, str(e)

# Check if user is logged in
def is_logged_in():
    return 'username' in st.session_state or 'facebook_id' in st.session_state

# Logout user
def logout_user():
    if 'username' in st.session_state:
        del st.session_state['username']
    if 'facebook_id' in st.session_state:
        del st.session_state['facebook_id']
    st.rerun()

# Initialize database on import
init_db() 