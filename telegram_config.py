#!/usr/bin/env python3
"""
Telegram Configuration for Document Storage
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TelegramConfig:
    """Telegram API configuration"""
    
    # Get these from https://my.telegram.org/apps
    API_ID = os.getenv('TELEGRAM_API_ID')
    API_HASH = os.getenv('TELEGRAM_API_HASH')
    PHONE_NUMBER = os.getenv('TELEGRAM_PHONE_NUMBER')
    
    # Optional: Private channel for document storage
    # Create a private channel and add your bot as admin
    STORAGE_CHANNEL = os.getenv('TELEGRAM_STORAGE_CHANNEL')  # e.g., '@your_private_channel'
    
    @classmethod
    def validate_config(cls):
        """Validate that all required config is present"""
        missing = []
        
        if not cls.API_ID:
            missing.append('TELEGRAM_API_ID')
        if not cls.API_HASH:
            missing.append('TELEGRAM_API_HASH')
        if not cls.PHONE_NUMBER:
            missing.append('TELEGRAM_PHONE_NUMBER')
        
        if missing:
            raise ValueError(f"Missing required Telegram configuration: {', '.join(missing)}")
        
        return True
    
    @classmethod
    def get_setup_instructions(cls):
        """Get setup instructions for Telegram API"""
        return """
🔧 Telegram Document Storage Setup Instructions:

1. Get Telegram API Credentials:
   - Go to https://my.telegram.org/apps
   - Log in with your phone number
   - Create a new application
   - Copy API ID and API Hash

2. Add to your .env file:
   TELEGRAM_API_ID=your_api_id_here
   TELEGRAM_API_HASH=your_api_hash_here
   TELEGRAM_PHONE_NUMBER=+1234567890

3. Optional - Create Private Storage Channel:
   - Create a private Telegram channel
   - Add yourself as admin
   - Get channel username (e.g., @my_docs_storage)
   - Add to .env: TELEGRAM_STORAGE_CHANNEL=@my_docs_storage

4. Install required packages:
   pip install telethon aiofiles

📝 Note: Without a private channel, documents will be stored in your "Saved Messages"
"""