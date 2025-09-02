#!/usr/bin/env python3
"""
Test environment variable loading
"""

import os
from dotenv import load_dotenv

print("🧪 Testing Environment Variable Loading")
print("=" * 40)

# Check current working directory
print(f"Current directory: {os.getcwd()}")

# Check if .env file exists
env_file = ".env"
env_exists = os.path.exists(env_file)
print(f".env file exists: {'✅' if env_exists else '❌'}")

if env_exists:
    # Read .env file content with different encodings
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(env_file, 'r', encoding='utf-16') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(env_file, 'r', encoding='latin-1') as f:
                content = f.read()
    
    print(f".env content length: {len(content)} chars")
    print(f".env content: '{content.strip()}'")
    print(f".env content repr: {repr(content)}")

# Load environment variables with explicit path
print("\nLoading environment variables...")
result = load_dotenv(dotenv_path=env_file, verbose=True)
print(f"load_dotenv result: {result}")

# Check if API key is loaded
api_key = os.getenv('GOOGLE_API_KEY')
print(f"GOOGLE_API_KEY loaded: {'✅' if api_key else '❌'}")

if api_key:
    print(f"API key (first 20 chars): {api_key[:20]}...")
else:
    print("API key not found in environment")

# Test direct environment variable access
print(f"\nDirect os.environ check: {'GOOGLE_API_KEY' in os.environ}")

# Try manual parsing
if env_exists:
    print("\nManual parsing test:")
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    print(f"Line {line_num}: key='{key}', value='{value[:20]}...'")
    except UnicodeDecodeError:
        try:
            with open(env_file, 'r', encoding='utf-16') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        print(f"Line {line_num}: key='{key}', value='{value[:20]}...'")
        except UnicodeDecodeError:
            print("Could not decode .env file with UTF-8 or UTF-16")

print("\n" + "=" * 40)
print("Environment test completed!")