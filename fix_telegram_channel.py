#!/usr/bin/env python3
"""
Fix Telegram Private Channel Setup
Helper script to create and configure a private channel for document storage
"""

import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv
import os

load_dotenv()

async def setup_telegram_channel():
    """Setup a private Telegram channel for document storage"""
    print("📱 Setting up Telegram Private Channel for Document Storage")
    print("=" * 60)
    
    # Get credentials
    api_id = os.getenv('TELEGRAM_API_ID')
    api_hash = os.getenv('TELEGRAM_API_HASH')
    phone = os.getenv('TELEGRAM_PHONE_NUMBER')
    
    if not all([api_id, api_hash, phone]):
        print("❌ Missing Telegram credentials in .env file")
        return False
    
    try:
        # Initialize client
        client = TelegramClient('channel_setup', api_id, api_hash)
        await client.start(phone=phone)
        
        print("✅ Connected to Telegram")
        
        # Option 1: Create a new private channel
        print("\n🔧 Channel Setup Options:")
        print("1. Create a new private channel")
        print("2. Use an existing channel")
        print("3. Use Saved Messages (no channel needed)")
        
        choice = input("\nChoose option (1, 2, or 3): ").strip()
        
        if choice == "1":
            # Create new private channel
            channel_name = input("Enter channel name (e.g., 'My RAG Documents'): ").strip()
            channel_description = "Private channel for RAG document storage"
            
            try:
                result = await client(functions.channels.CreateChannelRequest(
                    title=channel_name,
                    about=channel_description,
                    megagroup=False  # Channel, not supergroup
                ))
                
                channel_username = f"@{result.chats[0].username}" if result.chats[0].username else None
                channel_id = result.chats[0].id
                
                print(f"✅ Created private channel: {channel_name}")
                print(f"   Channel ID: {channel_id}")
                if channel_username:
                    print(f"   Username: {channel_username}")
                
                # Update .env file
                if channel_username:
                    update_env_channel(channel_username)
                else:
                    print("⚠️ Channel created but no username assigned")
                    print("   You can use the channel ID or set a username in Telegram")
                
            except Exception as e:
                print(f"❌ Failed to create channel: {e}")
                print("💡 Try using an existing channel or Saved Messages")
        
        elif choice == "2":
            # Use existing channel
            print("\n📋 Your channels:")
            
            async for dialog in client.iter_dialogs():
                if dialog.is_channel and not dialog.is_group:
                    print(f"   - {dialog.name} (@{dialog.entity.username if dialog.entity.username else 'no username'})")
            
            channel_input = input("\nEnter channel username (with @) or name: ").strip()
            
            try:
                entity = await client.get_entity(channel_input)
                print(f"✅ Found channel: {entity.title}")
                
                # Test if we can send messages
                test_message = await client.send_message(entity, "🧪 RAG Document Storage Test")
                await client.delete_messages(entity, test_message)
                
                print("✅ Channel access confirmed")
                
                # Update .env file
                if channel_input.startswith('@'):
                    update_env_channel(channel_input)
                else:
                    username = f"@{entity.username}" if entity.username else None
                    if username:
                        update_env_channel(username)
                    else:
                        print("⚠️ Channel has no username. Consider setting one in Telegram.")
                
            except Exception as e:
                print(f"❌ Failed to access channel: {e}")
                print("💡 Make sure you're an admin of the channel")
        
        else:
            # Use Saved Messages
            print("✅ Using Saved Messages for document storage")
            update_env_channel("")  # Empty means use Saved Messages
        
        await client.disconnect()
        print("\n✅ Telegram channel setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def update_env_channel(channel_username):
    """Update .env file with channel username"""
    env_file = '.env'
    
    # Read current .env
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update channel line
    updated_lines = []
    channel_updated = False
    
    for line in lines:
        if line.startswith('TELEGRAM_STORAGE_CHANNEL='):
            if channel_username:
                updated_lines.append(f'TELEGRAM_STORAGE_CHANNEL={channel_username}\n')
                print(f"✅ Updated .env with channel: {channel_username}")
            else:
                updated_lines.append('# TELEGRAM_STORAGE_CHANNEL=  # Using Saved Messages\n')
                print("✅ Updated .env to use Saved Messages")
            channel_updated = True
        else:
            updated_lines.append(line)
    
    # Write back to .env
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)
    
    if not channel_updated:
        print("⚠️ Could not find TELEGRAM_STORAGE_CHANNEL in .env file")

async def test_telegram_storage():
    """Test the configured Telegram storage"""
    print("\n🧪 Testing Telegram Document Storage")
    print("-" * 40)
    
    try:
        from telegram_document_store import TelegramDocumentStore
        
        api_id = os.getenv('TELEGRAM_API_ID')
        api_hash = os.getenv('TELEGRAM_API_HASH')
        phone = os.getenv('TELEGRAM_PHONE_NUMBER')
        channel = os.getenv('TELEGRAM_STORAGE_CHANNEL')
        
        # Clean channel value
        if channel == '@your_private_channel' or not channel:
            channel = None
        
        store = TelegramDocumentStore(api_id, api_hash, phone, channel)
        await store.initialize()
        
        # Test document operations
        test_doc_id = "test_channel_setup"
        test_content = "This is a test document to verify channel setup works correctly."
        
        print("📤 Testing document upload...")
        success = await store.add_document(test_doc_id, test_content, {'test': True})
        
        if success:
            print("✅ Document upload successful")
            
            # Test retrieval
            print("📥 Testing document retrieval...")
            doc = await store.get_document(test_doc_id)
            
            if doc:
                print("✅ Document retrieval successful")
                
                # Cleanup
                await store.remove_document(test_doc_id)
                print("✅ Test cleanup completed")
            else:
                print("❌ Document retrieval failed")
        else:
            print("❌ Document upload failed")
        
        # Get stats
        stats = await store.get_storage_stats()
        print(f"\n📊 Storage Stats:")
        print(f"   Backend: {stats['storage_backend']}")
        print(f"   Channel: {stats['channel']}")
        print(f"   Documents: {stats['total_documents']}")
        
        await store.close()
        print("\n✅ Telegram storage test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Telegram Channel Setup and Test")
    
    try:
        # Setup channel
        setup_success = asyncio.run(setup_telegram_channel())
        
        if setup_success:
            # Test storage
            test_input = input("\n🧪 Test the storage setup? (y/n): ").lower().strip()
            if test_input == 'y':
                asyncio.run(test_telegram_storage())
        
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"❌ Setup error: {e}")