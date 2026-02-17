import os
import discord
from discord import app_commands
from discord.ui import Button, View
import aiohttp
import asyncio
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
import hashlib
import json
from zoneinfo import ZoneInfo
import uuid
import sys
import logging

# Configure logging - stdout only (no file logging for Railway compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://web-production-4b33.up.railway.app").rstrip('/')

# Validate required environment variables
if not DISCORD_TOKEN:
    logger.error("❌ DISCORD_TOKEN not found in environment variables!")
    sys.exit(1)

if not API_BASE_URL or API_BASE_URL == "https://web-production-4b33.up.railway.app":
    logger.warning("⚠️ Using default API_BASE_URL. Set API_BASE_URL env variable for custom endpoint.")

logger.info(f"📍 API_BASE_URL: {API_BASE_URL}")

# Simple message deduplication
processed_messages = set()
processing_messages = set()
api_request_cache = {}

# Generate a unique session ID for this bot instance
BOT_SESSION_ID = str(uuid.uuid4())[:8]
logger.info(f"🤖 Discord Bot Session ID: {BOT_SESSION_ID}")

# Set up intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.reactions = True

# Create client with enhanced settings
client = discord.Client(
    intents=intents,
    status=discord.Status.online,
    heartbeat_timeout=60.0,  # Increase heartbeat timeout
    max_messages=None  # Don't limit message cache
)

@client.event
async def on_ready():
    logger.info(f"✅ Logged in as {client.user.name} (ID: {client.user.id})")
    try:
        await client.change_presence(
            status=discord.Status.online, 
            activity=discord.Game("Trading Assistant")
        )
        logger.info("✅ Status set successfully")
    except Exception as e:
        logger.error(f"Error setting presence: {e}")

@client.event
async def on_error(event, *args, **kwargs):
    """Global error handler for Discord events."""
    logger.error(f"Error in {event}: {sys.exc_info()}", exc_info=True)

class FeedbackView(discord.ui.View):
    def __init__(self, api_url, question, answer, analysis_data=None, image_url=None):
        super().__init__(timeout=600)
        self.api_url = api_url
        self.question = question
        self.answer = answer
        self.analysis_data = analysis_data
        self.image_url = image_url
        
    @discord.ui.button(label="★★★ Util", style=discord.ButtonStyle.gray, custom_id="positive_feedback", row=0)
    async def positive_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.send_feedback(interaction, "positive")
        
    @discord.ui.button(label="★★☆ Parțial", style=discord.ButtonStyle.gray, custom_id="neutral_feedback", row=0)
    async def neutral_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.send_feedback(interaction, "neutral")
        
    @discord.ui.button(label="★☆☆ Inutil", style=discord.ButtonStyle.gray, custom_id="negative_feedback", row=0)
    async def negative_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.send_feedback(interaction, "negative")
        
    async def send_feedback(self, interaction: discord.Interaction, feedback_type):
        try:
            for item in self.children:
                item.disabled = True
            
            endpoint = f"{self.api_url}/feedback"
            payload = {
                "session_id": "discord-" + str(interaction.user.id),
                "question": self.question,
                "answer": self.answer,
                "feedback": feedback_type,
                "query_type": "discord_query",
                "analysis_data": self.analysis_data,
                "image_url": self.image_url
            }
            
            logger.info(f"📤 Sending feedback to: {endpoint}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        logger.info("✅ Feedback sent successfully")
                        await interaction.response.send_message("Mulțumesc pentru feedback!", ephemeral=True)
                    else:
                        logger.error(f"Feedback error: {resp.status}")
                        await interaction.response.send_message("Eroare la trimiterea feedback-ului.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error sending feedback: {e}", exc_info=True)
            try:
                await interaction.response.send_message("Eroare la trimiterea feedback-ului.", ephemeral=True)
            except:
                pass

@client.event
async def on_message(message):
    """Handle incoming messages."""
    if message.author == client.user:
        return
    
    if client.user.mentioned_in(message):
        message_key = f"{message.author.id}:{hash(message.content)}:{message.id}"
        
        # Check deduplication
        if message_key in processed_messages:
            logger.info(f"⏭️ DUPLICATE: Already processed message: {message_key}")
            return
            
        if message_key in processing_messages:
            logger.info(f"⏳ CONCURRENT: Message already being processed: {message_key}")
            return
            
        processing_messages.add(message_key)
        processed_messages.add(message_key)
        
        try:
            # Cleanup old entries
            if len(processed_messages) > 50:
                old_keys = list(processed_messages)[:25]
                for key in old_keys:
                    processed_messages.discard(key)
            
            question = message.content.replace(f"<@{client.user.id}>", "").strip()
            
            logger.info(f"🔄 PROCESSING message ID: {message.id}")
            logger.info(f"❓ Question: {question[:100]}")
            
            if not question:
                await message.channel.send("Întrebarea este goală.")
                return

            if message.attachments:
                await message.channel.send(
                    "Îmi pare rău, analizarea imaginilor nu mai este disponibilă. "
                    "Te rog să îmi pui întrebarea în format text despre conceptele din mentoratul Trading Instituțional."
                )
                return
            
            # Process the request
            try:
                async with message.channel.typing():
                    endpoint = f"{API_BASE_URL}/ask"
                    payload = {
                        "question": question,
                        "session_id": f"discord-bot-{BOT_SESSION_ID}-{message.author.id}"
                    }
                    
                    logger.info(f"📡 Routing to {endpoint}")
                    answer = await process_request(endpoint, payload, False)
                    
            except discord.Forbidden:
                logger.warning("No permission to show typing indicator, continuing without it")
                endpoint = f"{API_BASE_URL}/ask"
                payload = {
                    "question": question,
                    "session_id": f"discord-bot-{BOT_SESSION_ID}-{message.author.id}"
                }
                answer = await process_request(endpoint, payload, False)

            # Send response with feedback buttons
            base_url = API_BASE_URL
            view = FeedbackView(base_url, question, answer, None, None)
            
            logger.info(f"✅ SENDING answer for message {message.id}")
            await message.channel.send(answer, view=view)
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}", exc_info=True)
            try:
                await message.channel.send(
                    "Am întâmpinat o eroare. Te rog încearcă din nou mai târziu."
                )
            except:
                pass
                
        finally:
            processing_messages.discard(message_key)
            logger.info(f"🏁 FINISHED processing message: {message_key}")

async def process_request(endpoint, payload, is_image_query):
    """Process the API request with enhanced error handling."""
    cache_key = hashlib.md5(f"{endpoint}:{json.dumps(payload, sort_keys=True)}".encode()).hexdigest()
    
    if cache_key in api_request_cache:
        logger.info(f"♻️ CACHE HIT: {cache_key[:8]}")
        return api_request_cache[cache_key]
    
    logger.info(f"🌐 API REQUEST: {cache_key[:8]} to {endpoint}")
    
    try:
        # Use longer timeout for text queries
        timeout = aiohttp.ClientTimeout(total=45, connect=15, sock_read=20)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload) as resp:
                return await handle_response(resp, cache_key)
                
    except asyncio.TimeoutError:
        logger.error(f"❌ API TIMEOUT: Request took too long")
        return "Am întâmpinat o eroare de timeout. Te rog încearcă din nou."
    except aiohttp.ClientConnectorError as e:
        logger.error(f"❌ API CONNECTION ERROR: {e}")
        return "Am întâmpinat o eroare de conexiune. Verifică dacă serverul API este activ."
    except Exception as e:
        logger.error(f"❌ API EXCEPTION: {e}", exc_info=True)
        return "Am întâmpinat o eroare la procesarea cererii. Te rog încearcă din nou."

async def handle_response(resp, cache_key):
    """Handle API response."""
    try:
        if resp.status == 200:
            data = await resp.json()
            
            if "answer" in data:
                answer = data["answer"]
            else:
                answer = data.get("context", "Nu am putut procesa răspunsul.")
            
            # Cache the response
            api_request_cache[cache_key] = answer
            
            # Cleanup old cache
            if len(api_request_cache) > 10:
                old_keys = list(api_request_cache.keys())[:5]
                for key in old_keys:
                    del api_request_cache[key]
            
            logger.info(f"✅ API SUCCESS: {cache_key[:8]}")
            return answer
        else:
            error_msg = f"Eroare la procesarea cererii (Status: {resp.status})"
            logger.error(f"❌ API ERROR: {error_msg}")
            return error_msg
            
    except Exception as e:
        logger.error(f"❌ Response handling error: {e}", exc_info=True)
        return "Am întâmpinat o eroare la citirea răspunsului."

def main():
    """Main entry point with error handling."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Discord Bot")
    logger.info("=" * 60)
    
    try:
        client.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("⛔ Bot interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
