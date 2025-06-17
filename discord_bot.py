import os
import discord
from discord import app_commands
from discord.ui import Button, View
import aiohttp
import asyncio
from dotenv import load_dotenv
import re
from datetime import datetime

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://web-production-4b33.up.railway.app")

# Simple message deduplication
processed_messages = set()
processing_messages = set()  # Track messages currently being processed

# Set up intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.reactions = True  # Add this for button interactions

client = discord.Client(
    intents=intents,
    status=discord.Status.online 
)

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user.name} (ID: {client.user.id})")
    await client.change_presence(status=discord.Status.online, activity=discord.Game("Trading Assistant"))
     # Wait a bit and set it again to ensure it takes effect
    await asyncio.sleep(5)
    await client.change_presence(status=discord.Status.online, activity=discord.Game("Trading Assistant"))

class FeedbackView(discord.ui.View):
    def __init__(self, api_url, question, answer, analysis_data=None, image_url=None):
        super().__init__(timeout=600)  # 10 minute timeout
        self.api_url = api_url
        self.question = question
        self.answer = answer
        self.analysis_data = analysis_data
        self.image_url = image_url  # Store the image URL
        
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
            # Disable all buttons
            for item in self.children:
                item.disabled = True
            
            # Send API request - use the direct endpoint without string replacement
            endpoint = f"{self.api_url}/feedback"
            payload = {
                "session_id": "discord-" + str(interaction.user.id),
                "question": self.question,
                "answer": self.answer,
                "feedback": feedback_type,
                "query_type": "discord_query",
                "analysis_data": self.analysis_data,
                "image_url": self.image_url  # Include the image URL in the payload
            }
            
            print(f"Sending feedback to: {endpoint}")
            print(f"Feedback includes analysis_data: {self.analysis_data is not None}")
            print(f"Feedback includes image_url: {self.image_url is not None}")
            
            # Add explicit timeout for feedback requests (30 seconds)
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.post(endpoint, json=payload) as resp:
                        print(f"Feedback response status: {resp.status}")
                        if resp.status == 200:
                            # Just update the view with disabled buttons, don't change the text
                            await interaction.response.edit_message(content=self.answer, view=self)
                        else:
                            # In case of error, you can either be silent or show a small error indicator
                            await interaction.response.edit_message(content=self.answer, view=self)
                except asyncio.TimeoutError:
                    print("Timeout while sending feedback")
                    await interaction.response.edit_message(content=self.answer, view=self)
                except aiohttp.ClientConnectorError as e:
                    print(f"Connection error while sending feedback: {e}")
                    await interaction.response.edit_message(content=self.answer, view=self)
            
        except Exception as e:
            print(f"Error sending feedback: {e}")
            # Keep original answer unchanged
            await interaction.response.edit_message(content=self.answer, view=self)

async def cache_training_examples():
    """Cache the example images to prevent GitHub rate limiting issues"""
    example_urls = [
        "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/DE30EUR_2025-05-05_12-29-24_69c08.png",
        "https://raw.githubusercontent.com/Ovidiu1731/Trade-images/main/Screenshot%202025-05-05%20at%2007.18.15%20copy.png",
        "https://github.com/Ovidiu1731/Trade-images/raw/main/Screenshot%202025-05-05%20at%2011.04.35.png"
    ]
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for url in example_urls:
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            print(f"Successfully connected to example URL: {url}")
                        else:
                            print(f"Failed to access example URL: {url} - Status: {resp.status}")
                except Exception as e:
                    print(f"Error accessing example URL: {url} - Error: {e}")
    except Exception as e:
        print(f"Error in cache_training_examples: {e}")

def is_market_analysis_request(question: str) -> tuple[bool, dict]:
    """Detect if the question is a market analysis request"""
    # Common patterns for market analysis
    patterns = [
        r"analizeaza\s+([A-Z]+/[A-Z]+)\s+pentru\s+(\d{2}-\d{2}-\d{4})\s+de\s+la\s+(\d{1,2}:\d{2})\s+pana\s+la\s+(\d{1,2}:\d{2})",
        r"analizeaza\s+([A-Z]+/[A-Z]+)\s+pentru\s+(\d{2}-\d{2}-\d{4})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question.lower())
        if match:
            symbol = match.group(1)
            date_str = match.group(2)
            
            # Parse the date
            try:
                from_date = datetime.strptime(date_str, "%d-%m-%Y").date()
                to_date = from_date
            except ValueError:
                continue  # Skip this pattern if date is invalid
            
            # If we have time range
            if len(match.groups()) > 2:
                from_time = match.group(3)
                to_time = match.group(4)
                return True, {
                    "symbol": symbol,
                    "from_date": from_date.strftime("%Y-%m-%d"),
                    "to_date": to_date.strftime("%Y-%m-%d"),
                    "from_time": from_time,
                    "to_time": to_time,
                    "timeframe": "1min"  # default timeframe
                }
            else:
                # If no time range specified, use full day
                return True, {
                    "symbol": symbol,
                    "from_date": from_date.strftime("%Y-%m-%d"),
                    "to_date": to_date.strftime("%Y-%m-%d"),
                    "timeframe": "1min"  # default timeframe
                }
    
    return False, {}

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    if client.user.mentioned_in(message):
        # More robust deduplication 
        message_key = f"{message.author.id}:{hash(message.content)}:{message.id}"
        
        # Check if already processed
        if message_key in processed_messages:
            print(f"❌ DUPLICATE: Already processed message: {message_key}")
            return
            
        # Check if currently being processed
        if message_key in processing_messages:
            print(f"❌ CONCURRENT: Message already being processed: {message_key}")
            return
            
        # Mark as being processed
        processing_messages.add(message_key)
        processed_messages.add(message_key)
        
        try:
            # Clean up old entries (keep only last 50)
            if len(processed_messages) > 50:
                # Remove oldest half
                old_keys = list(processed_messages)[:25]
                for key in old_keys:
                    processed_messages.discard(key)
            
            question = message.content.replace(f"<@{client.user.id}>", "").strip()
            
            # Check if it's a market analysis request
            is_market_analysis, market_params = is_market_analysis_request(question)
            
            # DEBUG
            print(f"🔄 PROCESSING message ID: {message.id}")
            print(f"📝 Message key: {message_key}")
            print("Raw message content:", message.content)
            print("Extracted question:", question)
            
            if not question:
                await message.channel.send("Întrebarea este goală.")
                return
            
            # Single processing path - no duplicated variables
            try:
                async with message.channel.typing():
                    # Check for image
                    if message.attachments:
                        image_url = message.attachments[0].url
                        endpoint = f"{API_BASE_URL.rstrip('/')}/ask-image-hybrid"
                        payload = {
                            "question": question,
                            "image_url": image_url
                        }
                        is_image_query = True
                        image_url_for_feedback = image_url
                        print(f"📷 Routing to {endpoint} with payload: {payload}")
                    else:
                        endpoint = f"{API_BASE_URL.rstrip('/')}/ask"
                        payload = {
                            "question": question
                        }
                        is_image_query = False
                        image_url_for_feedback = None
                        print(f"💬 Routing to {endpoint} with payload: {payload}")
                    
                    # Process the request and get the answer - SINGLE CALL
                    answer = await process_request(endpoint, payload, is_image_query)
                    
            except discord.Forbidden:
                # If we don't have permission to show typing, continue without it
                print("No permission to show typing indicator, continuing without it")
                
                # Check for image
                if message.attachments:
                    image_url = message.attachments[0].url
                    endpoint = f"{API_BASE_URL.rstrip('/')}/ask-image-hybrid"
                    payload = {
                        "question": question,
                        "image_url": image_url
                    }
                    is_image_query = True
                    image_url_for_feedback = image_url
                    print(f"📷 Routing to {endpoint} with payload: {payload}")
                else:
                    endpoint = f"{API_BASE_URL.rstrip('/')}/ask"
                    payload = {
                        "question": question
                    }
                    is_image_query = False
                    image_url_for_feedback = None
                    print(f"💬 Routing to {endpoint} with payload: {payload}")
                
                # Process the request and get the answer - SINGLE CALL
                answer = await process_request(endpoint, payload, is_image_query)

            # Create feedback view with correct endpoint and analysis data
            base_url = API_BASE_URL.split("/ask")[0] if "/ask" in API_BASE_URL else API_BASE_URL
            view = FeedbackView(base_url, question, answer, None, image_url_for_feedback)
            
            print(f"✅ SENDING answer for message {message.id}: {answer[:100]}...")
            await message.channel.send(answer, view=view)
            
        finally:
            # Remove from processing set when done
            processing_messages.discard(message_key)
            print(f"🏁 FINISHED processing message: {message_key}")

async def process_request(endpoint, payload, is_image_query):
    """Process the API request and return the formatted response."""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Handle the new response format - return only the answer, no sources
                    if "answer" in data:
                        answer = data["answer"]
                        return answer
                    else:
                        # Fallback for old format
                        return data.get("context", "Nu am putut procesa răspunsul.")
                else:
                    return f"Eroare la procesarea cererii (Status: {resp.status})"
    except Exception as e:
        print(f"Error in process_request: {e}")
        return "Am întâmpinat o eroare la procesarea cererii."

client.run(DISCORD_TOKEN)