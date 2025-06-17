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
        question = message.content.replace(f"<@{client.user.id}>", "").strip()
        
        # Check if it's a market analysis request
        is_market_analysis, market_params = is_market_analysis_request(question)
        
        if is_market_analysis:
            # Use market analysis endpoint
            endpoint = f"{API_BASE_URL}/candles/{market_params['symbol']}/analysis/assistant"
            params = {
                "from_date": market_params["from_date"],
                "to_date": market_params["to_date"],
                "timeframe": market_params["timeframe"]
            }
            # Make the request...
        else:
            # Use the regular ask endpoint
            endpoint = f"{API_BASE_URL}/ask"
            # Make the request...

        # DEBUG
        print("Raw message content:", message.content)
        print("Extracted question:", question)
        
        if not question:
            await message.channel.send("Întrebarea este goală.")
            return
        
        # Initialize variables that need to be accessible in both try and except blocks
        endpoint = API_BASE_URL
        is_image_query = False
        analysis_data = None
        image_url_for_feedback = None
        answer = None  # Initialize answer variable
        
        try:
            async with message.channel.typing():
                # Check for image
                if message.attachments:
                    image_url = message.attachments[0].url
                    image_url_for_feedback = image_url
                    endpoint = f"{API_BASE_URL.rstrip('/')}/ask-image-hybrid"
                    payload = {
                        "question": question,
                        "image_url": image_url
                    }
                    is_image_query = True
                    print(f"📷 Routing to {endpoint} with payload: {payload}")
                else:
                    endpoint = f"{API_BASE_URL.rstrip('/')}/ask"
                    payload = {
                        "question": question
                    }
                    print(f"💬 Routing to {endpoint} with payload: {payload}")
                
                # Process the request and get the answer
                answer = await process_request(endpoint, payload, is_image_query)
                
        except discord.Forbidden:
            # If we don't have permission to show typing, continue without it
            print("No permission to show typing indicator, continuing without it")
            
            # Check for image
            if message.attachments:
                image_url = message.attachments[0].url
                image_url_for_feedback = image_url
                endpoint = f"{API_BASE_URL.rstrip('/')}/ask-image-hybrid"
                payload = {
                    "question": question,
                    "image_url": image_url
                }
                is_image_query = True
                print(f"📷 Routing to {endpoint} with payload: {payload}")
            else:
                endpoint = f"{API_BASE_URL.rstrip('/')}/ask"
                payload = {
                    "question": question
                }
                print(f"💬 Routing to {endpoint} with payload: {payload}")
            
            # Process the request and get the answer
            answer = await process_request(endpoint, payload, is_image_query)

        # Create feedback view with correct endpoint and analysis data
        base_url = API_BASE_URL.split("/ask")[0] if "/ask" in API_BASE_URL else API_BASE_URL
        view = FeedbackView(base_url, question, answer, analysis_data, image_url_for_feedback)
        
        print(f"About to send answer to Discord: {answer[:100]}...")
        await message.channel.send(answer, view=view)

async def process_request(endpoint, payload, is_image_query):
    """Helper function to process the API request and return the answer"""
    timeout = aiohttp.ClientTimeout(total=90 if is_image_query else 30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # First check API health
            health_timeout = aiohttp.ClientTimeout(total=5)
            async with session.get(f"{API_BASE_URL.rstrip('/')}/health", timeout=health_timeout) as health_resp:
                if health_resp.status == 200:
                    health_data = await health_resp.json()
                    print(f"Health check: {health_data.get('status', 'unknown')}")
                    
                    if health_data.get("status") != "ok":
                        if "components" in health_data and "openai_api" in health_data["components"]:
                            if health_data["components"]["openai_api"]["status"] == "error":
                                return "⚠️ API-ul OpenAI este momentan indisponibil. Administratorii au fost notificați, dar răspunsul poate fi limitat."
                else:
                    print(f"Health check failed with status: {health_resp.status}")
        except Exception as health_error:
            print(f"Health check error: {str(health_error)}")
        
        # Make the main request
        start_time = asyncio.get_event_loop().time()
        retry_count = 0
        max_retries = 2
        
        while retry_count <= max_retries:
            try:
                async with session.post(endpoint, json=payload) as resp:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"Request took {elapsed:.2f} seconds")
                    print(f"Response status: {resp.status}")
                    
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"Response data keys: {data.keys()}")
                        return data.get("answer", "Nu am găsit un răspuns.")
                    
                    elif resp.status == 429:  # Rate limit
                        if retry_count < max_retries:
                            retry_delay = (2 ** retry_count) * 2
                            print(f"Rate limited, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_count += 1
                        else:
                            return "⚠️ Serviciul este momentan supraîncărcat. Te rog să încerci mai târziu."
                    
                    elif resp.status in [502, 503, 504]:  # Server errors
                        if retry_count < max_retries:
                            retry_delay = (2 ** retry_count) * 2
                            print(f"Server error, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_count += 1
                        else:
                            return f"⚠️ Serverul întâmpină dificultăți tehnice (cod {resp.status}). Te rog să încerci mai târziu."
                    
                    else:  # Other errors
                        if resp.status == 400:
                            return "❌ Cererea nu a putut fi procesată corect. Verifică imaginea sau întrebarea."
                        elif resp.status == 401:
                            return "❌ Probleme de autentificare cu serverul API."
                        elif resp.status == 403:
                            return "❌ Nu am permisiunea să accesez această resursă."
                        elif resp.status >= 500:
                            return f"❌ Eroare internă de server (cod {resp.status}). Echipa tehnică a fost notificată."
                        else:
                            return f"❌ Eroare la server. Cod: {resp.status}"
            
            except asyncio.TimeoutError:
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"Retrying after timeout ({retry_count}/{max_retries})...")
                else:
                    return "⏱️ Serverul procesează o cerere complexă și nu a răspuns la timp. Încearcă o întrebare mai simplă sau mai târziu."
            
            except aiohttp.ClientConnectorError as conn_err:
                return f"❌ Nu m-am putut conecta la server: {conn_err}"
            
            except Exception as e:
                return f"❌ Eroare la procesarea cererii: {e}"
        
        return "❌ Nu s-a putut procesa cererea după mai multe încercări."

client.run(DISCORD_TOKEN)