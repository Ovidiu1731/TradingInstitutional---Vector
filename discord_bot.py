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

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://web-production-4b33.up.railway.app")

# Simple message deduplication
processed_messages = set()
processing_messages = set()  # Track messages currently being processed
api_request_cache = {}  # Cache API responses to prevent duplicate requests

# Generate a unique session ID for this bot instance
BOT_SESSION_ID = str(uuid.uuid4())[:8]
print(f"🤖 Discord Bot Session ID: {BOT_SESSION_ID}")

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
    """
    Detect Romanian market analysis requests using the specific template:
    "analizeaza X de la Y pana la Z"
    
    This function is now simplified and focused on Romanian usage only.
    """
    import re
    from datetime import datetime, timedelta
    
    # Clean and normalize the question
    question_lower = question.lower().strip()
    
    # Check for the specific Romanian pattern "analizeaza"
    if not question_lower.startswith('analizeaza'):
        return False, {}
    
    print(f"🔍 DETECTED: Market analysis request pattern: {question}")
    
    # Instrument mappings for the Trading Institutional community
    instrument_mappings = {
        # Forex pairs - most common in your community
        'eurusd': 'EURUSD',
        'eur/usd': 'EURUSD', 
        'eur-usd': 'EURUSD',
        'gbpusd': 'GBPUSD',
        'gbp/usd': 'GBPUSD',
        'gbp-usd': 'GBPUSD',
        'usdjpy': 'USDJPY',
        'usd/jpy': 'USDJPY',
        'usdchf': 'USDCHF',
        'usd/chf': 'USDCHF',
        'audusd': 'AUDUSD',
        'aud/usd': 'AUDUSD',
        'nzdusd': 'NZDUSD',
        'nzd/usd': 'NZDUSD',
        'eurgbp': 'EURGBP',
        'eur/gbp': 'EURGBP',
        'eurjpy': 'EURJPY',
        'eur/jpy': 'EURJPY',
        'usdcad': 'USDCAD',
        'usd/cad': 'USDCAD',
        
        # Indices - popular in your community
        'dax': 'GER30',
        'ger30': 'GER30',
        'de30': 'GER30',
        'german30': 'GER30',
        'nasdaq': 'NASDAQ',
        'nas100': 'NASDAQ',
        'us30': 'US30',
        'dow': 'US30',
        'dowjones': 'US30',
        'sp500': 'SPX',
        's&p500': 'SPX',
        'uk100': 'UK100',
        'ftse': 'UK100',
        'ftse100': 'UK100'
    }
    
    # Extract instrument name from the pattern "analizeaza X de la..."
    # Look for instrument after "analizeaza" and before "de la" or "pentru"
    instrument_pattern = r'analizeaza\s+([a-zA-Z0-9/\-]+)(?:\s+(?:de\s+la|pentru))'
    match = re.search(instrument_pattern, question_lower)
    
    if not match:
        # Fallback: look for any known instrument in the text
        symbol = None
        for key, value in instrument_mappings.items():
            if key in question_lower:
                symbol = value
                break
        if not symbol:
            print(f"❌ NO INSTRUMENT FOUND in: {question}")
            return False, {}
    else:
        instrument_text = match.group(1).replace('/', '').replace('-', '')
        symbol = instrument_mappings.get(instrument_text)
        
        if not symbol:
            print(f"❌ UNKNOWN INSTRUMENT: {instrument_text}")
            return False, {}
    
    print(f"✅ EXTRACTED INSTRUMENT: {symbol}")
    
    # Extract date and time using Romanian patterns
    # Pattern: "de la 10:15 pana la 10:30" or "pentru 16-03-2024 de la 10:15 pana la 10:30"
    
    # First extract the date (DD-MM-YYYY format)
    date_pattern = r'(\d{1,2})[.-](\d{1,2})[.-](\d{4})'
    date_match = re.search(date_pattern, question)
    
    extracted_date = None
    if date_match:
        try:
            day, month, year = date_match.groups()
            extracted_date = datetime(int(year), int(month), int(day)).date()
            print(f"✅ EXTRACTED DATE: {extracted_date}")
        except ValueError as e:
            print(f"❌ DATE PARSING ERROR: {e}")
            return False, {}
    else:
        # If no specific date, check for "azi" (today)
        if 'azi' in question_lower or 'astazi' in question_lower:
            extracted_date = datetime.now().date()
            print(f"✅ USING TODAY'S DATE: {extracted_date}")
        else:
            print(f"❌ NO DATE FOUND in: {question}")
            return False, {}
    
    # Extract time range: "de la HH:MM pana la HH:MM"
    time_pattern = r'de\s+la\s+(\d{1,2}):(\d{2})\s+pana\s+la\s+(\d{1,2}):(\d{2})'
    time_match = re.search(time_pattern, question_lower)
    
    from_time = None
    to_time = None
    
    if time_match:
        start_hour, start_min, end_hour, end_min = time_match.groups()
        from_time = f"{start_hour.zfill(2)}:{start_min}"
        to_time = f"{end_hour.zfill(2)}:{end_min}"
        print(f"✅ EXTRACTED TIME RANGE: {from_time} - {to_time}")
    else:
        print(f"❌ NO TIME RANGE FOUND in: {question}")
        return False, {}
    
    # Build the result
    result = {
        "symbol": symbol,
        "from_date": extracted_date.strftime("%Y-%m-%d"),
        "to_date": extracted_date.strftime("%Y-%m-%d"),  # Same day analysis
        "from_time": from_time,
        "to_time": to_time,
        "timeframe": "1min"
    }
    
    print(f"✅ MARKET ANALYSIS REQUEST DETECTED:")
    print(f"   Symbol: {symbol}")
    print(f"   Date: {extracted_date}")
    print(f"   Time: {from_time} - {to_time}")
    
    return True, result

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
                    # Check if it's a market analysis request first
                    if is_market_analysis:
                        print(f"📊 MARKET ANALYSIS REQUEST detected: {market_params}")
                        # Use the new analyze-market endpoint with LLM bridging
                        endpoint = f"{API_BASE_URL.rstrip('/')}/analyze-market"
                        payload = {
                            "question": question,
                            "session_id": f"discord-bot-{BOT_SESSION_ID}-{message.author.id}"
                        }
                        is_image_query = False
                        image_url_for_feedback = None
                        print(f"📊 NEW: Routing to LLM-based market analysis endpoint: {endpoint}")
                        print(f"📊 Payload: {payload}")
                    # Check for image
                    elif message.attachments:
                        image_url = message.attachments[0].url
                        endpoint = f"{API_BASE_URL.rstrip('/')}/ask-image-hybrid"
                        payload = {
                            "question": question,
                            "image_url": image_url,
                            "session_id": f"discord-bot-{BOT_SESSION_ID}-{message.author.id}"
                        }
                        is_image_query = True
                        image_url_for_feedback = image_url
                        print(f"📷 Routing to {endpoint} with payload: {payload}")
                    else:
                        endpoint = f"{API_BASE_URL.rstrip('/')}/ask"
                        payload = {
                            "question": question,
                            "session_id": f"discord-bot-{BOT_SESSION_ID}-{message.author.id}"
                        }
                        is_image_query = False
                        image_url_for_feedback = None
                        print(f"💬 Routing to {endpoint} with payload: {payload}")
                    
                    # Process the request and get the answer - SINGLE CALL
                    if is_market_analysis:
                        # Use POST method for the new endpoint
                        answer = await process_request(endpoint, payload, is_image_query, "POST")
                    else:
                        answer = await process_request(endpoint, payload, is_image_query)
                    
            except discord.Forbidden:
                # If we don't have permission to show typing, continue without it
                print("No permission to show typing indicator, continuing without it")
                
                # Check if it's a market analysis request first
                if is_market_analysis:
                    print(f"📊 MARKET ANALYSIS REQUEST detected: {market_params}")
                    # Use the new analyze-market endpoint with LLM bridging
                    endpoint = f"{API_BASE_URL.rstrip('/')}/analyze-market"
                    payload = {
                        "question": question,
                        "session_id": f"discord-bot-{BOT_SESSION_ID}-{message.author.id}"
                    }
                    is_image_query = False
                    image_url_for_feedback = None
                    print(f"📊 NEW: Routing to LLM-based market analysis endpoint: {endpoint}")
                    print(f"📊 Payload: {payload}")
                # Check for image
                elif message.attachments:
                    image_url = message.attachments[0].url
                    endpoint = f"{API_BASE_URL.rstrip('/')}/ask-image-hybrid"
                    payload = {
                        "question": question,
                        "image_url": image_url,
                        "session_id": f"discord-bot-{BOT_SESSION_ID}-{message.author.id}"
                    }
                    is_image_query = True
                    image_url_for_feedback = image_url
                    print(f"📷 Routing to {endpoint} with payload: {payload}")
                else:
                    endpoint = f"{API_BASE_URL.rstrip('/')}/ask"
                    payload = {
                        "question": question,
                        "session_id": f"discord-bot-{BOT_SESSION_ID}-{message.author.id}"
                    }
                    is_image_query = False
                    image_url_for_feedback = None
                    print(f"💬 Routing to {endpoint} with payload: {payload}")
                
                # Process the request and get the answer - SINGLE CALL
                if is_market_analysis:
                    # Use POST method for the new endpoint
                    answer = await process_request(endpoint, payload, is_image_query, "POST")
                else:
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

async def process_request(endpoint, payload, is_image_query, method="POST"):
    """Process the API request and return the formatted response."""
    # Create a more specific cache key that includes the session ID
    cache_key = hashlib.md5(f"{method}:{endpoint}:{json.dumps(payload, sort_keys=True)}".encode()).hexdigest()
    
    # Check if we've already made this exact request recently
    if cache_key in api_request_cache:
        print(f"🔄 CACHE HIT: Returning cached response for {cache_key[:8]} (Session: {BOT_SESSION_ID})")
        return api_request_cache[cache_key]
    
    print(f"🌐 API REQUEST: Making new {method} request {cache_key[:8]} to {endpoint} (Session: {BOT_SESSION_ID})")
    print(f"📋 Request payload: {payload}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)  # Longer timeout for market analysis
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if method == "GET":
                # For market analysis endpoint (GET with query parameters)
                async with session.get(endpoint, params=payload) as resp:
                    return await handle_response(resp, cache_key)
            else:
                # For regular POST requests
                async with session.post(endpoint, json=payload) as resp:
                    return await handle_response(resp, cache_key)
    except Exception as e:
        error_msg = "Am întâmpinat o eroare la procesarea cererii."
        print(f"❌ API EXCEPTION (Session: {BOT_SESSION_ID}): {e}")
        return error_msg

async def handle_response(resp, cache_key):
    """Handle API response for both GET and POST requests."""
    if resp.status == 200:
        data = await resp.json()
        
        # Handle different response formats
        if "answer" in data:
            # Regular text query response
            answer = data["answer"]
        elif "analysis" in data:
            # Direct analysis response
            answer = data["analysis"]
        elif "analysis_possible" in data:
            # Market analysis AssistantContract response - convert to readable format
            answer = format_market_analysis_response(data)
        else:
            # Fallback for old format
            answer = data.get("context", "Nu am putut procesa răspunsul.")
        
        # Cache the response
        api_request_cache[cache_key] = answer
        
        # Clean up old cache entries (keep only last 10)
        if len(api_request_cache) > 10:
            old_keys = list(api_request_cache.keys())[:5]
            for key in old_keys:
                del api_request_cache[key]
        
        print(f"✅ API SUCCESS: Cached response {cache_key[:8]}")
        return answer
    else:
        error_msg = f"Eroare la procesarea cererii (Status: {resp.status})"
        print(f"❌ API ERROR: {error_msg}")
        return error_msg

def format_market_analysis_response(data: dict) -> str:
    """Convert AssistantContract data to a readable format for Discord users."""
    try:
        if not data.get("analysis_possible", False):
            return "Nu am putut realiza analiza pentru perioada solicitată. Te rog să verifici că instrumentul și intervalul de timp sunt corecte."
        
        # Build the response
        response_parts = []
        
        # Add direction and setup info
        direction = data.get("final_trade_direction", "necunoscut").upper()
        setup_type = data.get("setup_type", "Nedefinit")
        
        if direction != "UNKNOWN" and direction != "NECUNOSCUT":
            response_parts.append(f"🎯 **Direcție sugerată**: {direction}")
        
        if setup_type and setup_type != "Nedefinit":
            response_parts.append(f"📊 **Tip setup**: {setup_type}")
        
        # Add MSS info
        mss_type = data.get("final_mss_type")
        if mss_type:
            response_parts.append(f"📈 **MSS detectat**: {mss_type}")
        
        # Add FVG analysis
        fvg_analysis = data.get("fvg_analysis", {})
        if fvg_analysis and fvg_analysis.get("count", 0) > 0:
            fvg_count = fvg_analysis.get("count", 0)
            fvg_desc = fvg_analysis.get("description", "")
            response_parts.append(f"⚡ **FVG-uri**: {fvg_count} detectate - {fvg_desc}")
        
        # Add liquidity status
        liquidity_status = data.get("liquidity_status_suggestion")
        if liquidity_status:
            response_parts.append(f"💧 **Lichiditate**: {liquidity_status}")
        
        # Add confidence level
        confidence = data.get("direction_confidence", "low")
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
        response_parts.append(f"{confidence_emoji} **Nivel de încredere**: {confidence}")
        
        # Add setup quality if available
        setup_quality = data.get("setup_quality_summary")
        if setup_quality and setup_quality != "No clear setup detected":
            response_parts.append(f"📝 **Calitate setup**: {setup_quality}")
        
        # Add validity score
        validity_score = data.get("setup_validity_score")
        if validity_score is not None:
            score_percentage = int(validity_score * 100)
            response_parts.append(f"📊 **Scor validitate**: {score_percentage}%")
        
        # Join all parts
        if response_parts:
            return "\n".join(response_parts)
        else:
            return "Analiza a fost completată, dar nu am găsit semnale clare de tranzacționare pentru perioada specificată."
    
    except Exception as e:
        print(f"Error formatting market analysis response: {e}")
        return "Am primit datele de analiză, dar am întâmpinat o problemă la formatarea răspunsului."

client.run(DISCORD_TOKEN)