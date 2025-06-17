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
    """Detect if the question is a market analysis request using flexible NLP approach"""
    import re
    from datetime import datetime, timedelta
    
    # Clean and normalize the question
    question_lower = question.lower().strip()
    
    # Key indicators that this is a market analysis request
    analysis_indicators = [
        'analizeaza', 'analiza', 'analyze', 'chart', 'grafic', 'analiza-mi', 'analizami'
    ]
    
    # Check if it contains analysis indicators
    has_analysis_indicator = any(indicator in question_lower for indicator in analysis_indicators)
    if not has_analysis_indicator:
        return False, {}
    
    # Extract currency pair/instrument using flexible patterns
    # Handle the most traded instruments in the community
    instrument_mappings = {
        # Forex pairs
        'EURUSD': 'EURUSD',
        'EUR/USD': 'EURUSD', 
        'EUR-USD': 'EURUSD',
        'GBPUSD': 'GBPUSD',
        'GBP/USD': 'GBPUSD',
        'GBP-USD': 'GBPUSD',
        'GBPU/USD': 'GBPUSD',  # User's specific example
        'GBPU-USD': 'GBPUSD',
        
        # Indices - map to FMP format
        'GER30': 'GER30',
        'GERMAN': 'GER30',
        'DAX': 'GER30',
        'GER': 'GER30',
        'DE30': 'GER30',
        'NASDAQ': 'NASDAQ',
        'NAS100': 'NASDAQ',
        'NAS': 'NASDAQ',
        'US30': 'US30',
        'DJI': 'US30',
        'DOW': 'US30',
        'DOWJONES': 'US30',
        'UK100': 'UK100',
        'FTSE': 'UK100',
        'FTSE100': 'UK100',
        'UKX': 'UK100',
        'BRITISH': 'UK100',
    }
    
    # More flexible patterns to catch variations
    currency_patterns = [
        r'([A-Z]{3}/?[A-Z]{3})',      # GBPUSD or GBP/USD
        r'([A-Z]{6})',                # GBPUSD
        r'([a-z]{3}/?[a-z]{3})',      # gbpusd or gbp/usd  
        r'([a-z]{6})',                # gbpusd
        r'([A-Z]{2,6})',              # GER30, US30, etc.
        r'([a-z]{2,6})',              # ger30, nasdaq, etc.
        r'(GBPU/?USD)',               # Special case for GBPU/USD
        r'(gbpu/?usd)',               # lowercase version
    ]
    
    symbol = None
    for pattern in currency_patterns:
        matches = re.findall(pattern, question)
        for match in matches:
            candidate = match.upper().replace('-', '').replace('/', '')
            
            # First check direct mappings
            if candidate in instrument_mappings:
                symbol = instrument_mappings[candidate]
                break
                
            # Then check partial matches for indices
            for key, value in instrument_mappings.items():
                if candidate in key or key in candidate:
                    symbol = value
                    break
            
            if symbol:
                break
        if symbol:
            break
    
    if not symbol:
        return False, {}  # No valid currency pair found
    
    # Extract time ranges - be very flexible with formats
    time_patterns = [
        r'(\d{1,2}):(\d{2})',  # Any time format like 10:15, 9:30, etc.
        r'(\d{1,2})\.(\d{2})',  # 10.15 format
        r'(\d{1,2}):(\d{1,2})', # 10:5 format
    ]
    
    times = []
    for pattern in time_patterns:
        matches = re.findall(pattern, question)
        for match in matches:
            hour, minute = match
            # Normalize minutes to 2 digits
            minute = minute.zfill(2)
            times.append(f"{hour}:{minute}")
    
    # Extract dates - flexible date detection
    date_patterns = [
        r'(\d{1,2})[.-](\d{1,2})[.-](\d{4})',     # DD-MM-YYYY or DD.MM.YYYY
        r'(\d{1,2})/(\d{1,2})/(\d{4})',           # MM/DD/YYYY or DD/MM/YYYY
        r'(\d{4})[.-](\d{1,2})[.-](\d{1,2})',     # YYYY-MM-DD
        r'(\d{2})[.-](\d{2})[.-](\d{4})',         # DD-MM-YYYY
    ]
    
    extracted_date = None
    date_keywords = ['data', 'date', 'pentru', 'pe', 'in', 'la']
    
    for pattern in date_patterns:
        match = re.search(pattern, question)
        if match:
            try:
                parts = match.groups()
                if len(parts[0]) == 4:  # YYYY format first
                    year, month, day = parts
                else:
                    # Try both DD-MM-YYYY and MM-DD-YYYY
                    day_or_month, month_or_day, year = parts
                    
                    # Smart date detection based on context
                    if int(day_or_month) > 12:  # Must be day first
                        day, month = day_or_month, month_or_day
                    elif int(month_or_day) > 12:  # Must be month first
                        month, day = day_or_month, month_or_day
                    else:
                        # Default to European format (DD-MM-YYYY) for Romanian users
                        day, month = day_or_month, month_or_day
                
                extracted_date = datetime(int(year), int(month), int(day)).date()
                break
            except ValueError:
                continue
    
    # If no date found, check for relative date keywords
    if not extracted_date:
        today_keywords = ['azi', 'astazi', 'today', 'vandaag']
        yesterday_keywords = ['ieri', 'yesterday'] 
        tomorrow_keywords = ['maine', 'mâine', 'tomorrow']
        
        romanian_tz = ZoneInfo("Europe/Bucharest")
        today = datetime.now(romanian_tz).date()
        
        if any(keyword in question_lower for keyword in today_keywords):
            extracted_date = today
        elif any(keyword in question_lower for keyword in yesterday_keywords):
            extracted_date = today - timedelta(days=1)
        elif any(keyword in question_lower for keyword in tomorrow_keywords):
            extracted_date = today + timedelta(days=1)
        else:
            # Default to today if no date specified
            extracted_date = today
    
    # Build the result
    result = {
        "symbol": symbol,
        "from_date": extracted_date.strftime("%Y-%m-%d"),
        "to_date": extracted_date.strftime("%Y-%m-%d"),
        "timeframe": "1min"
    }
    
    # Add times if found
    if len(times) >= 2:
        result["from_time"] = times[0]
        result["to_time"] = times[1]
    elif len(times) == 1:
        # If only one time, assume it's the start time and add 15 minutes
        try:
            start_time = datetime.strptime(times[0], "%H:%M")
            end_time = start_time + timedelta(minutes=15)
            result["from_time"] = times[0]
            result["to_time"] = end_time.strftime("%H:%M")
        except:
            pass
    
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
                        endpoint = f"{API_BASE_URL.rstrip('/')}/candles/{market_params['symbol']}/analysis/assistant"
                        payload = {
                            "from_date": market_params["from_date"],
                            "to_date": market_params["to_date"],
                            "timeframe": market_params["timeframe"]
                        }
                        if "from_time" in market_params and "to_time" in market_params:
                            payload["from_time"] = market_params["from_time"]
                            payload["to_time"] = market_params["to_time"]
                        is_image_query = False
                        image_url_for_feedback = None
                        print(f"📊 Routing to {endpoint} with payload: {payload}")
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
                        answer = await process_request(endpoint, payload, is_image_query, "GET")
                    else:
                        answer = await process_request(endpoint, payload, is_image_query)
                    
            except discord.Forbidden:
                # If we don't have permission to show typing, continue without it
                print("No permission to show typing indicator, continuing without it")
                
                # Check if it's a market analysis request first
                if is_market_analysis:
                    print(f"📊 MARKET ANALYSIS REQUEST detected: {market_params}")
                    endpoint = f"{API_BASE_URL.rstrip('/')}/candles/{market_params['symbol']}/analysis/assistant"
                    payload = {
                        "from_date": market_params["from_date"],
                        "to_date": market_params["to_date"],
                        "timeframe": market_params["timeframe"]
                    }
                    if "from_time" in market_params and "to_time" in market_params:
                        payload["from_time"] = market_params["from_time"]
                        payload["to_time"] = market_params["to_time"]
                    is_image_query = False
                    image_url_for_feedback = None
                    print(f"📊 Routing to {endpoint} with payload: {payload}")
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
                    answer = await process_request(endpoint, payload, is_image_query, "GET")
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