import os
import discord
from discord import app_commands
from discord.ui import Button, View
import aiohttp
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://web-production-4b33.up.railway.app")  # Use local URL by default

# Set up intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.reactions = True  # Add this for button interactions

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user.name} (ID: {client.user.id})")

class FeedbackView(discord.ui.View):
    def __init__(self, api_url, question, answer, analysis_data=None):
        super().__init__(timeout=600)  # 10 minute timeout
        self.api_url = api_url
        self.question = question
        self.answer = answer
        self.analysis_data = analysis_data
        
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
                "analysis_data": self.analysis_data
            }
            
            print(f"Sending feedback to: {endpoint}")
            print(f"Feedback includes analysis_data: {self.analysis_data is not None}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload) as resp:
                    print(f"Feedback response status: {resp.status}")
                    if resp.status == 200:
                        # Just update the view with disabled buttons, don't change the text
                        await interaction.response.edit_message(content=self.answer, view=self)
                    else:
                        # In case of error, you can either be silent or show a small error indicator
                        await interaction.response.edit_message(content=self.answer, view=self)
            
        except Exception as e:
            print(f"Error sending feedback: {e}")
            # Keep original answer unchanged
            await interaction.response.edit_message(content=self.answer, view=self)

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    if client.user.mentioned_in(message):
        # DEBUG
        print("Raw message content:", message.content)
        question = message.content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()
        print("Extracted question:", question)
        
        if not question:
            await message.channel.send("Întrebarea este goală.")
            return
        
        async with message.channel.typing():
            try:
                # Initialize variables for tracking query type and analysis data
                endpoint = API_BASE_URL
                is_image_query = False
                analysis_data = None
                
                # Check for image
                if message.attachments:
                    image_url = message.attachments[0].url
                    endpoint = f"{API_BASE_URL.rstrip('/')}/ask-image-hybrid"
                    payload = {
                        "question": question,
                        "image_url": image_url
                    }
                    is_image_query = True
                    print(f"📷 Routing to {endpoint} with payload: {payload}")
                else:
                    # For text-only queries, use the base URL as is
                    endpoint = f"{API_BASE_URL.rstrip('/')}/ask"
                    payload = {
                        "question": question
                    }
                    print(f"💬 Routing to {endpoint} with payload: {payload}")
                
                print(f"Full request URL: {endpoint}")
                async with aiohttp.ClientSession() as session:
                    print(f"Making POST request to: {endpoint}")
                    try:
                        async with session.post(endpoint, json=payload) as resp:
                            print(f"Response status: {resp.status}")
                            print(f"Response headers: {resp.headers}")
                            if resp.status == 200:
                                data = await resp.json()
                                print(f"Response data: {data.keys()}")
                                answer = data.get("answer", "Nu am găsit un răspuns.")
                                
                                # Capture analysis data for feedback if present
                                if "analysis_data" in data and is_image_query:
                                    analysis_data = data["analysis_data"]
                                    print("Image analysis data captured for feedback")
                                
                                print(f"Answer (first 100 chars): {answer[:100]}...")
                            else:
                                response_text = await resp.text()
                                print(f"Error response text: {response_text[:200]}...")
                                answer = f"A apărut o eroare la server. Cod: {resp.status}"
                    except Exception as e:
                        print(f"Exception during request: {type(e).__name__}: {str(e)}")
                        answer = f"❌ Eroare la conectarea cu serverul: {e}"
            except Exception as e:
                print(f"❌ Exception occurred: {str(e)}")
                answer = f"❌ Eroare la conectarea cu serverul: {e}"

        # Create feedback view with correct endpoint and analysis data
        # Extract the base URL without the path part for feedback
        base_url = API_BASE_URL.split("/ask")[0] if "/ask" in API_BASE_URL else API_BASE_URL
        view = FeedbackView(base_url, question, answer, analysis_data)
        
        print(f"About to send answer to Discord: {answer[:100]}...")
        await message.channel.send(answer, view=view)

client.run(DISCORD_TOKEN)
