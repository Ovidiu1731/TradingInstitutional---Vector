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
API_BASE_URL = os.getenv("API_BASE_URL", "https://web-production-4b33.up.railway.app/ask")  # Production URL as default

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
    def __init__(self, api_url, question, answer):
        super().__init__(timeout=600)  # 10 minute timeout
        self.api_url = api_url
        self.question = question
        self.answer = answer
        
    @discord.ui.button(label="👍", style=discord.ButtonStyle.green)
    async def positive_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.send_feedback(interaction, "positive")
        
    @discord.ui.button(label="👎", style=discord.ButtonStyle.red)
    async def negative_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.send_feedback(interaction, "negative")
        
    async def send_feedback(self, interaction: discord.Interaction, feedback_type):
        try:
            # Disable all buttons
            for item in self.children:
                item.disabled = True
                
            # Send API request
            endpoint = self.api_url.replace("/ask", "") + "/feedback"
            payload = {
                "session_id": "discord-" + str(interaction.user.id),
                "question": self.question,
                "answer": self.answer,
                "feedback": feedback_type
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload) as resp:
                    if resp.status == 200:
                        await interaction.response.edit_message(content=f"{self.answer}\n\n*Feedback înregistrat: {'👍' if feedback_type == 'positive' else '👎'}*", view=self)
                    else:
                        await interaction.response.edit_message(content=f"{self.answer}\n\n*Nu am putut înregistra feedback-ul.*", view=self)
        except Exception as e:
            print(f"Error sending feedback: {e}")
            await interaction.response.edit_message(content=f"{self.answer}\n\n*Eroare la înregistrarea feedback-ului.*", view=self)

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
                # Check for image
                if message.attachments:
                    image_url = message.attachments[0].url
                    # Change this part: Remove "/ask" from endpoint construction
                    endpoint = API_BASE_URL.replace("/ask", "") + "/ask-image-hybrid"
                    payload = {
                        "question": question,
                        "image_url": image_url
                    }
                    print(f"📷 Routing to {endpoint} with payload: {payload}")
                else:
                    # For text-only queries, use the base URL as is
                    endpoint = API_BASE_URL
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

        print(f"About to send answer to Discord: {answer[:100]}...")
        await message.channel.send(answer, view=view)

client.run(DISCORD_TOKEN)
