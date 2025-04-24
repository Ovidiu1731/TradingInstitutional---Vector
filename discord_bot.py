import os
import discord
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("CHATBOT_API_URL", "http://localhost:8000/ask")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user.name} (ID: {client.user.id})")

@client.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == client.user:
        return

    # Check if bot is mentioned
    if client.user.mentioned_in(message):
        question = message.content.replace(f"<@{client.user.id}>", "").strip()

        if not question:
            await message.channel.send("Întrebarea este goală.")
            return

        async with message.channel.typing():  # Show "is typing" indicator
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(API_URL, json={"question": question}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            answer = data.get("answer", "Nu am găsit un răspuns.")
                        else:
                            answer = "A apărut o eroare la server. Încearcă din nou mai târziu."
            except Exception as e:
                answer = f"❌ Eroare la conectarea cu serverul: {e}"

        await message.channel.send(answer)

client.run(DISCORD_TOKEN)
