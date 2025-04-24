import os
import discord
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL", "https://web-production-4b33.up.railway.app/ask")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user.name} (ID: {client.user.id})")

@client.event
async def on_message(message):
    # Ignore the bot's own messages
    if message.author == client.user:
        return

    # Only respond if mentioned
    if client.user.mentioned_in(message):
        # strip out the mention and whitespace
        question = message.content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()

        if not question:
            await message.channel.send("Întrebarea este goală.")
            return

        # show typing indicator
        async with message.channel.typing():
            try:
                async with aiohttp.ClientSession() as session:
                    # send the user’s question under the "query" key
                    async with session.post(API_URL, json={"query": question}) as resp:
                        if resp.status == 200:
                            data   = await resp.json()
                            answer = data.get("answer", "Nu am găsit un răspuns.")
                        else:
                            answer = "A apărut o eroare la server. Încearcă din nou mai târziu."
            except Exception as e:
                answer = f"❌ Eroare la conectarea cu serverul: {e}"

        await message.channel.send(answer)

client.run(DISCORD_TOKEN)
