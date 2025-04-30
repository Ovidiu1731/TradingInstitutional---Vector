import os
import discord
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://web-production-4b33.up.railway.app")  # No trailing `/ask`

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user.name} (ID: {client.user.id})")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user.mentioned_in(message):
        # DEBUG LOGGING
        print("Raw message content:", message.content)

        # Strip mention tag
        question = message.content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()
        print("Extracted question:", question)

        if not question:
            await message.channel.send("Întrebarea este goală.")
            return

        async with message.channel.typing():
            try:
                # Check for image attachment
                if message.attachments:
                    image_url = message.attachments[0].url
                    endpoint = f"{API_BASE_URL}/ask-image"
                    payload = {
                        "question": question,
                        "image_url": image_url
                    }
                    print("📷 Sending to /ask-image:", image_url)
                else:
                    endpoint = f"{API_BASE_URL}/ask"
                    payload = {
                        "question": question
                    }
                    print("💬 Sending to /ask (text-only)")

                async with aiohttp.ClientSession() as session:
                    async with session.post(endpoint, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            answer = data.get("answer", "Nu am găsit un răspuns.")
                        else:
                            answer = f"A apărut o eroare la server. Cod: {resp.status}"
            except Exception as e:
                answer = f"❌ Eroare la conectarea cu serverul: {e}"

        await message.channel.send(answer)

client.run(DISCORD_TOKEN)
