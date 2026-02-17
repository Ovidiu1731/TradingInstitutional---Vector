import logging
import discord
from discord.ext import commands
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='!')
        self.add_command(self.ping)

    @commands.command()
    async def ping(self, ctx):
        await ctx.send('Pong!')

async def run_bot():
    bot = MyBot()
    try:
        await bot.start('YOUR_BOT_TOKEN', reconnect=True)
    except discord.LoginFailure:
        logger.error("Login failed: check your bot token").
    except asyncio.TimeoutError:
        logger.error("Connection timed out: please check your internet connection or Discord's status.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(run_bot())