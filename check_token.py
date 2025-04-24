import os
from dotenv import load_dotenv

# explicitly load the .env in this directory
load_dotenv('.env')

token = os.getenv("DISCORD_TOKEN", "")
print("Token loaded?", bool(token), "Length:", len(token))
