import os
from dotenv import load_dotenv
from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.bot import DefaultBotProperties
from google import genai

import prompt

load_dotenv()

CHAT_TRIGGER_WORD = "пупс"
IMAGE_TRIGGER_COMMAND = "нарисуй"
MUSIC_TRIGGER_COMMAND = 'спой'
PROMPT = prompt.PUPS
MAX_RETRIES = 30
RETRY_DELAY = 5
AIRFORCE_API_URL = "https://api.airforce/v1/chat/completions"
AIRFORCE_API_KEY = os.getenv('AIRFORCE_API_KEY')
API_KEY_GEMINI = os.getenv('API_KEY_GEMINI')
IMGBB_API_KEY = os.getenv('IMGBB_API_KEY')
PUPS_BOT_TOKEN = os.getenv('PUPS_BOT_TOKEN')
MASTER_KEY = os.getenv("MASTER_CRYPTO_KEY")
WORKER_URL = os.getenv("WORKER_URL")
WORKER_API_KEY = os.getenv("WORKER_API_KEY")

session = AiohttpSession(timeout=300)
bot = Bot(
    token=PUPS_BOT_TOKEN,
    session=session,
    default=DefaultBotProperties()
)

client = genai.Client(
    api_key=API_KEY_GEMINI,
)