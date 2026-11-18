import os
from dotenv import load_dotenv
from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
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
COMFY_URL = os.getenv('COMFY_URL', 'http://192.168.0.3:8188')  # порт 8188 по умолчанию, как в run.py
IMAGE_PROMPT = prompt.PUPS_IMAGE
VIDEO_TRIGGER_COMMAND = "анимируй"
VIDEO_PROMPT = prompt.PUPS_VIDEO
EDIT_PROMPT = prompt.PUPS_EDIT
FLF_VIDEO_PROMPT = prompt.PUPS_FLF_VIDEO
MULTI_EDIT_PROMPT = prompt.PUPS_MULTI_EDIT
MUSIC_PROMPT = prompt.PUPS_MUSIC
MUSIC_M2M_PROMPT = prompt.PUPS_MUSIC_M2M
DEFAULT_MUSIC_DURATION = 180   # сек
DEFAULT_MUSIC_LANGUAGE = "ru"
DEFAULT_LYRICS_STRENGTH = 1

session = AiohttpSession(timeout=300)
bot = Bot(
    token=PUPS_BOT_TOKEN,
    session=session,
    default=DefaultBotProperties(
        # ГЛАВНЫЙ ИСТОЧНИК ОШИБКИ "can't parse entities" БЫЛ ЗДЕСЬ:
        # parse_mode=ParseMode.MARKDOWN заставлял ЛЮБУЮ отправку в проекте
        # (даже из файлов, где parse_mode не указан: bot_dialogue.py, summary.py
        # и т.д., а также message.reply/answer в bot11_pups.py) парситься как
        # legacy-Markdown. Незакрытые _ * ` в ответах моделей роняли отправку.
        # ТЕПЕРЬ глобального парсинга нет — текст уходит «как есть», сломаться
        # нечему. Форматирование там, где оно нужно, возвращает
        # tg_helpers.send_message_safe: ступень 0 шлёт с ЯВНЫМ "Markdown",
        # а если Телега не смогла разобрать разметку — повторяет без неё.
        parse_mode=None
    )
)

client = genai.Client(
    api_key=API_KEY_GEMINI,
)