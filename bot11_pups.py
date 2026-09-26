import asyncio
import uuid
import json
import os
import requests
import base64
import re
import prompt
import time
import random
import speech_recognition as sr
from pydub import AudioSegment
import io
from bot_dialogue import dialogue_router
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from aiogram import BaseMiddleware
from aiogram.filters import CommandStart, Filter
from aiogram import Dispatcher, Router, types as aiogram_types, F
from aiogram.types import Message, BufferedInputFile
from aiogram.enums import ParseMode, ChatType
from google.genai import types
from utils import (load_memory, save_memory, append_history, clear_memory,
                   load_airforce_user_key, save_airforce_user_key, load_gemini_user_key, save_gemini_user_key,
                   get_chat_model, get_image_model, get_vision_model, get_music_model, 
                   save_chat_model, save_image_model, save_vision_model, save_music_model, 
                   load_chat_settings, load_image_settings, load_vision_settings, load_music_settings,
                   get_chat_log, save_chat_log, clear_chat_log, # для функции пересказа
                   save_chat_max_history,
                   get_prompt_enhancer_state, save_prompt_enhancer_state, load_prompt_settings)
from summary import is_summary_enabled, set_summary_state, process_pupps_summary, daily_summary_executor
from variables import (CHAT_TRIGGER_WORD, IMAGE_TRIGGER_COMMAND, MUSIC_TRIGGER_COMMAND, PROMPT, MAX_RETRIES, RETRY_DELAY,
                       AIRFORCE_API_URL, AIRFORCE_API_KEY, IMGBB_API_KEY, PUPS_BOT_TOKEN, bot, API_KEY_GEMINI,
                       client,
                       COMFY_URL, IMAGE_PROMPT,
                       VIDEO_TRIGGER_COMMAND, VIDEO_PROMPT, MULTI_EDIT_PROMPT, EDIT_PROMPT,
                       MUSIC_PROMPT, DEFAULT_MUSIC_DURATION, DEFAULT_MUSIC_LANGUAGE)
import pupps_info
import get_models_info
from gemini import priem as gemini_priem, priem_vision as gemini_priem_vision, priem_video as gemini_priem_video, process_voice_turn

IGNORED_BOT_IDS = {
    8718490988,  # Замените на реальный числовой ID целевого бота
}
commands = ['нейро инфо', 'нейро name', 'нейро prompt', 'нейро chat', 'нейро vision', 'нейро image', 'нейро music', 'нейро start', 'нейро stop', 'нейро 0',
            'кибер инфо', 'кибер name', 'кибер prompt', 'кибер chat', 'кибер vision', 'кибер image', 'кибер music', 'кибер start', 'кибер stop', 'кибер 0',
            'пупс инфо', 'пупс chat', 'пупс vision', 'пупс image', 'пупс music', 'пупс start', 'пупс stop', 'пупс 0',
            'няша инфо', 'няша chat', 'няша vision', 'няша image', 'няша music', 'няша start', 'няша stop', 'няша 0',
            'пупс context', 'няша context', 'нейро context', 'кибер context',
            'пупс image local', 'пупс image airforce', 'няша image local', 'няша image airforce',
            'пупс prompt off', 'пупс prompt on', 'пупс prompt', 'няша prompt off', 'няша prompt on', 'няша prompt',
            ]

scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
dp = Dispatcher()
main_router = Router()
dp.include_router(dialogue_router)
dp.include_router(main_router)

VIDEO_ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
DEFAULT_VIDEO_ASPECT = "9:16"   # вертикальный, как прежний дефолт 360x640; поменяй на "16:9", если нужен горизонтальный

MUSIC_LANGUAGE_ALIASES = {
    # код: (русское название, английское)
    "ru": ("русский", "russian"), "en": ("английский", "english"),
    "de": ("немецкий", "german"), "fr": ("французский", "french"),
    "es": ("испанский", "spanish"), "it": ("итальянский", "italian"),
    "ja": ("японский", "japanese"), "ko": ("корейский", "korean"),
    "zh": ("китайский", "chinese"), "pt": ("португальский", "portuguese"),
    "uk": ("украинский", "ukrainian"),
}
_ace_languages_cache = None

async def send_audio_to_chat(chat_id: int, thread_id, audio_bytes: bytes,
                             duration: int = None, title: str = None, performer: str = None):
    """Шлёт mp3 плеером Телеги; при отказе — документом."""
    try:
        await bot.send_audio(
            chat_id=chat_id,
            audio=BufferedInputFile(audio_bytes, filename="comfy_music.mp3"),
            duration=duration, title=title, performer=performer,
            message_thread_id=thread_id
        )
    except Exception as e:
        print(f"send_audio не сработал ({e}), отправляю как документ...")
        await bot.send_document(
            chat_id=chat_id,
            document=BufferedInputFile(audio_bytes, filename="comfy_music.mp3"),
            message_thread_id=thread_id
        )

def get_ace_languages():
    """Список допустимых языков узла TextEncodeAceStepAudio1.5 из /object_info (кэшируется)."""
    global _ace_languages_cache
    if _ace_languages_cache is not None:
        return _ace_languages_cache
    try:
        data = requests.get(f"{COMFY_URL}/object_info/TextEncodeAceStepAudio1.5", timeout=10).json()
        inputs = data.get("TextEncodeAceStepAudio1.5", {}).get("input", {})
        lang_input = (inputs.get("required", {}).get("language")
                      or inputs.get("optional", {}).get("language"))
        if lang_input and isinstance(lang_input[0], list):
            _ace_languages_cache = lang_input[0]
            return _ace_languages_cache
    except Exception as e:
        print(f"ComfyUI: не удалось получить список языков ACE-Step: {e}")
    return None
    
def extract_music_duration(text: str):
    """
    Длительность трека: 180сек / 180 сек / 3 мин / 90s. Диапазон 10–360 сек.
    Возвращает (текст_без_длительности, секунды) или (текст, None).
    """
    match = re.search(r'\b(\d{1,4})\s*(минут[а-яё]*|мин|minutes?|mins?|min|'
                      r'секунд[а-яё]*|сек|seconds?|secs?|sec|s|с)\b', text)
    if not match:
        return text, None

    value = int(match.group(1))
    if match.group(2).startswith(('мин', 'min')):
        value *= 60

    duration = max(10, min(360, value))
    cleaned = text[:match.start()] + text[match.end():]
    return " ".join(cleaned.split()), duration


def _strip_token(text: str, token: str) -> str:
    cleaned = re.sub(rf'\b{re.escape(token)}\b', '', text, count=1)
    return " ".join(cleaned.split())


def extract_music_language(text: str):
    """
    Язык трека. Полное название ('русский', 'english') — где угодно в тексте;
    голый код ('en') — только последним словом (защита от совпадений со словами промта).
    Языком считается только тот, что есть в списке ComfyUI.
    """
    available = get_ace_languages()
    if not available:
        return text, None

    codes = {c.lower() for c in available}
    names_to_code = {}
    for code in available:
        for alias in MUSIC_LANGUAGE_ALIASES.get(code, ()):
            names_to_code[alias] = code.lower()

    tokens = re.findall(r'[а-яёa-z\-]+', text)

    # 1. Полные названия — в любом месте
    for token in tokens:
        if token in names_to_code:
            return _strip_token(text, token), names_to_code[token]

    # 2. Код — только если это последнее слово
    if tokens and tokens[-1] in codes:
        return _strip_token(text, tokens[-1]), tokens[-1]

    return text, None
    
def _parse_music_prompt(response_text: str):
    """Разбирает ответ модели на (tags, lyrics) по маркерам TAGS:/LYRICS:."""
    text = response_text.strip()
    if text.startswith("```"):  # срезаем маркеры кода, если модель их добавила
        text = re.sub(r'^```\w*\s*|\s*```$', '', text).strip()
    text = text.replace('*', '')  # возможный markdown-жирный вокруг маркеров

    match = re.search(r'TAGS:\s*(.*?)\s*LYRICS:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return "", text  # формат не соблюдён — всё считаем лирикой


def _compose_music_system_prompt(language: str = "ru", duration: int = 180) -> str:
    lang_name = MUSIC_LANGUAGE_ALIASES.get(language, (language,))[0]
    return MUSIC_PROMPT + (
        f"\n\nПАРАМЕТРЫ ГЕНЕРАЦИИ:\n"
        f"- Длительность трека: {duration} секунд.\n"
        f"- Язык лирики: {language} ({lang_name}). Тэги TAGS — всегда на английском.\n"
        f"- Включи в TAGS темп в формате 'NN BPM', подходящий жанру."
    )


async def generate_music_prompt(chat_id, thread_id, current_user_message, user_id=None,
                                language="ru", duration=180):
    return await _generate_media_prompt(chat_id, thread_id, current_user_message,
                                        _compose_music_system_prompt(language, duration),
                                        user_id=user_id)

def extract_megapixels(text: str):
    """
    Пользовательский megapixels: mp=1, mp=1.9, mp:0.5 (регистр любой, точка или запятая).
    Диапазон 0.2–2.0. Возвращает (очищенный_текст, mp) или (текст, None).
    """
    match = re.search(r'\bmp\s*[=:]\s*(\d+(?:[.,]\d+)?)\b', text, re.IGNORECASE)
    if not match:
        return text, None

    mp = float(match.group(1).replace(',', '.'))
    mp = max(0.2, min(2.0, mp))

    cleaned_text = text[:match.start()] + text[match.end():]
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text, mp

def extract_seed(text: str):
    """
    Пользовательский сид: seed=123, seed:123, сид=123 (регистр любой).
    Возвращает (очищенный_текст, seed) или (текст, None).
    Сид вырезается из текста — LLM его не увидит и не утащит в промт.
    """
    match = re.search(r'\b(?:seed|сид)\s*[=:]\s*(\d{1,16})\b', text, re.IGNORECASE)
    if not match:
        return text, None

    seed = int(match.group(1)) % (2**53)  # тот же диапазон, что у random.randint в сабмитах

    cleaned_text = text[:match.start()] + text[match.end():]
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text, seed

def _pick_seed(user_seed=None) -> int:
    """Пользовательский сид или случайный — единая точка для всех workflow."""
    if user_seed is not None:
        return int(user_seed)
    return random.randint(0, 2**53 - 1)

def extract_video_aspect_ratio(text: str):
    """Ищет соотношение сторон для видео (1:1, 2:3, ..., 21:9). Возвращает (текст_без_него, ratio)."""
    for ratio in VIDEO_ASPECT_RATIOS:
        # \b защищает: "2:3" не вырежется из "12:30", "1:1" не зацепится внутри "21:9"
        match = re.search(rf'\b{ratio}\b', text)
        if match:
            cleaned = text[:match.start()] + text[match.end():]
            cleaned = " ".join(cleaned.split())
            return cleaned, ratio
    return text, None
    
ASPECT_RATIO_FALLBACK = {
    "1:1": "1:1 (Square)", "2:3": "2:3 (Portrait Photo)", "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standart)", "4:3": "4:3 (Standart)", "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)", "21:9": "21:9 (Ultrawide)",
}
_resolution_selector_options = None

def _get_resolution_selector_options():
    """Валидные строки aspect_ratio из ComfyUI (кэшируется)."""
    global _resolution_selector_options
    if _resolution_selector_options:
        return _resolution_selector_options
    try:
        data = requests.get(f"{COMFY_URL}/object_info/ResolutionSelector", timeout=10).json()
        aspect = data.get("ResolutionSelector", {}).get("input", {}).get("required", {}).get("aspect_ratio")
        if aspect and isinstance(aspect[0], list):
            _resolution_selector_options = aspect[0]
            return _resolution_selector_options
    except Exception as e:
        print(f"ComfyUI: не удалось получить варианты aspect_ratio: {e}")
    return None

DEFAULT_VIDEO_MP = 0.3
_mp_options_cache = None

def _resolve_mp(mp) -> float:
    """None -> дефолт; защита от null в workflow."""
    return float(mp) if mp is not None else DEFAULT_VIDEO_MP

def _get_mp_options():
    """Допустимые значения megapixels узла ResolutionSelector из /object_info (кэшируется)."""
    global _mp_options_cache
    if _mp_options_cache is not None:
        return _mp_options_cache
    try:
        data = requests.get(f"{COMFY_URL}/object_info/ResolutionSelector", timeout=10).json()
        mp_input = data.get("ResolutionSelector", {}).get("input", {}).get("required", {}).get("megapixels")
        if mp_input and isinstance(mp_input[0], list):
            _mp_options_cache = sorted(float(x) for x in mp_input[0])
            return _mp_options_cache
    except Exception as e:
        print(f"ComfyUI: не удалось получить варианты megapixels: {e}")
    return None

def _clamp_mp_to_options(mp: float) -> float:
    """Прижимает mp к ближайшему допустимому значению узла (если список известен)."""
    options = _get_mp_options()
    if not options:
        return mp
    return min(options, key=lambda x: abs(x - mp))

def _map_aspect_ratio_to_comfy(ratio: str) -> str:
    """'16:9' -> '16:9 (Widescreen)' — точное значение для узла 409."""
    options = _get_resolution_selector_options()
    if options:
        for opt in options:
            if opt.startswith(ratio):
                return opt
        raise RuntimeError(f"ComfyUI: для {ratio} нет варианта в ResolutionSelector. Доступные: {options}")
    fallback = ASPECT_RATIO_FALLBACK.get(ratio)
    if fallback:
        return fallback
    raise RuntimeError(f"Не удалось определить строку aspect_ratio для {ratio}")

def extract_video_duration(text: str):
    """
    Ищет длительность видео (2–30 сек). Понимает: 25sec, 25 sec, 25s, 25 сек, 25с, 25 секунд.
    Число обязано иметь единицу измерения — так не цепляются случайные цифры и разрешения.
    Выход за диапазон прижимается к границам (1 -> 2, 45 -> 30).
    Возвращает (очищенный_текст, длительность) или (текст, None), если не указана.
    """
    # Длинные варианты в начале, чтобы срабатывали раньше коротких
    match = re.search(r'\b(\d{1,3})\s*(?:секунд[а-яё]*|second[zs]?|sec|s|сек|с)\b', text)
    if not match:
        return text, None

    duration = max(2, min(30, int(match.group(1))))

    cleaned_text = text[:match.start()] + text[match.end():]
    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text, duration

def extract_resolution(text: str):
    """
    Ищет в тексте разрешение вида W:H, WxH, WхH (например, 480:640, 768x1024).
    Возвращает (очищенный_текст, (width, height)) или (текст, None), если разрешения нет.
    
    Правила:
    - каждая сторона от 64 до 1024 px (больше — масштабируется пропорционально);
    - итоговые значения округляются до кратных 8 (требование латентов ComfyUI);
    - значения меньше 64 считаются НЕ разрешением (не трогаем текст).
    """
    # Обе стороны должны быть 2-4-значными — благодаря этому соотношения сторон
    # вроде "16:9" или "3:4" (с однозначными числами) не перехватываются
    match = re.search(r'\b(\d{2,4})\s*[:xх×]\s*(\d{2,4})\b', text)
    if not match:
        return text, None

    width, height = int(match.group(1)), int(match.group(2))

    # «10:20» и прочая мелочь — это не разрешение, оставляем текст как есть
    if width < 64 or height < 64:
        return text, None

    # Максимум 1024x1024: при превышении сжимаем пропорционально (1080:1920 -> 576:1024)
    if width > 1024 or height > 1024:
        scale = 1024 / max(width, height)
        width, height = int(width * scale), int(height * scale)

    # Латенты ComfyUI требуют размеры, кратные 8
    width = max(64, round(width / 8) * 8)
    height = max(64, round(height / 8) * 8)

    # Вырезаем разрешение из текста и чистим двойные пробелы
    cleaned_text = text[:match.start()] + text[match.end():]
    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text, (width, height)
        
def extract_aspect_ratio(text: str):
    """
    Ищет строгое совпадение соотношения сторон из списка allowed_ratios.
    Возвращает (очищенный_текст, aspect_ratio).
    """
    allowed_ratios = ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"]
    
    for ratio in allowed_ratios:
        if ratio in text:
            # Удаляем соотношение сторон из текста
            cleaned_text = text.replace(ratio, "").strip()
            # Очищаем от возможных двойных пробелов, которые могли остаться
            cleaned_text = " ".join(cleaned_text.split())
            print(cleaned_text)
            return cleaned_text, ratio
            
    return text, "1:1"

async def send_log_to_telegram(error_message: str, handler_name: str, chat_id: int, thread_id: int, user_info="Неизвестен"):
    """Отправляет отформатированное сообщение об ошибке в отдельный лог-чат."""
    # Экранируем обратные кавычки для Markdown
    safe_error = str(error_message).replace('`', "'")
    full_log_message = f'❌ {user_info}, ошибка в обработчике {handler_name}:\n\n{safe_error}'
    
    try:
        await bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=full_log_message
        )
    except Exception as e:
        print(f"Ошибка: Не удалось отправить лог в Telegram. {e}")
        
# Функция для определения наличия медиа и добавления префикса
def format_message_text(message: aiogram_types.Message) -> str:
    prefix = ""
    # Проверяем наличие фото, видео, анимации или документа
    if message.photo or message.video or message.animation or message.document:
        prefix = "[Медиа] "
    
    # Берем текст или описание (caption)
    text = message.text or message.caption or ""
    user_name = message.from_user.full_name if message.from_user else "Система"
    
    return f"{prefix}{user_name}: {text}".strip()
    
def upload_to_imgbb(file_bytes, url_to_image, chat_id, thread_id):
    api_key = IMGBB_API_KEY
    url = "https://api.imgbb.com/1/upload"
    max_retries = 2  # Количество попыток
    retry_delay = 5  # Пауза между попытками в секундах
    
    if file_bytes:
        payload = {
            "key": api_key,
            "image": base64.b64encode(file_bytes), # ImgBB любит base64
        }
        
    else:
        payload = {
            "key": api_key,
            "image": url_to_image,
        }
    
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(url, 
                                payload, 
                                timeout=30)
            res.raise_for_status()
            data = res.json()
            
            if data.get("success"):
                link = data["data"]["url"]
                print(f"✅ Попытка {attempt}: ImgBB: {link}")
                return link
            else:
                print(f"⚠️ Попытка {attempt}: Ошибка в ответе API")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Попытка {attempt} завершилась ошибкой: {e}")
    
        if attempt < max_retries:
            time.sleep(retry_delay)
    if url_to_image:
        tg_url = f"https://api.telegram.org/bot{PUPS_BOT_TOKEN}/sendMessage"
        msg_payload = {
            "chat_id": chat_id,
            "text": f"⚠️ {CHAT_TRIGGER_WORD.capitalize()} не смог скачать картинку, но вот тебе прямая ссылка:\n{url_to_image}",
            "message_thread_id": thread_id
        }
        requests.post(tg_url, json=msg_payload)
        return url_to_image

async def generate_response(chat_id: int, 
                            thread_id: int, 
                            system_prompt: str, 
                            current_user_message: str, 
                            user_id: int = None) -> str:
                            
    chat_model = get_chat_model(chat_id)
    
    if chat_model == "gemini":
        user_key = load_gemini_user_key(user_id)
        active_key = user_key if user_key else API_KEY_GEMINI
        return await gemini_priem(chat_id, current_user_message, user_key=active_key)
    else:
        user_key = load_airforce_user_key(user_id)
        active_key = user_key if user_key else AIRFORCE_API_KEY
        return await asyncio.to_thread(_sync_airforce_request,
                                       chat_id, 
                                       thread_id, 
                                       system_prompt, 
                                       current_user_message, 
                                       active_key
        )

def _sync_airforce_request(chat_id: int, thread_id: int, system_prompt: str, current_user_message: str, user_key: str = None) -> str:
    """Синхронная функция: Загружает память, делает запрос, удаляет рекламу, сохраняет память."""
    error_paid = ''
    memory = load_memory(chat_id)
    chat_model = get_chat_model(chat_id)
    messages_to_send = [{"role": "system", "content": system_prompt}]

    for item in memory.get("history", []):
        role = item.get("role")
        content = item.get("content")
        if not content and "parts" in item:
             content = item["parts"][0]["text"]
             
        if role in ["user", "assistant"] and content: 
            messages_to_send.append({"role": role, "content": content})
    
    payload = {
      "model": chat_model,
      "messages": messages_to_send,
      "temperature": 1,
      "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {user_key}", 
        "Content-Type": "application/json"
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"💬 Попытка {attempt}/{MAX_RETRIES} запроса к AirForce...")
        
        try:
            response = requests.post(AIRFORCE_API_URL, 
                                     headers=headers, 
                                     json=payload,  
                                     timeout=180)
            
            # Обработка ошибок 429/5xx
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue 
                else:
                    raise RuntimeError(f"Превышен лимит попыток. HTTP {response.status_code}: {response.text}")
            
            if 400 <= response.status_code < 500:
                response.raise_for_status()
                
            data = response.json()
            
            try:
                if data["error"]["message"] == "This model requires an active subscription or a positive Pay-as-you-Go balance. Subscribe or top up at https://api.airforce/dashboard, or use a free model.":
                    error_paid = "Для этой модели требуется активная подписка или положительный баланс. Подпишитесь или пополните счет на сайте https://api.airforce/dashboard или воспользуйтесь бесплатной моделью."
            except Exception as e:
                print(e)
                
            response_content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Проверка на текстовые ошибки лимитов
            if re.search(r'ratelimit exceeded|quota|error', response_content, re.IGNORECASE):
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise RuntimeError(f"Ошибка API в ответе: {response_content}")
            
            if response_content:
                # Удаление рекламы
                final_content = response_content.replace('Want best roleplay experience?', '')
                final_content = final_content.replace('https://llmplayground.net', '')
                final_content = final_content.replace('discord.gg/airforce', '')
                final_content = final_content.replace('Need proxies cheaper than the market?\nhttps://op.wtf', '')
                final_content = final_content.strip()
                print(final_content)

                updated_memory = append_history(memory, my_response=final_content, chat_id=chat_id)
                save_memory(chat_id, updated_memory)
                if final_content:
                    return final_content
                else:
                    continue
            
            raise RuntimeError(f"Пустой ответ от API: {json.dumps(data)}")

        except Exception as e:
            print(e)
            if error_paid:
                raise RuntimeError(error_paid)
                break
            elif attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            raise RuntimeError(f"Ошибка после {MAX_RETRIES} попыток: {e}")
    raise RuntimeError("Не удалось получить ответ.")
    
async def generate_vision_response(chat_id: int,
                                   thread_id: int,
                                   system_prompt: str,
                                   current_user_message: str, 
                                   base64_image: str, 
                                   user_id: int = None) -> str:
    vision_model = get_vision_model(chat_id)
    
    if vision_model == "gemini":
        user_key = load_gemini_user_key(user_id)
        active_key = user_key if user_key else API_KEY_GEMINI
        return await gemini_priem_vision(chat_id, current_user_message, base64_image, user_key=active_key, system_prompt=system_prompt)
    else:
        user_key = load_airforce_user_key(user_id)
        active_key = user_key if user_key else AIRFORCE_API_KEY
        return await asyncio.to_thread(_sync_vision_request, 
                                       chat_id, 
                                       thread_id, 
                                       system_prompt, 
                                       current_user_message, 
                                       base64_image, 
                                       active_key
        )

def _sync_vision_request(chat_id: int, thread_id: int, system_prompt: str, current_user_message: str, base64_image: str, user_key: str = None) -> str:
    """Синхронная функция для обработки фото + текста."""
    error_paid = ''
    memory = load_memory(chat_id)
    messages_to_send = [{"role": "system", "content": system_prompt}]
    vision_model = get_vision_model(chat_id)
    
    for item in memory.get("history", []):
        if item.get("role") in ["user", "assistant"]:
            messages_to_send.append({"role": item["role"], "content": item["content"]})
    
    '''
    current_content = [
        {"type": "text", "text": current_user_message},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
    ]
    '''
    
    images = base64_image if isinstance(base64_image, list) else [base64_image]
    current_content = [{"type": "text", "text": current_user_message}]
    for img in images:
        current_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })
    
    messages_to_send.append({"role": "user", "content": current_content})
    
    payload = {
        "model": vision_model,
        "messages": messages_to_send,
        "temperature": 1
    }
    
    headers = {
        "Authorization": f"Bearer {user_key}", 
        "Content-Type": "application/json"
    }
    
    # 4. Цикл попыток
    for attempt in range(1, MAX_RETRIES + 1):
        log_message = f"💬 Попытка {attempt}/{MAX_RETRIES}..."
        print(f"💬 Попытка {attempt}/{MAX_RETRIES} запроса к AirForce...")
        
        try:
            response = requests.post(AIRFORCE_API_URL, 
                                     headers=headers, 
                                     json=payload, 
                                     #proxies=proxies, 
                                     timeout=300)
            
            # Обработка ошибок 429/5xx
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue 
                else:
                    raise RuntimeError(f"Превышен лимит попыток. HTTP {response.status_code}: {response.text}")

            if 400 <= response.status_code < 500:
                response.raise_for_status()
                
            data = response.json()
            
            try:
                if data["error"]["message"] == "This model requires an active subscription or a positive Pay-as-you-Go balance. Subscribe or top up at https://api.airforce/dashboard, or use a free model.":
                    error_paid = "Для этой модели требуется активная подписка или положительный баланс. Подпишитесь или пополните счет на сайте https://api.airforce/dashboard или воспользуйтесь бесплатной моделью."
            except Exception as e:
                print(e)
            
            response_content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Проверка на текстовые ошибки лимитов
            if re.search(r'ratelimit exceeded|quota|error', response_content, re.IGNORECASE):
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise RuntimeError(f"Ошибка API в ответе: {response_content}")
            
            # Успех
            if response_content:
                # Удаление рекламы
                final_content = response_content.replace('Want best roleplay experience?', '')
                final_content = final_content.replace('https://llmplayground.net', '')
                final_content = final_content.replace('discord.gg/airforce', '')
                final_content = final_content.replace('Need proxies cheaper than the market?\nhttps://op.wtf', '')
                print(final_content)
                
                # СОХРАНЕНИЕ В ПАМЯТЬ: Сохраняем пометку [ФОТО] вместо Base64
                updated_memory = append_history(memory, my_response=final_content, chat_id=chat_id)
                save_memory(chat_id, updated_memory)
                
                if final_content:
                    return final_content
                else:
                    continue
            
        except Exception as e:
            print(e)
            if error_paid:
                raise RuntimeError(error_paid)
                break
            elif attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            raise RuntimeError(f"Ошибка после {MAX_RETRIES} попыток: {e}")
            
async def generate_flf2v_prompt(chat_id, thread_id, current_user_message, images_b64: list,
                                user_id=None, duration=None):
    """Промт для видео-перехода между двумя кадрами: vision-модель видит оба + контекст чата."""
    memory = load_memory(chat_id)
    request_core = current_user_message.split(":", 1)[-1].strip()
    recent_users = [m for m in memory.get("history", [])[-6:] if m.get("role") == "user"]
    already_saved = bool(request_core) and any(request_core in m.get("content", "") for m in recent_users)

    if not already_saved:
        memory = append_history(memory, opponent_message=f"[Медиа] {current_user_message}", chat_id=chat_id)
        save_memory(chat_id, memory)
        
    # Улучшатель выключен — сырой запрос уходит в генератор напрямую
    if not get_prompt_enhancer_state(chat_id):
        return request_core

    return await generate_vision_response(chat_id, thread_id,
                                          _compose_video_system_prompt(duration),
                                          current_user_message, images_b64,
                                          user_id=user_id)
            
async def _generate_media_prompt(chat_id, thread_id, current_user_message, system_prompt, user_id=None):
    """Универсальный генератор промтов (картинки/видео): текущая чат-модель + контекст чата."""
    chat_model = get_chat_model(chat_id)
    memory = load_memory(chat_id)

    # Антидубль: сообщение мог уже сохранить чат-роутер
    request_core = current_user_message.split(":", 1)[-1].strip()
    recent_users = [m for m in memory.get("history", [])[-6:] if m.get("role") == "user"]
    already_saved = bool(request_core) and any(request_core in m.get("content", "") for m in recent_users)

    if not already_saved:
        memory = append_history(memory, opponent_message=current_user_message, chat_id=chat_id)
        save_memory(chat_id, memory)
    
    # Улучшатель выключен — сырой запрос уходит в генератор напрямую
    if not get_prompt_enhancer_state(chat_id):
        return request_core

    if chat_model == "gemini":
        user_key = load_gemini_user_key(user_id)
        active_key = user_key if user_key else API_KEY_GEMINI
        return await gemini_priem(chat_id, current_user_message, user_key=active_key,
                                  system_prompt=system_prompt)
    else:
        user_key = load_airforce_user_key(user_id)
        active_key = user_key if user_key else AIRFORCE_API_KEY
        return await asyncio.to_thread(_sync_airforce_request, chat_id, thread_id,
                                       system_prompt, current_user_message, active_key)

async def send_video_to_chat(chat_id: int, thread_id, video_bytes: bytes, width: int = None, height: int = None):
    """Шлёт видео; при отказе Телеги — документом. width/height опциональны."""
    try:
        await bot.send_video(chat_id=chat_id, video=BufferedInputFile(video_bytes, filename="comfy_video.mp4"),
                             width=width, height=height, supports_streaming=True, message_thread_id=thread_id)
    except Exception as e:
        print(f"send_video не сработал ({e}), отправляю как документ...")
        await bot.send_document(chat_id=chat_id, document=BufferedInputFile(video_bytes, filename="comfy_video.mp4"),
                                message_thread_id=thread_id)

async def generate_image_prompt(chat_id, thread_id, current_user_message, user_id=None):
    return await _generate_media_prompt(chat_id, thread_id, current_user_message, IMAGE_PROMPT, user_id)

async def generate_video_i2v_prompt(chat_id: int,
                                    thread_id: int,
                                    current_user_message: str,
                                    encoded_image: str,
                                    user_id: int = None,
                                    duration: int = None) -> str:
    """
    Промт для видео из картинки (i2v): активная vision-модель видит и саму
    картинку, и контекст чата (историю переписки).
    """
    # 1. Сохраняем запрос в память с пометкой [Медиа] — так модель получит контекст.
    #    Для gemini priem_vision заменит эту запись версией с картинкой,
    #    для airforce дубль уберёт антидубль-pop из Шага 3.
    memory = load_memory(chat_id)
    request_core = current_user_message.split(":", 1)[-1].strip()
    recent_users = [m for m in memory.get("history", [])[-6:] if m.get("role") == "user"]
    already_saved = bool(request_core) and any(request_core in m.get("content", "") for m in recent_users)

    if not already_saved:
        memory = append_history(memory, opponent_message=f"[Медиа] {current_user_message}", chat_id=chat_id)
        save_memory(chat_id, memory)
        
    # Улучшатель выключен — сырой запрос уходит в генератор напрямую
    if not get_prompt_enhancer_state(chat_id):
        return request_core

    # 2. Генерация через vision-модель (управляется командой "пупс vision ...")
    return await generate_vision_response(chat_id, thread_id, _compose_video_system_prompt(duration),
                                           current_user_message, encoded_image,
                                           user_id=user_id)
                                           
async def generate_edit_prompt(chat_id: int,
                               thread_id: int,
                               current_user_message: str,
                               encoded_image: str,
                               user_id: int = None) -> str:
    """
    Промт для редактирования одной картинки (i2i): активная vision-модель
    видит картинку, запрос пользователя и контекст чата.
    """
    memory = load_memory(chat_id)
    request_core = current_user_message.split(":", 1)[-1].strip()
    recent_users = [m for m in memory.get("history", [])[-6:] if m.get("role") == "user"]
    already_saved = bool(request_core) and any(request_core in m.get("content", "") for m in recent_users)

    if not already_saved:
        memory = append_history(memory, opponent_message=f"[Медиа] {current_user_message}", chat_id=chat_id)
        save_memory(chat_id, memory)
        
    # Улучшатель выключен — сырой запрос уходит в генератор напрямую
    if not get_prompt_enhancer_state(chat_id):
        return request_core

    return await generate_vision_response(chat_id, thread_id, EDIT_PROMPT,
                                          current_user_message, encoded_image,
                                          user_id=user_id)
                                           
async def generate_multi_edit_prompt(chat_id, thread_id, current_user_message, images_b64: list, user_id=None):
    """Промт для совмещения двух картинок: vision-модель видит обе + контекст чата."""
    memory = load_memory(chat_id)
    request_core = current_user_message.split(":", 1)[-1].strip()
    recent_users = [m for m in memory.get("history", [])[-6:] if m.get("role") == "user"]
    already_saved = bool(request_core) and any(request_core in m.get("content", "") for m in recent_users)

    if not already_saved:
        memory = append_history(memory, opponent_message=f"[Медиа] {current_user_message}", chat_id=chat_id)
        save_memory(chat_id, memory)
        
    # Улучшатель выключен — сырой запрос уходит в генератор напрямую
    if not get_prompt_enhancer_state(chat_id):
        return request_core

    return await generate_vision_response(chat_id, thread_id, MULTI_EDIT_PROMPT,
                                           current_user_message, images_b64, user_id=user_id)

def _compose_video_system_prompt(duration=None) -> str:
    system_prompt = VIDEO_PROMPT
    if duration:
        system_prompt += (
            f"\n\nПАРАМЕТР ГЕНЕРАЦИИ: длительность ролика — {duration} секунд.\n"
            f"Учитывай её строго по разделу «ДЛИТЕЛЬНОСТЬ»."
        )
    else:
        system_prompt += (
            "\n\nПАРАМЕТР ГЕНЕРАЦИИ: длительность не задана — её подберёт сама модель LTX. "
            "Описывай события естественно и полно, не подгоняя под конкретный хронометраж."
        )
    return system_prompt

async def generate_video_prompt(chat_id, thread_id, current_user_message, user_id=None, duration: int = None):
    return await _generate_media_prompt(chat_id, thread_id, current_user_message, _compose_video_system_prompt(duration), user_id)

def generate_media_sync(user_id, prompt_text, chat_id, thread_id, image_url=[], is_music=False, aspect_ratio="1:1"):
    url = "https://api.airforce/v1/images/generations"
    user_key = load_airforce_user_key(user_id)
    active_key = user_key if user_key else AIRFORCE_API_KEY
    headers = {"Authorization": f"Bearer {active_key}", "Content-Type": "application/json"}
    
    if is_music:
        music_model = get_music_model(chat_id)
        if isinstance(prompt_text, dict):
            payload = {
              "model": music_model,
              "prompt": prompt_text['lyrics'],
              "n": 1,
              "size": "1024x1024",
              "response_format": "url",
              "sse": True,
              "custom": True,
              "instrumental": False,
              "style": prompt_text['style']
            }

        elif "instrumental" in prompt_text:
            prompt_text = prompt_text.replace('instrumental', '').strip()
            payload = {
              "model": music_model,
              "prompt": prompt_text,
              "n": 1,
              "size": "1024x1024",
              "response_format": "url",
              "sse": True,
              "custom": False,
              "instrumental": True
            }
        else:
            payload = {
              "model": music_model,
              "prompt": prompt_text,
              "n": 1,
              "size": "1024x1024",
              "response_format": "url",
              "sse": True,
              "custom": False,
              "instrumental": False
            }
    else:
        image_model = get_image_model(chat_id)
        payload = {
          "model": image_model,
          "prompt": prompt_text,
          "n": 1,
          "size": "1024x1024",
          "response_format": "url",
          "sse": True,
          "aspectRatio": aspect_ratio,
          "resolution": "1k",
          "image_urls": image_url,
          "mode": 'normal',
        }

    for attempt in range(1, MAX_RETRIES + 1):
        print(f'Попытка {attempt}')
        
        try:
            data = None
            with requests.post(url, 
                               headers=headers, 
                               json=payload, 
                               stream=True,  
                               timeout=300) as resp:
                print(f"DEBUG: Статус ответа API: {resp.status_code}") # Лог статуса HTTP
                if resp.status_code == 429:
                    print("DEBUG: Превышен лимит (429). Ожидание...")
                    time.sleep(2)
                    continue
                if 500 <= resp.status_code < 600:
                    time.sleep(5)
                    continue
                    
                resp.raise_for_status()
                
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            content = line_str[6:].strip()
                            if content == "[DONE]":
                                break
                            try:
                                data = json.loads(content)
                                print(f"DEBUG: Парсинг JSON успешен. Ключи: {list(data.keys())}")
                            except json.JSONDecodeError:
                                continue
            if data is None:
                print("Ошибка: API не прислало данных в потоке.")
                time.sleep(5)
                continue
                
            if "data" not in data or not data["data"]:
                print(f"Ошибка: Некорректный формат ответа: {data}")
                time.sleep(5)
                continue

            media_url = data["data"][0]["url"]
            print(f"Скачивание: {media_url}")
            return media_url
            
        except Exception as e:
            print(f"Непредвиденная ошибка: {e}. Повтор...")
            time.sleep(5)
            continue
    
    raise RuntimeError("Не удалось сгенерировать медиа после всех попыток.")
    
# ================== ЛОКАЛЬНАЯ ГЕНЕРАЦИЯ ЧЕРЕЗ ComfyUI ==================

def _parse_comfy_queue(queue_data: dict):
    """
    Достаёт prompt_id из ответа GET /queue.
    Формат элемента: [number, prompt_id, prompt, extra_data, outputs_to_execute].
    prompt_id — строка (UUID) на позиции 1; number — int на позиции 0.
    """
    running_ids, pending_ids = [], []
    for key, dst in (("queue_running", running_ids), ("queue_pending", pending_ids)):
        for item in (queue_data.get(key) or []):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            # prompt_id — строка на item[1]; на случай другого порядка парсим оба
            if isinstance(item[1], str):
                dst.append(item[1])
            elif isinstance(item[0], str):
                dst.append(item[0])
    return running_ids, pending_ids
    
def get_comfy_queue_position(prompt_id: str):
    """(0, True) — выполняется на GPU; (N, False) — ждёт, позиция N; (None, False) — не найдена."""
    try:
        queue_data = requests.get(f"{COMFY_URL}/queue", timeout=10).json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"ComfyUI: не удалось получить очередь: {e}")
        return None, False
    running_ids, pending_ids = _parse_comfy_queue(queue_data)
    if prompt_id in running_ids:
        return 0, True
    if prompt_id in pending_ids:
        return pending_ids.index(prompt_id) + 1, False
    return None, False

def _format_queue_info(position, running) -> str:
    """Строка о состоянии задачи в очереди для статус-сообщения."""
    if running:
        return "🚀 GPU уже взял задачу — генерация выполняется.\n"
    if position:
        return f"📋 Позиция в очереди ComfyUI: {position}.\n"
    return "📋 Задача поставлена в очередь ComfyUI.\n"
    
def _submit_comfy_workflow(workflow: dict) -> str:
    """Ставит workflow в очередь ComfyUI. Возвращает prompt_id."""
    response = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"ComfyUI: ошибка запроса (HTTP {response.status_code}): {response.text}")
    return response.json()['prompt_id']

def wait_comfy_result(prompt_id: str, timeout: int = 3600) -> bytes:
    """Ждёт завершения задачи в ComfyUI и скачивает результат."""
    url = poll_comfyui(prompt_id, timeout=timeout)
    return _download_comfy_result(url)

def cancel_comfy_task(prompt_id: str, verify_timeout: int = 30) -> str:
    """
    Отменяет задачу в ComfyUI и ПРОВЕРЯЕТ результат.
    ВАЖНО: /interrupt — мягкий флаг. Во время загрузки моделей (минуты у LTX)
    он не действует — задача остановится только на границе следующего узла/шага.
    """
    # 1. Где сейчас задача?
    try:
        resp = requests.get(f"{COMFY_URL}/queue", timeout=10)
        resp.raise_for_status()
        queue_data = resp.json()
    except Exception as e:
        return f"отменить не удалось (не удалось получить очередь: {e})"

    running_ids, pending_ids = _parse_comfy_queue(queue_data)

    # 2. Ждёт в очереди → удаляем из очереди (как Cancel в UI ComfyUI для pending)
    if prompt_id in pending_ids:
        try:
            resp = requests.post(f"{COMFY_URL}/queue", json={"delete": [prompt_id]}, timeout=10)
            resp.raise_for_status()
            for _ in range(5):  # верификация: ждём исчезновения из очереди
                time.sleep(1)
                try:
                    q = requests.get(f"{COMFY_URL}/queue", timeout=10).json()
                    _, pending_now = _parse_comfy_queue(q)
                    if prompt_id not in pending_now:
                        return "задача удалена из очереди ожидания"
                except Exception:
                    continue
            return "задача удалена из очереди, но всё ещё видна в очереди — проверь вручную"
        except Exception as e:
            return f"отменить не удалось (ошибка удаления из очереди: {e})"

    # 3. Выполняется → interrupt + ожидание подтверждения остановки
    if prompt_id in running_ids:
        try:
            resp = requests.post(f"{COMFY_URL}/interrupt", timeout=10)
            resp.raise_for_status()
        except Exception as e:
            return f"отменить не удалось (ошибка прерывания: {e})"

        start = time.time()
        while time.time() - start < verify_timeout:
            time.sleep(2)
            try:
                q = requests.get(f"{COMFY_URL}/queue", timeout=10).json()
                running_now, _ = _parse_comfy_queue(q)
                if prompt_id not in running_now:
                    return "задача прервана, GPU освобождён"
            except Exception:
                continue
        return ("прерывание отправлено, но задача всё ещё выполняется — "
                "если сейчас идёт загрузка моделей, interrupt сработает только после её завершения")

    # 4. Ни в очереди, ни в выполнении
    return "задача уже не в очереди (завершилась или была снята ранее)"

def _load_comfy_workflow(filename: str):
    """Безопасно грузит шаблон workflow, чтобы бот не падал, если файла нет."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ ComfyUI: файл {filename} не найден! Локальная генерация недоступна.")
        return None

#workflow_t2v_video = _load_comfy_workflow('video_ltx2_3_t2v.json')  # текст -> видео
workflow_t2v_video = _load_comfy_workflow('video_ltx2_5_t2v.json')  # текст -> видео (LTX-2.5)
#workflow_i2v_video = _load_comfy_workflow('video_ltx2_3_i2v.json')  # картинка -> видео
workflow_i2v_video = _load_comfy_workflow('video_ltx2_5_i2v.json')  # картинка -> видео (LTX-2.5)
workflow_t2i = _load_comfy_workflow('workflow_t2i.json')      # текст -> картинка
workflow_multi = _load_comfy_workflow('flux_multiple_input.json')  # 2 картинки -> 1 результат
workflow_audio = _load_comfy_workflow('audio_ace_step1_5_xl_turbo.json')  # музыка (ACE-Step 1.5)
workflow_flf2v_video = _load_comfy_workflow('video_ltx2_5_flf2v.json')  # 2 кадра -> видео (LTX-2.5)

def _prune_workflow_to_output(workflow: dict, output_node_id: str) -> dict:
    """Оставляет только узлы, нужные для выходного узла output_node_id (BFS по связям входов).
    В multiple-input workflow выкидывает одиночную ветку 75:* — иначе ComfyUI выполнит обе
    и сгенерит лишнюю картинку."""
    keep = set()
    stack = [output_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in keep or node_id not in workflow:
            continue
        keep.add(node_id)
        for value in workflow[node_id].get("inputs", {}).values():
            # связь в API-формате: [node_id (строка), slot]
            if (isinstance(value, list) and len(value) == 2
                    and isinstance(value[0], str) and value[0] in workflow):
                stack.append(value[0])
    return {node_id: workflow[node_id] for node_id in workflow if node_id in keep}

# ID узла с шириной/высотой в workflow_t2i.json (узел EmptyLatentImage, например "57:5").
# Если оставить None — узел ищется автоматически (первый, у которого в inputs есть width и height).
LATENT_NODE_ID = None

def _set_workflow_resolution(workflow: dict, width: int, height: int) -> bool:
    """Прописывает разрешение в узел EmptyLatentImage (или аналогичный)."""
    # 1. Явно заданный ID — приоритет
    if LATENT_NODE_ID and LATENT_NODE_ID in workflow:
        workflow[LATENT_NODE_ID]["inputs"]["width"] = width
        workflow[LATENT_NODE_ID]["inputs"]["height"] = height
        return True

    # 2. Автопоиск: первый узел с width/height в inputs
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if "width" in inputs and "height" in inputs:
            inputs["width"] = width
            inputs["height"] = height
            return True

    return False

workflow_edit = _load_comfy_workflow('flux_image_edit.json')  # редактирование

def is_comfyui_online(timeout: int = 5) -> bool:
    """Пингует ComfyUI через /system_stats. True — сервер доступен."""
    try:
        response = requests.get(f"{COMFY_URL}/system_stats", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

COMFY_MEDIA_KEYS = ('images', 'gifs', 'videos', 'audio')

def poll_comfyui(prompt_id: str, timeout: int = 3600) -> str:
    """Опрашивает ComfyUI до завершения генерации. По таймауту отменяет задачу."""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            cancel_info = cancel_comfy_task(prompt_id)
            raise RuntimeError(f"ComfyUI: превышено время ожидания ({timeout} сек). {cancel_info}.")

        try:
            history = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30).json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"ComfyUI: ошибка опроса статуса: {e}")
            time.sleep(2)
            continue

        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get('status', {})
            if status.get('status_str') == 'error':
                raise RuntimeError(f"ComfyUI: ошибка выполнения workflow: {status}")

            for node_id, node_output in entry.get('outputs', {}).items():
                for key in COMFY_MEDIA_KEYS:
                    items = node_output.get(key)
                    if items:
                        item = items[0]
                        return f"{COMFY_URL}/view?filename={item['filename']}&subfolder={item.get('subfolder', '')}&type={item.get('type', 'output')}"

            if status.get('completed'):
                outputs_dump = json.dumps(entry.get('outputs', {}), ensure_ascii=False)[:500]
                raise RuntimeError(f"ComfyUI: генерация завершилась без результата. Outputs: {outputs_dump}")

        time.sleep(2)

def _download_comfy_result(img_url: str) -> bytes:
    """Скачивает готовую картинку (Телега не может открыть локальную ссылку 192.168.x.x)."""
    img_response = requests.get(img_url, timeout=120)
    if img_response.status_code != 200:
        raise RuntimeError(f"ComfyUI: не удалось скачать результат (HTTP {img_response.status_code}).")
    return img_response.content
    
def submit_video_flf2v_comfy(prompt_text: str, first_frame_bytes: bytes, last_frame_bytes: bytes,
                             aspect_ratio: str = DEFAULT_VIDEO_ASPECT, duration: int = 10, seed: int = None,
                             mp: float = None):
    """Первый и последний кадр -> видео через ComfyUI (LTX-2.5 flf2v). Возвращает (prompt_id, seed)."""
    if workflow_flf2v_video is None:
        raise RuntimeError("ComfyUI: video_ltx2_5_flf2v.json не загружен.")

    first_name = upload_image_to_comfy(first_frame_bytes)
    last_name = upload_image_to_comfy(last_frame_bytes)
    if not first_name or not last_name:
        raise RuntimeError("ComfyUI: не удалось загрузить один из кадров.")

    first_frame_node_id = "31"      # LoadImage "Load First Frame"
    last_frame_node_id = "39"       # LoadImage "Load Last Frame"
    prompt_node_id = "251:252"      # PrimitiveStringMultiline "Prompt"
    selector_node_id = "259"        # ResolutionSelector
    duration_node_id = "251:198"    # PrimitiveInt "Duration"
    noise_node_id = "251:196"       # RandomNoise

    workflow = json.loads(json.dumps(workflow_flf2v_video))
    workflow[first_frame_node_id]["inputs"]["image"] = first_name
    workflow[last_frame_node_id]["inputs"]["image"] = last_name
    workflow[prompt_node_id]["inputs"]["value"] = prompt_text
    workflow[selector_node_id]["inputs"]["aspect_ratio"] = _map_aspect_ratio_to_comfy(aspect_ratio)
    workflow[selector_node_id]["inputs"]["megapixels"] = _resolve_mp(mp)
    workflow[duration_node_id]["inputs"]["value"] = duration

    used_seed = _pick_seed(seed)
    workflow[noise_node_id]["inputs"]["noise_seed"] = used_seed

    return _submit_comfy_workflow(workflow), used_seed
    
def submit_music_comfy(tags: str, lyrics: str, duration: int = 180, language: str = "ru"):
    """Музыка через ComfyUI (ACE-Step 1.5 XL Turbo). Возвращает (prompt_id, seed)."""
    if workflow_audio is None:
        raise RuntimeError("ComfyUI: audio_ace_step1_5_xl_turbo.json не загружен.")

    encode_node_id = "94"    # TextEncodeAceStepAudio1.5
    latent_node_id = "98"    # EmptyAceStep1.5LatentAudio
    seed_node_id = "109"     # PrimitiveInt (общий сид)

    workflow = json.loads(json.dumps(workflow_audio))
    workflow[encode_node_id]["inputs"]["tags"] = tags
    workflow[encode_node_id]["inputs"]["lyrics"] = lyrics
    workflow[encode_node_id]["inputs"]["duration"] = duration
    workflow[encode_node_id]["inputs"]["language"] = language
    workflow[latent_node_id]["inputs"]["seconds"] = duration

    # Синхронизируем поле bpm с тем, что LLM написал в тэгах ("128 BPM")
    bpm_match = re.search(r'(\d{2,3})\s*BPM', tags, re.IGNORECASE)
    if bpm_match:
        workflow[encode_node_id]["inputs"]["bpm"] = int(bpm_match.group(1))

    random_seed = random.randint(0, 2**53 - 1)
    workflow[seed_node_id]["inputs"]["value"] = random_seed

    return _submit_comfy_workflow(workflow), random_seed
    
def submit_image_comfy(prompt_text: str, width: int = 768, height: int = 1024, seed: int = None):
    """t2i: готовит workflow и ставит в очередь. Возвращает (prompt_id, seed)."""
    if workflow_t2i is None:
        raise RuntimeError("ComfyUI: workflow_t2i.json не загружен.")
    workflow = json.loads(json.dumps(workflow_t2i))
    workflow["57:27"]["inputs"]["text"] = prompt_text
    if not _set_workflow_resolution(workflow, width, height):
        raise RuntimeError("ComfyUI: не найден узел с width/height — впиши ID в LATENT_NODE_ID.")
    used_seed = _pick_seed(seed)
    workflow["57:3"]["inputs"]["seed"] = used_seed
    # Пользовательский сид фиксируем, иначе ComfyUI рандомизирует его после генерации
    workflow["57:3"]["inputs"]["control_after_generate"] = "fixed" if seed is not None else "randomize"
    return _submit_comfy_workflow(workflow), used_seed

def submit_edit_comfy(prompt_text: str, image_bytes: bytes, seed: int = None):
    """i2i: готовит workflow и ставит в очередь. Возвращает (prompt_id, seed)."""
    if workflow_edit is None:
        raise RuntimeError("ComfyUI: flux_image_edit.json не загружен.")
    image_filename = upload_image_to_comfy(image_bytes)
    if not image_filename:
        raise RuntimeError("ComfyUI: не удалось загрузить картинку в нейросеть.")
    workflow = json.loads(json.dumps(workflow_edit))
    workflow["75:74"]["inputs"]["text"] = prompt_text
    workflow["76"]["inputs"]["image"] = image_filename
    used_seed = _pick_seed(seed)
    workflow["75:73"]["inputs"]["noise_seed"] = used_seed
    return _submit_comfy_workflow(workflow), used_seed
    
def submit_multi_edit_comfy(prompt_text: str, image1_bytes: bytes, image2_bytes: bytes, seed: int = None):
    """Две картинки -> одна через multiple-input workflow. Возвращает (prompt_id, seed)."""
    if workflow_multi is None:
        raise RuntimeError("ComfyUI: flux_multiple_input.json не загружен.")

    img1_name = upload_image_to_comfy(image1_bytes)
    img2_name = upload_image_to_comfy(image2_bytes)
    if not img1_name or not img2_name:
        raise RuntimeError("ComfyUI: не удалось загрузить одну из картинок.")

    workflow = json.loads(json.dumps(workflow_multi))
    workflow = _prune_workflow_to_output(workflow, "94")  # оставляем только multiple-ветку

    workflow["92:109"]["inputs"]["text"] = prompt_text   # промт
    workflow["76"]["inputs"]["image"] = img1_name        # image1
    workflow["81"]["inputs"]["image"] = img2_name        # image2

    used_seed = _pick_seed(seed)
    workflow["92:106"]["inputs"]["noise_seed"] = used_seed

    return _submit_comfy_workflow(workflow), used_seed

def submit_video_t2v_comfy(prompt_text: str, aspect_ratio: str = DEFAULT_VIDEO_ASPECT, duration: int = 10, seed: int = None,
                             mp: float = None):
    """Текст -> видео через ComfyUI (LTX-2.5). duration=0 — модель сама подбирает длительность.
    Возвращает (prompt_id, seed)."""
    if workflow_t2v_video is None:
        raise RuntimeError("ComfyUI: video_ltx2_5_t2v.json не загружен.")

    prompt_node_id = "405:376"      # PrimitiveStringMultiline "Prompt"
    selector_node_id = "409"        # ResolutionSelector
    duration_node_id = "405:362"    # PrimitiveInt "Duration" (0 = авто)
    noise_node_ids = ["405:339", "405:338"]

    workflow = json.loads(json.dumps(workflow_t2v_video))
    workflow[prompt_node_id]["inputs"]["value"] = prompt_text
    workflow[selector_node_id]["inputs"]["aspect_ratio"] = _map_aspect_ratio_to_comfy(aspect_ratio)
    workflow[selector_node_id]["inputs"]["megapixels"] = _resolve_mp(mp)
    workflow[duration_node_id]["inputs"]["value"] = duration

    used_seed = _pick_seed(seed)
    for node_id in noise_node_ids:
        workflow[node_id]["inputs"]["noise_seed"] = used_seed

    return _submit_comfy_workflow(workflow), used_seed

def submit_video_i2v_comfy(prompt_text: str, image_bytes: bytes,
                           aspect_ratio: str = DEFAULT_VIDEO_ASPECT, duration: int = 10, seed: int = None,
                             mp: float = None):
    """Картинка -> видео через ComfyUI (LTX-2.5). duration=0 — модель сама подбирает.
    Возвращает (prompt_id, seed)."""
    if workflow_i2v_video is None:
        raise RuntimeError("ComfyUI: video_ltx2_5_i2v.json не загружен.")

    image_filename = upload_image_to_comfy(image_bytes)
    if not image_filename:
        raise RuntimeError("ComfyUI: не удалось загрузить картинку для анимации.")

    load_image_node_id = "395"        # LoadImage "Load First Frame"
    prompt_node_id = "398:376"        # PrimitiveStringMultiline "Prompt"
    selector_node_id = "403"          # ResolutionSelector
    duration_node_id = "398:362"      # PrimitiveInt "Duration"
    resize_node_id = "398:351"        # ResizeImageMaskNode
    noise_node_ids = ["398:339", "398:338"]

    workflow = json.loads(json.dumps(workflow_i2v_video))
    workflow[load_image_node_id]["inputs"]["image"] = image_filename
    workflow[prompt_node_id]["inputs"]["value"] = prompt_text
    workflow[selector_node_id]["inputs"]["aspect_ratio"] = _map_aspect_ratio_to_comfy(aspect_ratio)
    workflow[selector_node_id]["inputs"]["megapixels"] = _resolve_mp(mp)
    workflow[duration_node_id]["inputs"]["value"] = duration

    # Приводим картинку к выбранному соотношению (как в 2.3): точные размеры
    # из селектора + center crop — иначе при несовпадении пропорций картинки
    # и латента LTXVImgToVideoInplace может упасть на форме
    resize_inputs = workflow[resize_node_id]["inputs"]
    resize_inputs["resize_type"] = "scale dimensions"
    resize_inputs["resize_type.width"] = ["398:372", 0]   # Width из селектора
    resize_inputs["resize_type.height"] = ["398:360", 0]  # Height из селектора
    resize_inputs["resize_type.crop"] = "center"
    resize_inputs.pop("resize_type.longer_size", None)

    used_seed = _pick_seed(seed)
    for node_id in noise_node_ids:
        workflow[node_id]["inputs"]["noise_seed"] = used_seed

    return _submit_comfy_workflow(workflow), used_seed

def upload_image_to_comfy(image_bytes: bytes):
    """Загружает картинку в ComfyUI. Возвращает её имя во внутреннем хранилище."""
    unique_name = f"tg_{uuid.uuid4().hex[:12]}.jpg"
    files = {'image': (unique_name, image_bytes, 'image/jpeg')}
    try:
        response = requests.post(f"{COMFY_URL}/upload/image", files=files, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"ComfyUI: ошибка загрузки картинки: {e}")
        return None
    if response.status_code == 200:
        data = response.json()
        if data.get('subfolder'):
            return f"{data['subfolder']}/{data['name']}"
        return data['name']
    return None

# --- Буфер альбомов (Telegram шлёт альбом как N сообщений с общим media_group_id) ---
album_buffer: dict = {}  # (chat_id, media_group_id) -> {"messages": [], "task": ...}

@main_router.message(F.photo, lambda m: m.media_group_id is not None)
async def handle_album(message: aiogram_types.Message):
    """Ловит все фото альбома, копит и после паузы обрабатывает группу целиком."""
    key = (message.chat.id, message.media_group_id)
    entry = album_buffer.setdefault(key, {"messages": [], "task": None})

    if any(m.message_id == message.message_id for m in entry["messages"]):
        return  # защита от дублей
    entry["messages"].append(message)

    if entry["task"]:
        entry["task"].cancel()
    entry["task"] = asyncio.create_task(_flush_album(key))


async def _flush_album(key, delay: float = 1.5):
    """Ждёт delay сек после последнего фото группы, затем роутит альбом."""
    await asyncio.sleep(delay)
    entry = album_buffer.pop(key, None)
    if not entry:
        return
    messages = entry["messages"]
    first = messages[0]
    try:
        await _process_album(messages)
    except Exception as e:
        print(f"Ошибка обработки альбома: {e}")
        thread_id = first.message_thread_id if first.is_topic_message else None
        await send_log_to_telegram(str(e), "Album", first.chat.id, thread_id,
                                   first.from_user.full_name if first.from_user else "")

async def _process_flf2v(messages: list, caption: str):
    """Альбом из 2+ фото с «анимируй»: первый кадр -> последний кадр -> видео."""
    first = messages[0]
    chat_id = first.chat.id
    thread_id = first.message_thread_id if first.is_topic_message else None
    user = first.from_user
    user_name = user.full_name if user else "Пользователь"
    user_id = user.id if user else None
    status_msg = None

    raw_caption = caption.lower()
    cleaned_caption, user_seed = extract_seed(raw_caption)
    cleaned_caption, aspect_ratio = extract_video_aspect_ratio(raw_caption)
    cleaned_caption, duration = extract_video_duration(cleaned_caption)
    cleaned_caption, mp = extract_megapixels(cleaned_caption)
    request_text = cleaned_caption.replace(CHAT_TRIGGER_WORD, '').replace(VIDEO_TRIGGER_COMMAND, '').strip()
    prompt_text = f'{user_name}: {request_text}' if request_text else \
                  f'{user_name}: придумай плавный и эффектный переход от первого кадра ко второму'

    aspect_ratio = aspect_ratio or DEFAULT_VIDEO_ASPECT
    duration = duration if duration is not None else 10

    try:
        comfy_online = await asyncio.to_thread(is_comfyui_online)
        if not comfy_online:
            await first.reply(f"🖥❌ Генератор временно отключён, попробуй позже.")
            return

        # Скачиваем первые два фото альбома: 1-е = первый кадр, 2-е = последний
        files = []
        for m in messages[:2]:
            file = await bot.get_file(m.photo[-1].file_id)
            files.append((await bot.download_file(file.file_path)).read())
        encoded_images = [base64.b64encode(b).decode('utf-8') for b in files]

        # Vision-модель видит оба кадра; роли кадров указываем в сообщении
        await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
        flf_context = (f"{prompt_text}\n\n"
                       f"=== КАДРЫ ===\n"
                       f"image1 (первое присланное фото) — ПЕРВЫЙ кадр видео.\n"
                       f"image2 (второе присланное фото) — ПОСЛЕДНИЙ кадр видео.")
        detailed_prompt = await generate_flf2v_prompt(chat_id, thread_id, flf_context, encoded_images,
                                                      user_id=user_id, duration=duration)

        prompt_id, used_seed = await asyncio.to_thread(submit_video_flf2v_comfy, detailed_prompt,
                                                        files[0], files[1], aspect_ratio, duration, user_seed, mp)
        position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)

        duration_text = f"{duration} сек" if duration else "длительность выберет нейронка"
        status_msg = await first.answer(
            f"⌛ Анимирую переход между твоими кадрами ({aspect_ratio}, {duration_text}).\n"
            f"{_format_queue_info(position, running)}Это надолго, жди..."
        )
        await bot.send_chat_action(chat_id, "upload_video", message_thread_id=thread_id)

        print(f'Генерация видео FLF2V ({aspect_ratio}, {duration_text})...')
        video_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)

        await send_video_to_chat(chat_id, thread_id, video_bytes)
        seed_text = f"{used_seed}" if user_seed is not None else str(used_seed)
        await first.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n📐 {aspect_ratio} | ⏱ {duration_text} | 🌱 Seed: {seed_text}")
        print('Готово!')

    except Exception as e:
        print(f"Ошибка flf2v: {e}")
        await send_log_to_telegram(str(e), "Video_FLF2V", chat_id, thread_id, user_name)
    finally:
        if status_msg:
            await status_msg.delete()

async def _process_multi_edit(messages: list, caption: str):
    """Две картинки + «нарисуй»: local → flux_multiple_input, online → AirForce с двумя image_urls."""
    first = messages[0]
    chat_id = first.chat.id
    thread_id = first.message_thread_id if first.is_topic_message else None
    user = first.from_user
    user_name = user.full_name if user else "Пользователь"
    user_id = user.id if user else None
    status_msg = None

    raw_caption = caption.lower()
    cleaned_caption, user_seed = extract_seed(raw_caption)
    cleaned_caption, _ = extract_resolution(raw_caption)
    request_text = cleaned_caption.replace(CHAT_TRIGGER_WORD, '').replace(IMAGE_TRIGGER_COMMAND, '').strip()
    if not request_text:
        request_text = "совмести эти две картинки: перенеси стиль и элементы из image2 в image1"
    prompt_text = f'{user_name}: {request_text}'

    current_image_model = get_image_model(chat_id)

    try:
        # Скачиваем первые две картинки альбома
        files = []
        for m in messages[:2]:
            file = await bot.get_file(m.photo[-1].file_id)
            files.append((await bot.download_file(file.file_path)).read())
        encoded_images = [base64.b64encode(b).decode('utf-8') for b in files]

        if current_image_model == "local":
            comfy_online = await asyncio.to_thread(is_comfyui_online)
            if not comfy_online:
                await first.reply(f"🖥❌ Генератор временно отключён, попробуй позже.", parse_mode="Markdown")
                return

            # Промт пишет vision-модель, видя ОБЕ картинки
            await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
            detailed_prompt = await generate_multi_edit_prompt(chat_id, thread_id, prompt_text, encoded_images, user_id=user_id)

            prompt_id, used_seed = await asyncio.to_thread(submit_multi_edit_comfy, detailed_prompt, files[0], files[1], user_seed)
            position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)
            status_msg = await first.answer(
                f"⌛ Совмещаю две твои картинки.\n{_format_queue_info(position, running)}Может занять до 10 мин..."
            )
            await bot.send_chat_action(chat_id, "upload_photo", message_thread_id=thread_id)

            photo_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)
            await bot.send_photo(chat_id=chat_id, photo=BufferedInputFile(photo_bytes, filename="comfy_multi.png"), message_thread_id=thread_id)
            seed_text = f"{used_seed}" if user_seed is not None else str(used_seed)
            await first.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n🌱 Seed: {seed_text}")
        else:
            status_msg = await first.answer("⌛ Совмещаю две твои картинки (может занять до 10 мин)...")

            imgbb_urls = []
            for b in files:
                url = await asyncio.to_thread(upload_to_imgbb, b, None, chat_id, thread_id)
                if not url:
                    raise Exception("Не удалось загрузить одну из картинок на хостинг")
                imgbb_urls.append(url)

            detailed_prompt = await generate_multi_edit_prompt(chat_id, thread_id, prompt_text, encoded_images, user_id=user_id)
            await asyncio.sleep(65)

            path = await asyncio.to_thread(generate_media_sync, user_id, detailed_prompt, chat_id, thread_id,
                                           image_url=imgbb_urls)
            if path:
                path = upload_to_imgbb(None, path, chat_id, thread_id)
                await bot.send_photo(chat_id=chat_id, photo=path, message_thread_id=thread_id)
                await first.answer(f"📝 Промт:\n\n{detailed_prompt}")

    except Exception as e:
        print(f"Ошибка multiple edit: {e}")
        await send_log_to_telegram(str(e), "Multi_Edit", chat_id, thread_id, user_name)
    finally:
        if status_msg:
            await status_msg.delete()

async def _process_album(messages: list):
    """Роутинг альбома по содержимому подписи."""
    first = messages[0]
    caption = next((m.caption for m in messages if m.caption), None)
    caption_lower = (caption or "").lower()
    captioned_msg = next((m for m in messages if m.caption), first)

    has_image_cmd = IMAGE_TRIGGER_COMMAND in caption_lower
    has_video_cmd = VIDEO_TRIGGER_COMMAND in caption_lower
    triggered = CHAT_TRIGGER_WORD in caption_lower
    
    # НОВОЕ: команды генерации — только со СВОИМ триггер-словом.
    # Важен порядок: сначала это, потом приват-проверка для vision.
    if (has_image_cmd or has_video_cmd) and not triggered:
        return  # команда адресована другому боту — выходим, не генерим

    # 1. Не наша команда — сохраняем старое поведение (vision по подписи)
    if not (has_image_cmd or has_video_cmd):
        if triggered or first.chat.type == ChatType.PRIVATE:
            # [Медиа]-запись в память, как это делал monitor_all_messages
            memory = load_memory(first.chat.id)
            for m in messages:
                if m.caption:
                    memory = append_history(memory, opponent_message=format_message_text(m), chat_id=first.chat.id)
            save_memory(first.chat.id, memory)
            await handle_photo_vision(captioned_msg)
        return

    # 2. Анимация — i2v по фото с подписью (первому)
    if has_video_cmd:
        if len(messages) >= 2:
            await _process_flf2v(messages, caption)
        else:
            await handle_video_from_photo(captioned_msg)
        return

    # 3. «нарисуй» + 2 и более фото → multiple-input workflow
    if len(messages) >= 2:
        await _process_multi_edit(messages, caption)
        return

    # 4. «нарисуй» + одно фото в альбоме → обычное одиночное редактирование
    await handle_photo_edit_request(captioned_msg)

async def _is_chat_admin(message: aiogram_types.Message) -> bool:
    if message.chat.type == "private":
        return True
    try:
        member = await bot.get_chat_member(message.chat.id, message.from_user.id)
        return member.status in ["creator", "administrator"]
    except Exception:
        return False

@main_router.message(F.text.lower() == f"{CHAT_TRIGGER_WORD} prompt off")
async def handle_prompt_enhancer_off(message: aiogram_types.Message):
    save_prompt_enhancer_state(message.chat.id, False)
    await message.reply(
        f"🔌 Улучшатель промтов ВЫКЛЮЧЕН для этого чата.\n\n"
        f"Промт теперь уходит в генератор как есть, без нейронки. "
        f"Для лучшего качества пиши запрос сразу на английском.\n\n"
        f"Включить обратно: `{CHAT_TRIGGER_WORD} prompt on`",
        parse_mode="Markdown"
    )

@main_router.message(F.text.lower() == f"{CHAT_TRIGGER_WORD} prompt on")
async def handle_prompt_enhancer_on(message: aiogram_types.Message):
    save_prompt_enhancer_state(message.chat.id, True)
    await message.reply(
        f"🔌 Улучшатель промтов снова ВКЛЮЧЕН — нейронка пишет промты за тебя.",
    )

@main_router.message(F.text.lower() == f"{CHAT_TRIGGER_WORD} prompt")
async def handle_prompt_enhancer_status(message: aiogram_types.Message):
    state = get_prompt_enhancer_state(message.chat.id)
    state_text = "🟢 ВКЛЮЧЕН — нейронка улучшает промты" if state else "🔴 ВЫКЛЮЧЕН — промты идут как есть"
    await message.reply(
        f"Улучшатель промтов в этом чате: {state_text}\n\n"
        f"Управление: `{CHAT_TRIGGER_WORD} prompt on` / `{CHAT_TRIGGER_WORD} prompt off`",
        parse_mode="Markdown"
    )

@main_router.message(F.photo, lambda m: m.caption and f'{CHAT_TRIGGER_WORD}' in m.caption.lower() and IMAGE_TRIGGER_COMMAND in m.caption.lower())
async def handle_photo_edit_request(message: aiogram_types.Message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    user_name = message.from_user.full_name
    raw_caption = message.caption.lower()
    cleaned_caption, user_seed = extract_seed(raw_caption)
    cleaned_caption, aspect_ratio = extract_aspect_ratio(raw_caption)
    request_text = cleaned_caption.replace(CHAT_TRIGGER_WORD, '').replace(IMAGE_TRIGGER_COMMAND, '').strip()
    prompt_text = f'{user_name}: {request_text}'
    status_msg = None

    current_image_model = get_image_model(chat_id)

    if not request_text:
        await message.reply(f"⚠️ {CHAT_TRIGGER_WORD.capitalize()} не понял, что сделать с картинкой. Напиши, что изменить, после команды.")
        return

    try:
        # Скачиваем и кодируем картинку один раз — нужно обеим веткам
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_bytes = (await bot.download_file(file.file_path)).read()
        encoded_image = base64.b64encode(file_bytes).decode('utf-8')

        if current_image_model == "local":
            comfy_online = await asyncio.to_thread(is_comfyui_online)
            if not comfy_online:
                await message.reply("🖥❌ Генератор временно отключён, попробуй позже.")
                return

            # Промт пишет vision-модель — она видит картинку
            await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
            detailed_prompt = await generate_edit_prompt(chat_id, thread_id, prompt_text, encoded_image, user_id=user_id)

            prompt_id, used_seed = await asyncio.to_thread(submit_edit_comfy, detailed_prompt, file_bytes, user_seed)
            position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)

            status_msg = await message.answer(
                f"⌛ Пупс переделает твою картинку.\n{_format_queue_info(position, running)}Может занять до 10 мин..."
            )
            await bot.send_chat_action(chat_id, "upload_photo", message_thread_id=thread_id)

            photo_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)
            await bot.send_photo(chat_id=chat_id, photo=BufferedInputFile(photo_bytes, filename="comfy_edit.png"), message_thread_id=thread_id)
            await message.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n🌱 Seed: {used_seed}")
        else:
            status_msg = await message.answer("⌛ Идёт генерация картинки (может занять до 10 мин)...")

            imgbb_url = await asyncio.to_thread(upload_to_imgbb, file_bytes, None, chat_id, thread_id)
            if not imgbb_url:
                raise Exception("Не удалось загрузить изображение на хостинг")

            # Тот же edit-промт: vision-модель видит картинку и в онлайн-ветке
            detailed_prompt = await generate_edit_prompt(chat_id, thread_id, prompt_text, encoded_image, user_id=user_id)

            await asyncio.sleep(65)

            path = await asyncio.to_thread(generate_media_sync, user_id, detailed_prompt, chat_id, thread_id,
                                           image_url=[imgbb_url], aspect_ratio=aspect_ratio)
            if path:
                print('Отправка в тг...')
                path = upload_to_imgbb(None, path, chat_id, thread_id)
                await bot.send_photo(chat_id=chat_id, photo=path, message_thread_id=thread_id)
                await message.answer(f"📝 **Промт:**\n\n`{detailed_prompt}`")
        print('Готово!')

    except Exception as e:
        print(f"Ошибка генерации: {e}")
        await send_log_to_telegram(str(e), "Edit_Photo", chat_id, thread_id, user_name)
    finally:
        if status_msg:
            await status_msg.delete()
            
@main_router.message(F.text, lambda m: f'{CHAT_TRIGGER_WORD}' in m.text.lower() and VIDEO_TRIGGER_COMMAND in m.text.lower())
async def handle_video_generation(message: aiogram_types.Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    raw_text = message.text.lower()
    cleaned_text, user_seed = extract_seed(raw_text)
    cleaned_text, aspect_ratio = extract_video_aspect_ratio(cleaned_text)
    cleaned_text, duration = extract_video_duration(cleaned_text)
    cleaned_text, mp = extract_megapixels(cleaned_text)
    request_text = cleaned_text.replace(CHAT_TRIGGER_WORD, '').replace(VIDEO_TRIGGER_COMMAND, '').strip()
    prompt_text = f'{user_name}: {request_text}'
    chat_id = message.chat.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    status_msg = None

    aspect_ratio = aspect_ratio or DEFAULT_VIDEO_ASPECT
    duration = duration if duration is not None else 10                  # 0 = авто

    if not request_text:
        await message.reply(f"⚠️ {CHAT_TRIGGER_WORD.capitalize()} не понял, что анимировать. Напиши запрос после команды.")
        return

    try:
        comfy_online = await asyncio.to_thread(is_comfyui_online)
        if not comfy_online:
            await message.reply(f"🖥❌ Генератор временно отключён, попробуй позже.")
            return

        # 1. Пупс сочиняет промт (в чате горит «печатает...»)
        await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
        detailed_prompt = await generate_video_prompt(chat_id, thread_id, prompt_text, user_id=user_id, duration=duration)

        # 2. Постановка в очередь
        prompt_id, used_seed = await asyncio.to_thread(submit_video_t2v_comfy, detailed_prompt, aspect_ratio, duration, user_seed, mp)
        duration_text = f"{duration} сек" if duration else "длительность выберет нейронка"

        # 3. Первое и единственное статус-сообщение — сразу с очередью
        position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)
        status_msg = await message.answer(
            f"⌛ Идёт генерация видео {aspect_ratio}, {duration} сек.\n"
            f"{_format_queue_info(position, running)}"
            f"Это надолго, жди..."
        )
        await bot.send_chat_action(chat_id, "upload_video", message_thread_id=thread_id)

        # 4. Ожидание результата
        print(f'Генерация видео T2V ({aspect_ratio}, {duration} сек)...')
        video_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)

        await send_video_to_chat(chat_id, thread_id, video_bytes)
        seed_text = f"{used_seed}" if user_seed is not None else str(used_seed)
        await message.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n📐 {aspect_ratio} | ⏱ {duration} сек | 🌱 Seed: {seed_text}")
        print('Готово!')

    except Exception as e:
        print(e)
        await send_log_to_telegram(f"{e}", "Video_T2V", chat_id, thread_id, user_name)
    finally:
        if status_msg:
            await status_msg.delete()

@main_router.message(F.text, lambda m: f'{CHAT_TRIGGER_WORD}' in m.text.lower() and IMAGE_TRIGGER_COMMAND in m.text.lower())
async def handle_image_generation(message: aiogram_types.Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    raw_text = message.text.lower()
    cleaned_text, user_seed = extract_seed(raw_text)
    cleaned_text, resolution = extract_resolution(cleaned_text)          # <-- НОВОЕ: сначала вырезаем разрешение
    cleaned_text, aspect_ratio = extract_aspect_ratio(cleaned_text)  # потом соотношение (для онлайн-пути)
    request_text = cleaned_text.replace(CHAT_TRIGGER_WORD, '').replace(IMAGE_TRIGGER_COMMAND, '').strip()
    prompt_text = f'{user_name}: {request_text}'
    chat_id = message.chat.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    status_msg = None

    current_image_model = get_image_model(chat_id)

    if not request_text:
        await message.reply(f"⚠️ {CHAT_TRIGGER_WORD.capitalize()} не понял, что рисовать. Напиши запрос после команды.")
        return

    try:
        if current_image_model == "local":
            comfy_online = await asyncio.to_thread(is_comfyui_online)
            if not comfy_online:
                await message.reply(f"🖥❌ Генератор временно отключён, попробуй позже.")
                return

            width, height = resolution if resolution else (768, 1024)

            await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
            detailed_prompt = await generate_image_prompt(chat_id, thread_id, prompt_text, user_id=user_id)

            prompt_id, used_seed = await asyncio.to_thread(submit_image_comfy, detailed_prompt, width, height, user_seed)
            position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)

            status_msg = await message.answer(
                f"⌛ Идёт генерация картинки {width}x{height}.\n{_format_queue_info(position, running)}Может занять до 10 мин..."
            )
            await bot.send_chat_action(chat_id, "upload_photo", message_thread_id=thread_id)

            photo_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)
            await bot.send_photo(chat_id=chat_id, photo=BufferedInputFile(photo_bytes, filename="comfy_image.png"), message_thread_id=thread_id)
            seed_text = f"{used_seed}" if user_seed is not None else str(used_seed)
            await message.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n📐 {width}x{height} | 🌱 Seed: {seed_text}")
        else:
            status_msg = await message.answer("⌛ Идёт генерация картинки (может занять до 10 мин)...")
            detailed_prompt = await generate_image_prompt(chat_id, thread_id, prompt_text, user_id=user_id)
            await asyncio.sleep(65)
            print('Генерация...')

            path = await asyncio.to_thread(generate_media_sync, user_id, detailed_prompt, chat_id, thread_id, aspect_ratio=aspect_ratio)

            if path:
                print('Отправка в тг...')
                path = upload_to_imgbb(None, path, chat_id, thread_id)
                await bot.send_photo(chat_id=chat_id, photo=path, message_thread_id=thread_id)
                await message.answer(f"📝 Промт:\n\n{detailed_prompt}")

        print('Готово!')

    except Exception as e:
        print(e)
        user_info = message.from_user.full_name
        await send_log_to_telegram(f"{e}", "Image", chat_id, thread_id, user_info)
    finally:
        if status_msg:
            await status_msg.delete()

def split_by_lines(text: str, max_length: int = 4000) -> list[str]:
    """Разбивает текст на чанки строго по границам строк (\n)."""
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        # Учитываем длину строки + символ переноса (1 символ)
        # Если строка сама по себе длиннее max_length (маловероятно для названий моделей),
        # она уйдет в отдельный чанк.
        if current_length + len(line) + 1 > max_length:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = len(line)
        else:
            current_chunk.append(line)
            current_length += len(line) + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

@main_router.message(F.text.lower().contains(f"{CHAT_TRIGGER_WORD} chat"))
async def handle_set_chat_model(message: aiogram_types.Message):
    parts = message.text.split(maxsplit=2)
    models_text, allowed_ids = await get_models_info.get_chat_models_for_telegram()
    
    if not allowed_ids:
        await message.reply(
            f"❌ Не удалось получить список моделей.\n{models_text}"
        )
        return
    
    if len(parts) < 3:
        current_model = get_chat_model(message.chat.id)
        full_text = (
            f"🤖 *Текущая модель в этом чате:* {current_model}\n\n"
            f"Доступные варианты для смены (нажмите на имя, чтобы скопировать):\n"
            f"{models_text}\n\n"
            f"Чтобы изменить, напиши:\n{CHAT_TRIGGER_WORD} chat название-модели"
        )
        chunks = split_by_lines(full_text, max_length=4000)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Пропускаем пустые куски, если они возникнут
                continue
            if i == 0:
                await message.reply(chunk, parse_mode="Markdown")
            else:
                await message.answer(chunk, parse_mode="Markdown")
        return
        
    chosen_model = parts[2].strip().lower()
    if chosen_model not in allowed_ids:
        await message.reply("❌ Неверное название модели. Скопируй имя строго из списка доступных.")
        return
        
    save_chat_model(message.chat.id, chosen_model)
    await message.reply(f"✅ Успешно! Теперь в этом чате запросы отправляются в: {chosen_model}")
    
@main_router.message(F.text.lower() == f"{CHAT_TRIGGER_WORD} vision gemini")
async def handle_set_vision_gemini(message: aiogram_types.Message):
    save_vision_model(message.chat.id, "gemini")
    await message.reply("✅ Успешно! Теперь запросы с картинками в этом чате обрабатываются через Gemini.")
    
@main_router.message(F.text.lower() == f"{CHAT_TRIGGER_WORD} gemini")
async def handle_set_gemini_model(message: aiogram_types.Message):
    save_chat_model(message.chat.id, "gemini")
    await message.reply("✅ Успешно! Теперь в этом чате ответы генерируются через Gemini.")
    
@main_router.message(F.text.lower().contains(f"{CHAT_TRIGGER_WORD} music"))
async def handle_set_music_model(message: aiogram_types.Message):
    parts = message.text.split(maxsplit=2)
    models_text, allowed_ids = await get_models_info.get_music_models_for_telegram()
    
    if not allowed_ids:
        await message.reply(
            f"❌ Не удалось получить список моделей.\n{models_text}"
        )
        return
    
    if len(parts) < 3:
        current_model = get_music_model(message.chat.id)
        full_text = (
            f"🤖 Текущая модель для музыки в этом чате: {current_model}\n\n"
            f"Доступные варианты для смены (нажмите на имя, чтобы скопировать):\n"
            f"{models_text}\n\n"  # Выводим уже готовый текст из get_models_info.py
            f"Чтобы изменить, напиши:\n`{CHAT_TRIGGER_WORD} music название_модели`"
        )
        
        chunks = split_by_lines(full_text, max_length=4000)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Пропускаем пустые куски, если они возникнут
                continue
            if i == 0:
                await message.reply(chunk, parse_mode="Markdown")
            else:
                await message.answer(chunk, parse_mode="Markdown")
        return
        
    chosen_model = parts[2].strip().lower()
    if chosen_model not in allowed_ids:
        await message.reply("❌ Неверное название модели. Скопируй имя строго из списка доступных.")
        return
        
    save_music_model(message.chat.id, chosen_model)
    await message.reply(f"✅ Успешно! Теперь в этом чате запросы отправляются в: {chosen_model}")
    
@main_router.message(F.text.lower().contains(f"{CHAT_TRIGGER_WORD} vision"))
async def handle_set_vision_model(message: aiogram_types.Message):
    parts = message.text.split(maxsplit=2)
    models_text, allowed_ids = await get_models_info.get_vision_models_for_telegram()
    
    if not allowed_ids:
        await message.reply(
            f"❌ Не удалось получить список моделей.\n{models_text}"
        )
        return
    
    if len(parts) < 3:
        current_model = get_vision_model(message.chat.id)
        full_text = (
            f"🤖 Текущая модель vision в этом чате: {current_model}\n\n"
            f"Доступные варианты для смены (нажмите на имя, чтобы скопировать):\n"
            f"{models_text}\n\n"  # Выводим уже готовый текст из get_models_info.py
            f"Чтобы изменить, напиши:\n`{CHAT_TRIGGER_WORD} vision название_модели`"
        )
        
        chunks = split_by_lines(full_text, max_length=4000)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Пропускаем пустые куски, если они возникнут
                continue
            if i == 0:
                await message.reply(chunk, parse_mode="Markdown")
            else:
                await message.answer(chunk, parse_mode="Markdown")
        return
        
        return
        
    chosen_model = parts[2].strip().lower()
    if chosen_model not in allowed_ids:
        await message.reply("❌ Неверное название модели. Скопируй имя строго из списка доступных.")
        return
        
    save_vision_model(message.chat.id, chosen_model)
    await message.reply(f"✅ Успешно! Теперь в этом чате запросы с картинками отправляются в: {chosen_model}")
    
@main_router.message(F.text.lower().contains(f"{CHAT_TRIGGER_WORD} image local"))
async def handle_set_image_local(message: aiogram_types.Message):
    save_image_model(message.chat.id, "local")
    await message.reply(
        f"🖥 Локалка включена! Теперь картинки в этом чате генерятся через ComfyUI админа\n\n"
        f"Генерация: `{CHAT_TRIGGER_WORD} {IMAGE_TRIGGER_COMMAND} ...`\n"
        f"Редактирование: скинь фото с подписью `{CHAT_TRIGGER_WORD} {IMAGE_TRIGGER_COMMAND} ...`\n\n"
        f"Вернуться на онлайн: `{CHAT_TRIGGER_WORD} image airforce`",
        parse_mode="Markdown"
    )

@main_router.message(F.text.lower().contains(f"{CHAT_TRIGGER_WORD} image airforce"))
async def handle_set_image_airforce(message: aiogram_types.Message):
    save_image_model(message.chat.id, "flux-2-klein-9b")
    await message.reply(f"🌐 Вернул онлайн-генерацию через AirForce (flux-2-klein-9b).\n\nСписок моделей: `{CHAT_TRIGGER_WORD} image`", parse_mode="Markdown")
    
@main_router.message(F.text.lower().contains(f"{CHAT_TRIGGER_WORD} image"))
async def handle_set_image_model(message: aiogram_types.Message):
    parts = message.text.split(maxsplit=2)
    models_text, allowed_ids = await get_models_info.get_image_models_for_telegram()
    
    if not allowed_ids:
        await message.reply(
            f"❌ Не удалось получить список моделей.\n{models_text}"
        )
        return
    
    if len(parts) < 3:
        current_model = get_image_model(message.chat.id)
        full_text = (
            f"🤖 Текущая модель для генерации картинок в этом чате: {current_model}\n\n"
            f"Доступные варианты для смены (нажмите на имя, чтобы скопировать):\n"
            f"{models_text}\n\n"  # Выводим уже готовый текст из get_models_info.py
            f"Чтобы изменить, напиши:\n`{CHAT_TRIGGER_WORD} image название_модели`"
        )
        
        chunks = split_by_lines(full_text, max_length=4000)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Пропускаем пустые куски, если они возникнут
                continue
            if i == 0:
                await message.reply(chunk, parse_mode="Markdown")
            else:
                await message.answer(chunk, parse_mode="Markdown")
        return
        
        return
        
    chosen_model = parts[2].strip().lower()
    if chosen_model not in allowed_ids:
        await message.reply("❌ Неверное название модели. Скопируй имя строго из списка доступных.")
        return
        
    save_image_model(message.chat.id, chosen_model)
    await message.reply(f"✅ Успешно! Теперь в этом чате запросы с картинками отправляются в: {chosen_model}")

#@main_router.message(F.video, lambda m: (m.caption and f'{CHAT_TRIGGER_WORD}' in m.caption.lower()) or (m.text and f'{CHAT_TRIGGER_WORD}' in m.text.lower()))
@main_router.message(
    (F.video | F.video_note), 
    lambda m: (
        m.chat.type == ChatType.PRIVATE or 
        (m.caption and f'{CHAT_TRIGGER_WORD}'.lower() in m.caption.lower()) or 
        (m.text and f'{CHAT_TRIGGER_WORD}'.lower() in m.text.lower())
    )
)
async def handle_video_vision(message: aiogram_types.Message):
    chat_id = message.chat.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    user_name = message.from_user.full_name if message.from_user else "Пользователь"
    user_id = message.from_user.id
    
    # Проверяем модель
    if get_vision_model(chat_id) != "gemini":
        await message.reply(f"⚠️ Анализ видео поддерживается только моделью Gemini! Включи её командой: `{CHAT_TRIGGER_WORD} vision gemini`", parse_mode="Markdown")
        return

    user_text = message.caption or "Опиши, что происходит на этом видео"
    text = f"{user_name}: {user_text}"

    try:
        await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)

        # 1. Получаем объект видео
        video_obj = message.video or message.video_note
        
        # Проверка размера (Telegram Bot API не позволяет скачивать файлы > 20 МБ напрямую через бота)
        if video_obj.file_size and video_obj.file_size > 20 * 1024 * 1024:
            await message.reply("❌ Видео слишком большое! Telegram разрешает ботам скачивать файлы только до 20 МБ.")
            return

        # 2. Скачиваем файл
        print('Загрузка видео файла...')
        file = await bot.get_file(video_obj.file_id)
        file_bytes = await bot.download_file(file.file_path)
        
        # 💡 ВАЖНО: Сбрасываем каретку BytesIO в начало перед чтением
        file_bytes.seek(0)
        raw_bytes = file_bytes.read()
        
        if not raw_bytes:
            raise ValueError("Скачанный файл видео оказался пустым (0 байт).")

        # 3. Кодируем в base64
        encoded_video = base64.b64encode(raw_bytes).decode('utf-8')
        
        # Определяем mime_type (для video_note это обычно video/mp4)
        mime_type = getattr(message.video, 'mime_type', None) or "video/mp4"

        # 4. Отправляем в Gemini
        user_key = load_gemini_user_key(user_id)
        active_key = user_key if user_key else API_KEY_GEMINI
        response = await gemini_priem_video(chat_id, text, encoded_video, mime_type=mime_type, user_key=active_key)
        await message.reply(response)
        
        if is_summary_enabled(chat_id):
                bot_name = CHAT_TRIGGER_WORD.capitalize() # или имя бота
                log_data = get_chat_log(chat_id)
                log_data.append({
                    "user": bot_name,
                    "text": response,
                    "time": time.strftime("%H:%M")
                })
                save_chat_log(chat_id, log_data)

    except Exception as e:
        # Форматируем ошибку детально, чтобы в лог не уходила пустая строка
        error_details = f"{type(e).__name__}: {str(e)}" if str(e) else f"Неизвестное исключение {repr(e)}"
        print(f"❌ Ошибка в handle_video_vision: {error_details}")
        await send_log_to_telegram(error_details, "Video Handler", chat_id, thread_id, user_name)

@main_router.message(F.photo, lambda m: m.caption and f'{CHAT_TRIGGER_WORD}' in m.caption.lower() and VIDEO_TRIGGER_COMMAND in m.caption.lower())
async def handle_video_from_photo(message: aiogram_types.Message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    user_name = message.from_user.full_name
    raw_caption = message.caption.lower()
    cleaned_caption, user_seed = extract_seed(raw_caption)
    cleaned_caption, aspect_ratio = extract_video_aspect_ratio(raw_caption)
    cleaned_caption, duration = extract_video_duration(cleaned_caption)
    cleaned_caption, mp = extract_megapixels(cleaned_caption)
    request_text = cleaned_caption.replace(CHAT_TRIGGER_WORD, '').replace(VIDEO_TRIGGER_COMMAND, '').strip()
    prompt_text = f'{user_name}: {request_text}' if request_text else \
                  f'{user_name}: оживи это изображение — придумай естественное движение и звук, сохраняя суть сцены'
    status_msg = None

    aspect_ratio = aspect_ratio or DEFAULT_VIDEO_ASPECT   # дефолт 9:16, как прежний 360x640
    duration = duration if duration is not None else 10   # <-- дефолт 10 сек

    try:
        comfy_online = await asyncio.to_thread(is_comfyui_online)
        if not comfy_online:
            await message.reply(f"🖥❌ Генератор временно отключён, попробуй позже.")
            return

        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_bytes = (await bot.download_file(file.file_path)).read()

        await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
        encoded_image = base64.b64encode(file_bytes).decode('utf-8')
        detailed_prompt = await generate_video_i2v_prompt(chat_id, thread_id, prompt_text,
                                                           encoded_image, user_id=user_id, duration=duration)

        prompt_id, used_seed = await asyncio.to_thread(submit_video_i2v_comfy, detailed_prompt, file_bytes,
                                                        aspect_ratio, duration, user_seed, mp)

        position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)
        duration_text = f"{duration} сек" if duration else "длительность выберет нейронка"
        status_msg = await message.answer(
            f"⌛ Анимирую твою картинку ({aspect_ratio}, {duration_text}).\n"
            f"{_format_queue_info(position, running)}Это надолго, жди..."
        )
        await bot.send_chat_action(chat_id, "upload_video", message_thread_id=thread_id)

        print(f'Генерация видео I2V ({aspect_ratio}, {duration_text})...')
        video_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)

        await send_video_to_chat(chat_id, thread_id, video_bytes)
        seed_text = f"{used_seed}" if user_seed is not None else str(used_seed)
        await message.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n📐 {aspect_ratio} | ⏱ {duration_text} | 🌱 Seed: {seed_text}")
        print('Готово!')

    except Exception as e:
        print(f"Ошибка анимации: {e}")
        await send_log_to_telegram(str(e), "Video_I2V", chat_id, thread_id, user_name)
    finally:
        if status_msg:
            await status_msg.delete()

#@main_router.message(F.photo, lambda m: (m.caption and f'{CHAT_TRIGGER_WORD}' in m.caption.lower()))
@main_router.message(F.photo, lambda m: (
    m.chat.type == ChatType.PRIVATE or (m.caption and f'{CHAT_TRIGGER_WORD}'.lower() in m.caption.lower())
))
async def handle_photo_vision(message: aiogram_types.Message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    user_text = message.caption or ""
    user_name = message.from_user.full_name
    text = f"{user_name}: {user_text}"
    
    try:
        await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)

        # 1. Получаем файл фото
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_bytes = await bot.download_file(file.file_path)
        
        # 2. Кодируем в Base64
        encoded_image = base64.b64encode(file_bytes.read()).decode('utf-8')

        # 3. Запрос к ИИ
        response = await generate_vision_response(
            chat_id, 
            thread_id, 
            PROMPT, 
            text, 
            encoded_image,
            user_id=user_id
        )
        
        await message.reply(response)
        
        if is_summary_enabled(chat_id):
                bot_name = CHAT_TRIGGER_WORD.capitalize() # или имя бота
                log_data = get_chat_log(chat_id)
                log_data.append({
                    "user": bot_name,
                    "text": response,
                    "time": time.strftime("%H:%M")
                })
                save_chat_log(chat_id, log_data)
        
    except Exception as e:
        print(f"Ошибка в handle_photo_vision: {e}")
        user_info = message.from_user.full_name
        await send_log_to_telegram(f"{e}", "Vision Handler", chat_id, thread_id, user_info)

@main_router.message(F.text.lower().contains(f"{CHAT_TRIGGER_WORD} инфо"))
async def handle_info(message: aiogram_types.Message):
    await message.answer(pupps_info.pupps_info, parse_mode=ParseMode.MARKDOWN_V2)

### САММАРИ
@main_router.message(
    F.text.lower().contains("вкл пересказ") & 
    F.text.lower().contains(f'{CHAT_TRIGGER_WORD}') & 
    (F.chat.type.in_({"group", "supergroup"}))  # Сработает только в группах и супергруппах
)
async def handle_toggle_summary_on(message: aiogram_types.Message):
    chat_id = message.chat.id   
    thread_id = message.message_thread_id if message.is_topic_message else None
    is_admin = False
    
    try:
        member = await bot.get_chat_member(chat_id, message.from_user.id)
        is_admin = member.status in ["creator", "administrator"]
    except: 
        is_admin = False

    if is_admin:
        # Ищем время в формате HH:MM (например, 13:30 или 09:05)
        match = re.search(r'\b([0-1]?[0-9]|2[0-3]):([0-5][0-9])\b', message.text)
        if match:
            # Приводим к формату HH:MM (с ведущим нулем для часов, если нужно)
            hours, minutes = match.groups()
            target_time = f"{int(hours):02d}:{minutes}"
        else:
            target_time = "21:00"  # Дефолтное время

        set_summary_state(chat_id, True, thread_id=thread_id, target_time=target_time)
        
        reply_msg = f"✅ Принято! В этом топике я теперь записываю всё и выдам саммари в {target_time} по Москве."
        await message.reply(reply_msg)
    else:
        await message.reply("🚫 Слышь, ты не админ!")

@main_router.message(F.text.lower().contains("выкл пересказ") & F.text.lower().contains(f'{CHAT_TRIGGER_WORD}'))
async def handle_toggle_summary_off(message: aiogram_types.Message):
    chat_id = message.chat.id
    is_admin = message.chat.type == "private"
    if not is_admin:
        try:
            member = await bot.get_chat_member(chat_id, message.from_user.id)
            is_admin = member.status in ["creator", "administrator"]
        except: 
            is_admin = False

    if is_admin:
        set_summary_state(chat_id, False)
        clear_chat_log(chat_id)
        await message.reply("🔇 Всё, завалил. Больше не записываю, логи стёр.")
    else:
        await message.reply("🚫 Слышь, ты не админ!")

# --- 2. СБРОС ПАМЯТИ ---
@main_router.message(F.text.lower() == f"{CHAT_TRIGGER_WORD} 0")
async def handle_clear_memory(message: aiogram_types.Message):
    chat_id = message.chat.id
    is_admin = message.chat.type == "private"
    if not is_admin:
        try:
            member = await bot.get_chat_member(chat_id, message.from_user.id)
            is_admin = member.status in ["creator", "administrator"]
        except: is_admin = False

    if is_admin:
        if clear_memory(chat_id):
            await message.reply("🧼 Память стёрта. Я всё забыл.")
        else:
            await message.reply("❌ Ошибка при очистке.")
    else:
        await message.reply("🚫 Слышь, ты не админ!")

class ExactWordsFilter(Filter):
    def __init__(self, *words: str):
        self.words = [w.lower() for w in words]

    async def __call__(self, message: Message) -> bool:
        if not message.text:
            return False
        
        # Очищаем текст от пунктуации и разбиваем на слова
        text_words = re.findall(r'[а-яёa-z0-9]+', message.text.lower())
        
        # Проверяем, что ВСЕ триггеры есть в тексте как отдельные слова
        return all(word in text_words for word in self.words)

'''
@main_router.message(ExactWordsFilter(f'{CHAT_TRIGGER_WORD}', MUSIC_TRIGGER_COMMAND))
async def handle_music_generation(message: aiogram_types.Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    chat_id = message.chat.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    user_text = message.text.lower() or message.caption.lower() or ""
    status_msg = None
    prompt_text = {}
    
    try:
        status_msg = await message.answer("⌛ Идёт генерация музыки (может занять до 10 мин)...")
        
        user_text = user_text.replace(f'{CHAT_TRIGGER_WORD}', '').strip()
        user_text = user_text.replace('спой', '').strip()
        if "style:" in user_text:
            style_part = user_text.split("style:")[1].split("lyrics:")[0]
            prompt_text["style"] = style_part.strip()

        if "lyrics:" in user_text:
            lyrics_part = user_text.split("lyrics:")[1]
            prompt_text["lyrics"] = lyrics_part.strip()
            
        if not prompt_text:
            prompt_text = user_text
        print(prompt_text)

        print('Генерация...')
        if user_text:
            path = await asyncio.to_thread(generate_media_sync, user_id, prompt_text, chat_id, thread_id, is_music=True)
            print('Отправка в тг...')
            await message.reply(f"Ссылка на файл:\n{path}")
            print('Готово!')

    except Exception as e:
        print(e)
        user_info = message.from_user.full_name
        await send_log_to_telegram(f"{e}", "Music", chat_id, thread_id, user_info)
    finally:
        # Удаляем сообщение об ожидания в любом случае
        if status_msg:
            await status_msg.delete()'''
            
@main_router.message(ExactWordsFilter(f'{CHAT_TRIGGER_WORD}', MUSIC_TRIGGER_COMMAND))
async def handle_music_generation(message: aiogram_types.Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    chat_id = message.chat.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    status_msg = None

    raw_text = message.text.lower()
    cleaned_text, duration = extract_music_duration(raw_text)
    await asyncio.to_thread(get_ace_languages)  # прогрев кэша до синхронного парсинга
    cleaned_text, language = extract_music_language(cleaned_text)
    request_text = cleaned_text.replace(CHAT_TRIGGER_WORD, '').replace(MUSIC_TRIGGER_COMMAND, '').strip()
    prompt_text = f'{user_name}: {request_text}'

    duration = duration or DEFAULT_MUSIC_DURATION
    language = language or DEFAULT_MUSIC_LANGUAGE
    available_langs = get_ace_languages()
    if available_langs and language not in available_langs:
        language = available_langs[0]  # если дефолта нет в списке узла

    if not request_text:
        await message.reply(f"⚠️ {CHAT_TRIGGER_WORD.capitalize()} не понял, что спеть. Напиши запрос после команды.")
        return

    try:
        comfy_online = await asyncio.to_thread(is_comfyui_online)
        if not comfy_online:
            await message.reply(f"🖥❌ Генератор временно отключён, попробуй позже.")
            return

        await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
        music_prompt_raw = await generate_music_prompt(chat_id, thread_id, prompt_text,
                                                       user_id=user_id, language=language,
                                                       duration=duration)
        tags, lyrics = _parse_music_prompt(music_prompt_raw)

        prompt_id, used_seed = await asyncio.to_thread(submit_music_comfy, tags, lyrics, duration, language)
        position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)

        status_msg = await message.answer(
            f"⌛ Идёт генерация музыки ({duration} сек, язык: {language}).\n"
            f"{_format_queue_info(position, running)}Это надолго, жди..."
        )
        await bot.send_chat_action(chat_id, "upload_document", message_thread_id=thread_id)

        print(f'Генерация музыки ({duration} сек, {language})...')
        audio_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)

        await send_audio_to_chat(chat_id, thread_id, audio_bytes, duration=duration,
                                 title=tags.split(",")[0].strip() if tags else None,
                                 performer=CHAT_TRIGGER_WORD.capitalize())

        info_text = (f"🏷 Тэги:\n{tags}\n\n🎤 Лирика:\n{lyrics}\n\n"
                     f"⏱ {duration} сек | 🌐 {language} | 🌱 Seed: {used_seed}")
        for chunk in split_by_lines(info_text, max_length=4000):
            await message.answer(chunk)

        print('Готово!')

    except Exception as e:
        print(e)
        user_info = message.from_user.full_name
        await send_log_to_telegram(f"{e}", "Music", chat_id, thread_id, user_info)
    finally:
        if status_msg:
            await status_msg.delete()
            
@main_router.message(F.text.lower().contains(f"{CHAT_TRIGGER_WORD} context"))
async def handle_set_max_history(message: aiogram_types.Message):
    chat_id = message.chat.id
    
    # 1. Проверка прав администратора
    is_admin = message.chat.type == "private"
    if not is_admin:
        try:
            member = await bot.get_chat_member(chat_id, message.from_user.id)
            is_admin = member.status in ["creator", "administrator"]
        except Exception:
            is_admin = False

    if not is_admin:
        await message.reply("🚫 Слышь, ты не админ!")
        return

    # 2. Проверка и сохранение аргументов
    parts = message.text.split()
    
    if len(parts) < 3 or not parts[2].isdigit():
        await message.reply(f"⚠️ Укажи число! Пример: `{CHAT_TRIGGER_WORD} context 500`", parse_mode="Markdown")
        return

    new_limit = int(parts[2])

    if new_limit < 1 or new_limit > 10000:
        await message.reply("❌ Число должно быть в диапазоне от 1 до 10000.")
        return

    save_chat_max_history(chat_id, new_limit)
    await message.reply(f"✅ Успешно! Теперь максимальный размер истории в этом чате: {new_limit} сообщений.")
            
@main_router.message(F.voice)
async def handle_voice_message(message: aiogram_types.Message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    user_name = message.from_user.full_name
    is_private = message.chat.type == "private"

    try:
        # 1. Скачиваем аудиофайл
        voice = message.voice
        file_info = await bot.get_file(voice.file_id)
        file_bytes_io = await bot.download_file(file_info.file_path)
        file_bytes_io.seek(0)
        ogg_bytes = file_bytes_io.read()

        if not ogg_bytes:
            return

        # 2. Конвертируем в WAV в памяти для SpeechRecognition
        audio_seg = AudioSegment.from_file(io.BytesIO(ogg_bytes), format="ogg")
        wav_io = io.BytesIO()
        audio_seg.export(wav_io, format="wav")
        wav_io.seek(0)

        # 3. Распознаем речь пользователя
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)
                recognized_text = recognizer.recognize_google(audio_data, language="ru-RU")
                print(f"🎙️ [Голос от {user_name}]: {recognized_text}")
        except sr.UnknownValueError:
            recognized_text = "[Неразборчивое голосовое]"
        except Exception as e:
            recognized_text = "[Голосовое сообщение]"
            print(f"Ошибка распознавания речи: {e}")

        # 4. Проверка условий вызова (в личке всегда, в группах — по триггеру)
        is_triggered = CHAT_TRIGGER_WORD in recognized_text.lower()

        if is_private or is_triggered:
            await bot.send_chat_action(chat_id, "record_voice", message_thread_id=thread_id)

            # 1. Загружаем память и сохраняем текущий голосовой запрос пользователя
            memory = load_memory(chat_id)
            formatted_user_msg = f"[Голосовое] {user_name}: {recognized_text}"
            memory = append_history(memory, opponent_message=formatted_user_msg, chat_id=chat_id)
            save_memory(chat_id, memory)

            # 2. Выбираем API ключ
            user_key = load_gemini_user_key(user_id)
            active_key = user_key if user_key else API_KEY_GEMINI

            # 3. Передаем историю диалога в генератор
            response_voice_io, response_text = await process_voice_turn(
                ogg_bytes=ogg_bytes,
                system_prompt=PROMPT,
                api_key=active_key,
                voice_name="Fenrir",
                history=memory.get("history", [])
            )

            # 4. Сохраняем текстовый ответ бота в память
            memory = append_history(memory, my_response=response_text, chat_id=chat_id)
            save_memory(chat_id, memory)

            # 5. Отправка голосового сообщения с текстом в описании
            voice_file = BufferedInputFile(response_voice_io.read(), filename="voice_answer.ogg")
            if len(response_text) <= 1024:
                await message.reply_voice(voice=voice_file, caption=response_text)
            else:
                await message.reply_voice(voice=voice_file)
                await message.reply(response_text)

            # Сохраняем в лог для пересказа (саммари)
            if is_summary_enabled(chat_id):
                bot_name = CHAT_TRIGGER_WORD.capitalize()
                log_data = get_chat_log(chat_id)
                log_data.append({
                    "user": user_name,
                    "text": f"[Голосовое] {recognized_text}",
                    "time": time.strftime("%H:%M")
                })
                log_data.append({
                    "user": bot_name,
                    "text": f"[Голосовой ответ] {response_text}",
                    "time": time.strftime("%H:%M")
                })
                save_chat_log(chat_id, log_data)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {repr(e)}"
        print(f"❌ Ошибка в handle_voice_message: {error_msg}")
        await send_log_to_telegram(error_msg, "Voice Handler", chat_id, thread_id, user_name)
            
# Хендлер для записи всех сообщений в контекст
@main_router.message(lambda m: not (m.text or "").startswith('/'))
async def monitor_all_messages(message: aiogram_types.Message):
    # Игнорируем сообщения от самого бота, чтобы не дублировать
    if message.from_user and message.from_user.id == bot.id:
        return

    chat_id = message.chat.id
    user_id = message.from_user.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    user_text = message.text or message.caption or ""
    user_name = message.from_user.full_name
    
    # Форматируем текст с учетом префикса [Медиа]
    formatted_text = format_message_text(message)
    
    if IMAGE_TRIGGER_COMMAND in (message.text or message.caption or "").lower():
        return
        
    if VIDEO_TRIGGER_COMMAND in (message.text or message.caption or "").lower():
        return
        
    is_private = message.chat.type == "private"
    is_triggered = CHAT_TRIGGER_WORD in user_text.lower()
    
    if is_private or is_triggered:
        text_to_ai = f"{user_name}: {user_text}"
        
        try:
            await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
            response = await generate_response(chat_id, thread_id, PROMPT, text_to_ai, user_id=user_id)
            await bot.send_message(chat_id, response, message_thread_id=thread_id, reply_to_message_id=message.message_id)
            
            if is_summary_enabled(chat_id):
                bot_name = CHAT_TRIGGER_WORD.capitalize() # или имя бота
                log_data = get_chat_log(chat_id)
                log_data.append({
                    "user": bot_name,
                    "text": response,
                    "time": time.strftime("%H:%M")
                })
                save_chat_log(chat_id, log_data)
            
        except Exception as e:
            print(f"Ошибка в ответах: {e}")
            await send_log_to_telegram(f"{e}", "Chat", chat_id, thread_id, user_name)
    
    # Если текста совсем нет и это не медиа (например, просто стикер), 
    # можно либо игнорировать, либо записывать тип события
    if not formatted_text:
        return

async def cmd_start(message: aiogram_types.Message):
    await message.answer(pupps_info.pupps_info, parse_mode=ParseMode.MARKDOWN_V2)
    
@main_router.message(F.chat.type == "private", lambda m: m.text and m.text.startswith('/setkey airforce'))
async def handle_set_airforce_key(message: aiogram_types.Message):
    user_id = message.from_user.id
    parts = message.text.split(maxsplit=2)
    
    if len(parts) < 3:
        await message.reply("🔑 Чтобы установить ключ, напиши команду и твой ключ через пробел:\n`/setkey airforce твой_api_ключ`\n\nЧтобы удалить свой ключ, напиши `/setkey airforce delete`", parse_mode="Markdown")
        return
        
    user_key = parts[2].strip()
    
    if user_key.lower() == 'delete':
        # Логика удаления (просто затрем файл пустым ключом или удалим его)
        if save_airforce_user_key(user_id, None):
            await message.reply("🗑️ Твой персональный API-ключ удален. Теперь запросы снова будут идти через ключ администратора.")
        else:
            await message.reply("❌ Не удалось удалить ключ.")
        return

    # Сохраняем ключ
    if save_airforce_user_key(user_id, user_key):
        await message.reply(f"✅ Твой персональный API-ключ успешно сохранен! Теперь {CHAT_TRIGGER_WORD.capitalize()} будет отвечать тебе в любых чатах, используя твою квоту.")
    else:
        await message.reply("❌ Произошла ошибка при сохранении ключа.")
        
@main_router.message(F.chat.type == "private", lambda m: m.text and m.text.startswith('/setkey gemini'))
async def handle_set_gemini_key(message: aiogram_types.Message):
    user_id = message.from_user.id
    parts = message.text.split(maxsplit=2)
    
    if len(parts) < 3:
        await message.reply("🔑 Чтобы установить ключ, напиши команду и твой ключ через пробел:\n`/setkey gemini твой_api_ключ`\n\nЧтобы удалить свой ключ, напиши `/setkey gemini delete`", parse_mode="Markdown")
        return
        
    user_key = parts[2].strip()
    
    if user_key.lower() == 'delete':
        # Логика удаления (просто затрем файл пустым ключом или удалим его)
        if save_gemini_user_key(user_id, None):
            await message.reply("🗑️ Твой персональный API-ключ удален. Теперь запросы снова будут идти через ключ администратора.")
        else:
            await message.reply("❌ Не удалось удалить ключ.")
        return

    # Сохраняем ключ
    if save_gemini_user_key(user_id, user_key):
        await message.reply(f"✅ Твой персональный API-ключ успешно сохранен! Теперь {CHAT_TRIGGER_WORD.capitalize()} будет отвечать тебе в любых чатах, используя твою квоту.")
    else:
        await message.reply("❌ Произошла ошибка при сохранении ключа.")

def is_trigger_message(message: aiogram_types.Message) -> bool:
    text = (message.text or message.caption or "").lower()
    return CHAT_TRIGGER_WORD in text

class HistoryMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        # Работаем строго с сообщениями
        if not isinstance(event, aiogram_types.Message):
            return await handler(event, data)

        chat_id = event.chat.id
        topic_id = event.message_thread_id or 0
        
        # Полностью игнорируем сообщения от самого себя
        if event.from_user and event.from_user.id == bot.id:
            return
            
        if event.from_user and event.from_user.id in IGNORED_BOT_IDS:
            return

        # Форматируем текст сообщения (поддерживает медиа)
        prefix = "[Медиа] " if (event.photo or event.video or event.animation or event.document) else ""
        user_name = event.from_user.full_name if event.from_user else "Система"
        content = event.text or event.caption or ""
        formatted_text = f"{prefix}{user_name}: {content}".strip()
        
        if content:
            if is_summary_enabled(chat_id):
                log_data = get_chat_log(chat_id)
                log_data.append({
                    "user": user_name,
                    "text": content,
                    "time": time.strftime("%H:%M")
                })
                save_chat_log(chat_id, log_data)
        
        # --- БЛОК ОБРАБОТКИ БОТОВ ---
        if event.from_user and event.from_user.is_bot:
            if formatted_text and content:
                flag = False
                for i in commands:
                    if i in content:
                        flag = True
                        break
                if not flag:
                    memory = load_memory(chat_id)
                    updated_memory = append_history(memory, opponent_message=formatted_text, chat_id=chat_id)
                    save_memory(chat_id, updated_memory)
                        
                        # КРИТИЧЕСКИ ВАЖНО: Всегда делаем return для чужих ботов.
                        # Мы не пускаем их к хэндлерам (handler), чтобы они не отвечали друг другу мгновенно.
                    return

        # --- БЛОК ОБРАБОТКИ ОБЫЧНЫХ ПОЛЬЗОВАТЕЛЕЙ (ЛЮДЕЙ) ---
        if formatted_text and content:
            flag = False
            for i in commands:
                if i in content:
                    flag = True
                    break
            if not flag:
                memory = load_memory(chat_id)
                updated_memory = append_history(memory, opponent_message=formatted_text, chat_id=chat_id)
                save_memory(chat_id, updated_memory)

        # Пропускаем сообщения людей дальше к хэндлерам (включая команды /start, няша stop и т.д.)
        return await handler(event, data)

async def main():
    print("Бот запущен...")
    load_chat_settings()
    load_image_settings()
    load_vision_settings()
    load_music_settings()
    load_prompt_settings()
    
    dp.message.outer_middleware(HistoryMiddleware())
    main_router.message.register(cmd_start, CommandStart())
    
    scheduler.add_job(daily_summary_executor, 'interval', minutes=1)
    scheduler.start()
    
    from bot_dialogue import restore_dialogues
    await restore_dialogues(bot)
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)