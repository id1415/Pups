import asyncio
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
                   save_chat_max_history)
from summary import is_summary_enabled, set_summary_state, process_pupps_summary, daily_summary_executor
from variables import (CHAT_TRIGGER_WORD, IMAGE_TRIGGER_COMMAND, MUSIC_TRIGGER_COMMAND, PROMPT, MAX_RETRIES, RETRY_DELAY,
                       AIRFORCE_API_URL, AIRFORCE_API_KEY, IMGBB_API_KEY, PUPS_BOT_TOKEN, bot, API_KEY_GEMINI,
                       client,
                       COMFY_URL, IMAGE_PROMPT,
                       VIDEO_TRIGGER_COMMAND, VIDEO_PROMPT)
import pupps_info
import get_models_info
from gemini import priem as gemini_priem, priem_vision as gemini_priem_vision, priem_video as gemini_priem_video, process_voice_turn

commands = ['нейро инфо', 'нейро name', 'нейро prompt', 'нейро chat', 'нейро vision', 'нейро image', 'нейро music', 'нейро start', 'нейро stop', 'нейро 0',
            'кибер инфо', 'кибер name', 'кибер prompt', 'кибер chat', 'кибер vision', 'кибер image', 'кибер music', 'кибер start', 'кибер stop', 'кибер 0',
            'пупс инфо', 'пупс chat', 'пупс vision', 'пупс image', 'пупс music', 'пупс start', 'пупс stop', 'пупс 0',
            'няша инфо', 'няша chat', 'няша vision', 'няша image', 'няша music', 'няша start', 'няша stop', 'няша 0',
            'пупс context', 'няша context', 'нейро context', 'кибер context',
            'пупс image local', 'пупс image airforce', 'няша image local', 'няша image airforce',
            ]

scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
dp = Dispatcher()
main_router = Router()
dp.include_router(dialogue_router)
dp.include_router(main_router)

VIDEO_ALLOWED_RESOLUTIONS = [("360", "640"), ("640", "360"), ("640", "640")]

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

def extract_video_resolution(text: str):
    """
    Ищет одно из разрешений для видео: 360x640 (дефолт), 640x360, 640x640.
    Разделитель: x, х, × или :. Возвращает (очищенный_текст, (w, h)) или (текст, None).
    """
    for w, h in VIDEO_ALLOWED_RESOLUTIONS:
        match = re.search(rf'\b{w}\s*[:xх×]\s*{h}\b', text)
        if match:
            cleaned_text = text[:match.start()] + text[match.end():]
            cleaned_text = " ".join(cleaned_text.split())
            return cleaned_text, (int(w), int(h))
    return text, None

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
    
    current_content = [
        {"type": "text", "text": current_user_message},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
    ]
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

async def send_video_to_chat(chat_id: int, thread_id, video_bytes: bytes, width: int, height: int):
    """Шлёт видео; если Телега не приняла формат/размер — отправляет как документ."""
    try:
        await bot.send_video(
            chat_id=chat_id,
            video=BufferedInputFile(video_bytes, filename="comfy_video.mp4"),
            width=width, height=height,
            supports_streaming=True,
            message_thread_id=thread_id
        )
    except Exception as e:
        print(f"send_video не сработал ({e}), отправляю как документ...")
        await bot.send_document(
            chat_id=chat_id,
            document=BufferedInputFile(video_bytes, filename="comfy_video.mp4"),
            message_thread_id=thread_id
        )

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

    # 2. Генерация через vision-модель (управляется командой "пупс vision ...")
    return await generate_vision_response(chat_id, thread_id, _compose_video_system_prompt(duration),
                                           current_user_message, encoded_image,
                                           user_id=user_id)

def _compose_video_system_prompt(duration: int = None) -> str:
    """
    Собирает системный промт для видео: базовый PUPS_VIDEO + точная длительность,
    чтобы модель строила промт под нужный хронометраж.
    """
    system_prompt = VIDEO_PROMPT
    if duration:
        system_prompt += (
            f"\n\nПАРАМЕТР ГЕНЕРАЦИИ: длительность ролика — {duration} секунд.\n"
            f"Учитывай её строго по разделу «ДЛИТЕЛЬНОСТЬ»: количество событий и темп "
            f"должны соответствовать этому времени."
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

workflow_t2v_video = _load_comfy_workflow('video_ltx2_3_t2v.json')  # текст -> видео
workflow_i2v_video = _load_comfy_workflow('video_ltx2_3_i2v.json')  # картинка -> видео
workflow_t2i = _load_comfy_workflow('workflow_t2i.json')      # текст -> картинка

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

COMFY_MEDIA_KEYS = ('images', 'gifs', 'videos')

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
    
def submit_image_comfy(prompt_text: str, width: int = 768, height: int = 1024):
    """t2i: готовит workflow и ставит в очередь. Возвращает (prompt_id, seed)."""
    if workflow_t2i is None:
        raise RuntimeError("ComfyUI: workflow_t2i.json не загружен.")
    workflow = json.loads(json.dumps(workflow_t2i))
    workflow["57:27"]["inputs"]["text"] = prompt_text
    if not _set_workflow_resolution(workflow, width, height):
        raise RuntimeError("ComfyUI: не найден узел с width/height — впиши ID в LATENT_NODE_ID.")
    random_seed = random.randint(0, 2**53 - 1)
    workflow["57:3"]["inputs"]["seed"] = random_seed
    workflow["57:3"]["inputs"]["control_after_generate"] = "randomize"
    return _submit_comfy_workflow(workflow), random_seed

def submit_edit_comfy(prompt_text: str, image_bytes: bytes):
    """i2i: готовит workflow и ставит в очередь. Возвращает (prompt_id, seed)."""
    if workflow_edit is None:
        raise RuntimeError("ComfyUI: flux_image_edit.json не загружен.")
    image_filename = upload_image_to_comfy(image_bytes)
    if not image_filename:
        raise RuntimeError("ComfyUI: не удалось загрузить картинку в нейросеть.")
    workflow = json.loads(json.dumps(workflow_edit))
    workflow["75:74"]["inputs"]["text"] = prompt_text
    workflow["76"]["inputs"]["image"] = image_filename
    random_seed = random.randint(0, 2**53 - 1)
    workflow["75:73"]["inputs"]["noise_seed"] = random_seed
    return _submit_comfy_workflow(workflow), random_seed

def submit_video_t2v_comfy(prompt_text: str, width: int = 360, height: int = 640, duration: int = 10):
    """t2v: готовит workflow и ставит в очередь. Возвращает (prompt_id, seed)."""
    if workflow_t2v_video is None:
        raise RuntimeError("ComfyUI: video_ltx2_3_t2v.json не загружен.")
    workflow = json.loads(json.dumps(workflow_t2v_video))
    workflow["267:266"]["inputs"]["value"] = prompt_text
    workflow["267:257"]["inputs"]["value"] = width
    workflow["267:258"]["inputs"]["value"] = height
    workflow["267:225"]["inputs"]["value"] = duration
    random_seed = random.randint(0, 2**53 - 1)
    for node_id in ("267:216", "267:237"):
        workflow[node_id]["inputs"]["noise_seed"] = random_seed
    return _submit_comfy_workflow(workflow), random_seed

def submit_video_i2v_comfy(prompt_text: str, image_bytes: bytes, width: int = 360, height: int = 640, duration: int = 10):
    """i2v: готовит workflow и ставит в очередь. Возвращает (prompt_id, seed)."""
    if workflow_i2v_video is None:
        raise RuntimeError("ComfyUI: video_ltx2_3_i2v.json не загружен.")
    image_filename = upload_image_to_comfy(image_bytes)
    if not image_filename:
        raise RuntimeError("ComfyUI: не удалось загрузить картинку для анимации.")
    workflow = json.loads(json.dumps(workflow_i2v_video))
    workflow["269"]["inputs"]["image"] = image_filename
    workflow["320:319"]["inputs"]["value"] = prompt_text
    workflow["320:312"]["inputs"]["value"] = width
    workflow["320:299"]["inputs"]["value"] = height
    workflow["320:301"]["inputs"]["value"] = duration
    random_seed = random.randint(0, 2**53 - 1)
    for node_id in ("320:276", "320:277"):
        workflow[node_id]["inputs"]["noise_seed"] = random_seed
    return _submit_comfy_workflow(workflow), random_seed

def upload_image_to_comfy(image_bytes: bytes):
    """Загружает картинку в ComfyUI. Возвращает её имя во внутреннем хранилище."""
    files = {'image': ('temp_image.jpg', image_bytes, 'image/jpeg')}
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

@main_router.message(F.photo, lambda m: m.caption and f'{CHAT_TRIGGER_WORD}' in m.caption.lower() and IMAGE_TRIGGER_COMMAND in m.caption.lower())
async def handle_photo_edit_request(message: aiogram_types.Message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    user_name = message.from_user.full_name
    raw_caption = message.caption.lower()
    cleaned_caption, aspect_ratio = extract_aspect_ratio(raw_caption)
    prompt_text = f'{user_name}: {cleaned_caption}'
    status_msg = None

    current_image_model = get_image_model(chat_id)

    try:
        if current_image_model == "local":
            comfy_online = await asyncio.to_thread(is_comfyui_online)
            if not comfy_online:
                await message.reply(f"🖥❌ Генератор временно отключён, попробуй позже.")
                return

            width, height = resolution if resolution else (768, 1024)

            await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
            detailed_prompt = await generate_image_prompt(chat_id, thread_id, prompt_text, user_id=user_id)

            prompt_id, used_seed = await asyncio.to_thread(submit_edit_comfy, detailed_prompt, width, height)
            position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)

            status_msg = await message.answer(
                f"⌛ Идёт генерация картинки {width}x{height}.\n{_format_queue_info(position, running)}Может занять до 10 мин..."
            )
            await bot.send_chat_action(chat_id, "upload_photo", message_thread_id=thread_id)

            photo_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)
            await bot.send_photo(chat_id=chat_id, photo=BufferedInputFile(photo_bytes, filename="comfy_image.png"), message_thread_id=thread_id)
            await message.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n📐 {width}x{height} | 🌱 Seed: {used_seed}")
        else:
            status_msg = await message.answer("⌛ Идёт генерация картинки (может занять до 10 мин)...")
            detailed_prompt = await generate_image_prompt(chat_id, thread_id, prompt_text, user_id=user_id)
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
    cleaned_text, resolution = extract_video_resolution(raw_text)
    cleaned_text, duration = extract_video_duration(cleaned_text)
    request_text = cleaned_text.replace(CHAT_TRIGGER_WORD, '').replace(VIDEO_TRIGGER_COMMAND, '').strip()
    prompt_text = f'{user_name}: {request_text}'
    chat_id = message.chat.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    status_msg = None

    width, height = resolution if resolution else (360, 640)
    duration = duration if duration else 10

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
        prompt_id, used_seed = await asyncio.to_thread(submit_video_t2v_comfy, detailed_prompt, width, height, duration)

        # 3. Первое и единственное статус-сообщение — сразу с очередью
        position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)
        status_msg = await message.answer(
            f"⌛ Идёт генерация видео {width}x{height}, {duration} сек.\n"
            f"{_format_queue_info(position, running)}"
            f"Это надолго, жди..."
        )
        await bot.send_chat_action(chat_id, "upload_video", message_thread_id=thread_id)

        # 4. Ожидание результата
        print(f'Генерация видео T2V ({width}x{height}, {duration} сек)...')
        video_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)

        await send_video_to_chat(chat_id, thread_id, video_bytes, width, height)
        await message.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n📐 {width}x{height} | ⏱ {duration} сек | 🌱 Seed: {used_seed}")
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
    cleaned_text, resolution = extract_resolution(raw_text)          # <-- НОВОЕ: сначала вырезаем разрешение
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

            prompt_id, used_seed = await asyncio.to_thread(submit_image_comfy, detailed_prompt, width, height)
            position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)

            status_msg = await message.answer(
                f"⌛ Идёт генерация картинки {width}x{height}.\n{_format_queue_info(position, running)}Может занять до 10 мин..."
            )
            await bot.send_chat_action(chat_id, "upload_photo", message_thread_id=thread_id)

            photo_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)
            await bot.send_photo(chat_id=chat_id, photo=BufferedInputFile(photo_bytes, filename="comfy_image.png"), message_thread_id=thread_id)
            await message.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n📐 {width}x{height} | 🌱 Seed: {used_seed}")
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
    cleaned_caption, resolution = extract_video_resolution(raw_caption)
    cleaned_caption, duration = extract_video_duration(cleaned_caption)
    request_text = cleaned_caption.replace(CHAT_TRIGGER_WORD, '').replace(VIDEO_TRIGGER_COMMAND, '').strip()
    prompt_text = f'{user_name}: {request_text}' if request_text else \
                  f'{user_name}: оживи это изображение — придумай естественное движение и звук, сохраняя суть сцены'
    status_msg = None

    width, height = resolution if resolution else (360, 640)
    duration = duration if duration else 10

    try:
        comfy_online = await asyncio.to_thread(is_comfyui_online)
        if not comfy_online:
            await message.reply(f"🖥❌ Генератор временно отключён, попробуй позже.")
            return

        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_bytes = (await bot.download_file(file.file_path)).read()

        # 1. Vision-модель сочиняет промт по картинке
        await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
        encoded_image = base64.b64encode(file_bytes).decode('utf-8')
        detailed_prompt = await generate_video_i2v_prompt(chat_id, thread_id, prompt_text, encoded_image, user_id=user_id, duration=duration)

        # 2. Постановка в очередь
        prompt_id, used_seed = await asyncio.to_thread(submit_video_i2v_comfy, detailed_prompt, file_bytes, width, height, duration)

        # 3. Статус с очередью
        position, running = await asyncio.to_thread(get_comfy_queue_position, prompt_id)
        status_msg = await message.answer(
            f"⌛ Анимирую твою картинку ({width}x{height}, {duration} сек).\n"
            f"{_format_queue_info(position, running)}"
            f"Это надолго, жди..."
        )
        await bot.send_chat_action(chat_id, "upload_video", message_thread_id=thread_id)

        # 4. Ожидание
        print(f'Генерация видео I2V ({width}x{height}, {duration} сек)...')
        video_bytes = await asyncio.to_thread(wait_comfy_result, prompt_id)

        await send_video_to_chat(chat_id, thread_id, video_bytes, width, height)
        await message.answer(f"📝 Промт:\n\n{detailed_prompt}\n\n📐 {width}x{height} | ⏱ {duration} сек | 🌱 Seed: {used_seed}")
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
            
        IGNORED_USERS = {5305038935, 8320108709, 1467812279}
            
        # --- БЛОК АВТОМАТИЧЕСКОЙ РЕАКЦИИ ---
        user_id = event.from_user.id if event.from_user else None
        if chat_id == -1002232705531 and topic_id == 196220 and user_id not in IGNORED_USERS: # -1002232705531 and topic_id == 196220:
            try:
                await event.react([aiogram_types.ReactionTypeEmoji(emoji='💩')])
            except Exception as e:
                print(f"❌ Не удалось поставить реакцию: {e}")
        # ----------------------------------

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
    
    dp.message.outer_middleware(HistoryMiddleware())
    main_router.message.register(cmd_start, CommandStart())
    
    scheduler.add_job(daily_summary_executor, 'interval', minutes=1)
    scheduler.start()
    
    from bot_dialogue import restore_dialogues
    await restore_dialogues(bot)
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)