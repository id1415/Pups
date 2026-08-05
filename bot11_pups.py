import asyncio
import json
import os
import requests
import base64
import re
import prompt
import time
import torch
import librosa
import soundfile as sf
import speech_recognition as sr
from pydub import AudioSegment
import io
from bot_dialogue import dialogue_router
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from aiogram import BaseMiddleware
from aiogram.filters import CommandStart, Filter
from aiogram import Dispatcher, Router, types as aiogram_types, F
from aiogram.types import Message, BufferedInputFile
from aiogram.enums import ParseMode
from google.genai import types
from utils import (load_memory, save_memory, append_history, clear_memory,
                   load_airforce_user_key, save_airforce_user_key, load_gemini_user_key, save_gemini_user_key,
                   get_chat_model, get_image_model, get_vision_model, get_music_model, 
                   save_chat_model, save_image_model, save_vision_model, save_music_model, 
                   load_chat_settings, load_image_settings, load_vision_settings, load_music_settings,
                   get_chat_log, save_chat_log, clear_chat_log,  # для функции пересказа
                   save_chat_max_history)
from summary import is_summary_enabled, set_summary_state, process_pupps_summary, daily_summary_executor
from variables import (CHAT_TRIGGER_WORD, IMAGE_TRIGGER_COMMAND, MUSIC_TRIGGER_COMMAND, PROMPT, MAX_RETRIES, RETRY_DELAY,
                       AIRFORCE_API_URL, AIRFORCE_API_KEY, IMGBB_API_KEY, PUPS_BOT_TOKEN, bot, API_KEY_GEMINI,
                       client)
import pupps_info
import get_models_info
from gemini import priem as gemini_priem, priem_vision as gemini_priem_vision, priem_video as gemini_priem_video

# Загружаем модель (делается 1 раз при старте)
device = torch.device('cpu')
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v4_ru'
)
model.to(device)

async def text_to_voice_bytes_silero(text: str) -> io.BytesIO:
    sample_rate = 48000
    speaker = 'aidar'

    # 1. Генерируем аудио-тензор
    audio = model.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate,
        put_accent=True,
        put_yo=True
    )
    
    # 2. Делаем массив 1D (.squeeze() убирает лишнюю размерность [1, N] -> [N])
    y = audio.squeeze().numpy()

    # 3. Применяем pitch_shift (значение от 3.0 до 6.0 дает классный детский/высокий тон)
    n_steps = 10
    y_child = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=n_steps)
    
    # 4. Сохраняем в BytesIO
    wav_io = io.BytesIO()
    sf.write(wav_io, y_child, sample_rate, format='OGG', subtype='OPUS')
    wav_io.seek(0)
    return wav_io

commands = ['нейробот инфо', 'нейробот name', 'нейробот prompt', 'нейробот chat', 'нейробот vision', 'нейробот image', 'нейробот music', 'нейробот start', 'нейробот stop', 'нейробот 0',
            'кибербот инфо', 'кибербот name', 'кибербот prompt', 'кибербот chat', 'кибербот vision', 'кибербот image', 'кибербот music', 'кибербот start', 'кибербот stop', 'кибербот 0',
            'пупс инфо', 'пупс chat', 'пупс vision', 'пупс image', 'пупс music', 'пупс start', 'пупс stop', 'пупс 0',
            'няша инфо', 'няша chat', 'няша vision', 'няша image', 'няша music', 'няша start', 'няша stop', 'няша 0',
            'дед инфо', 'дед chat', 'дед vision', 'дед image', 'дед music', 'дед start', 'дед stop', 'дед 0',
            'джейсон инфо', 'джейсон chat', 'джейсон vision', 'джейсон image', 'джейсон music', 'джейсон start', 'джейсон stop', 'джейсон 0',
            'нуар инфо', 'нуар chat',  'нуар vision', 'нуар image', 'нуар music', 'нуар start', 'нуар stop', 'нуар 0',
            'пупс context']

scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
dp = Dispatcher()
main_router = Router()
dp.include_router(dialogue_router)
dp.include_router(main_router)
        
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
        return await gemini_priem_vision(chat_id, current_user_message, base64_image, user_key=active_key)
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
    
    try:
        status_msg = await message.answer(f"⌛ Жди, {CHAT_TRIGGER_WORD.capitalize()} переделает твою картинку (может занять до 10 мин)...")

        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        
        file_bytes_io = await bot.download_file(file.file_path)
        file_bytes = file_bytes_io.read() # Получаем чистые байты
        imgbb_url = await asyncio.to_thread(upload_to_imgbb, file_bytes, None, chat_id, thread_id)
        
        if not imgbb_url:
            raise Exception("Не удалось загрузить изображение на хостинг")
            
        encoded_image = base64.b64encode(file_bytes).decode('utf-8')

        detailed_prompt = await generate_vision_response(chat_id, thread_id, PROMPT, prompt_text, encoded_image)
        
        await asyncio.sleep(65)
        
        path = await asyncio.to_thread(generate_media_sync, 
                                       user_id,
                                       detailed_prompt, 
                                       chat_id, 
                                       thread_id, 
                                       image_url=[imgbb_url], 
                                       aspect_ratio=aspect_ratio)
        
        if path:
            print('Отправка в тг...')
            path = upload_to_imgbb(None, path, chat_id, thread_id)
            await bot.send_photo(
                chat_id=chat_id, 
                photo=path, 
                message_thread_id=thread_id
            )
            await message.answer(f"📝 **Промт:**\n\n`{detailed_prompt}`")
        print('Готово!')

    except Exception as e:
        print(f"Ошибка генерации: {e}")
        await send_log_to_telegram(str(e), "Edit_Photo", chat_id, thread_id, user_name)
    finally:
        if status_msg:
            await status_msg.delete()

@main_router.message(F.text, lambda m: f'{CHAT_TRIGGER_WORD}' in m.text.lower() and IMAGE_TRIGGER_COMMAND in m.text.lower())
async def handle_image_generation(message: aiogram_types.Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id
    raw_text = message.text.lower()
    cleaned_text, aspect_ratio = extract_aspect_ratio(raw_text)
    prompt_text = f'{user_name}: {cleaned_text}'
    chat_id = message.chat.id
    thread_id = message.message_thread_id if message.is_topic_message else None
    status_msg = None
    
    current_image_model = get_image_model(chat_id)
    
    try:
        status_msg = await message.answer("⌛ Идёт генерация картинки (может занять до 10 мин)...")
        detailed_prompt = await generate_response(chat_id, thread_id, PROMPT, prompt_text, user_id=user_id)
        await asyncio.sleep(65)

        print('Генерация...')
        
        path = await asyncio.to_thread(generate_media_sync, user_id, detailed_prompt, chat_id, thread_id, aspect_ratio=aspect_ratio)
                
        if path:
            print('Отправка в тг...')
            path = upload_to_imgbb(None, path, chat_id, thread_id)
            await bot.send_photo(
                chat_id=chat_id, 
                photo=path, 
                message_thread_id=thread_id
            )
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
        await message.reply(
            f"🤖 Текущая модель для музыки в этом чате: {current_model}\n\n"
            f"Доступные варианты для смены (нажмите на имя, чтобы скопировать):\n"
            f"{models_text}\n\n"  # Выводим уже готовый текст из get_models_info.py
            f"Чтобы изменить, напиши:\n`{CHAT_TRIGGER_WORD} music название_модели`",
            parse_mode="Markdown",  # Важно для работы кликабельности!
        )
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
        await message.reply(
            f"🤖 Текущая модель vision в этом чате: {current_model}\n\n"
            f"Доступные варианты для смены (нажмите на имя, чтобы скопировать):\n"
            f"{models_text}\n\n"  # Выводим уже готовый текст из get_models_info.py
            f"Чтобы изменить, напиши:\n`{CHAT_TRIGGER_WORD} vision название_модели`",
            parse_mode="Markdown",  # Важно для работы кликабельности!
        )
        return
        
    chosen_model = parts[2].strip().lower()
    if chosen_model not in allowed_ids:
        await message.reply("❌ Неверное название модели. Скопируй имя строго из списка доступных.")
        return
        
    save_vision_model(message.chat.id, chosen_model)
    await message.reply(f"✅ Успешно! Теперь в этом чате запросы с картинками отправляются в: {chosen_model}")
    
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
        await message.reply(
            f"🤖 Текущая модель для генерации картинок в этом чате: {current_model}\n\n"
            f"Доступные варианты для смены (нажмите на имя, чтобы скопировать):\n"
            f"{models_text}\n\n"  # Выводим уже готовый текст из get_models_info.py
            f"Чтобы изменить, напиши:\n`{CHAT_TRIGGER_WORD} image название_модели`",
            parse_mode="Markdown",  # Важно для работы кликабельности!
        )
        return
        
    chosen_model = parts[2].strip().lower()
    if chosen_model not in allowed_ids:
        await message.reply("❌ Неверное название модели. Скопируй имя строго из списка доступных.")
        return
        
    save_image_model(message.chat.id, chosen_model)
    await message.reply(f"✅ Успешно! Теперь в этом чате запросы с картинками отправляются в: {chosen_model}")

@main_router.message((F.video | F.video_note), lambda m: (m.caption and f'{CHAT_TRIGGER_WORD}' in m.caption.lower()) or (m.text and f'{CHAT_TRIGGER_WORD}' in m.text.lower()))
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

@main_router.message(F.photo, lambda m: (m.caption and f'{CHAT_TRIGGER_WORD}' in m.caption.lower()))
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

    # 1. Скачиваем ГС
    voice = message.voice
    file_info = await bot.get_file(voice.file_id)
    file_bytes_io = await bot.download_file(file_info.file_path)

    # 2. Конвертируем в WAV
    try:
        audio = AudioSegment.from_file(file_bytes_io, format="ogg")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
    except Exception as e:
        print(f"Ошибка конвертации аудио: {e}")
        return

    # 3. Распознаем речь
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio_data, language="ru-RU")
            print(f"🎙️ [Голос от {user_name}]: {recognized_text}")
    except sr.UnknownValueError:
        print("Гугл не смог разобрать речь в ГС.")
        return
    except Exception as e:
        print(f"Ошибка распознавания: {e}")
        return

    # 4. Проверяем условия триггера
    is_triggered = CHAT_TRIGGER_WORD in recognized_text.lower()

    if is_private or is_triggered:
        text_to_ai = f"{user_name}: {recognized_text}"
        
        try:
            # Ставим статус "Записывает голосовое сообщение..."
            await bot.send_chat_action(chat_id, "record_voice", message_thread_id=thread_id)
            
            # Сохраняем входящее сообщение пользователя в контекст
            memory = load_memory(chat_id)
            formatted_user_msg = f"[Голосовое] {user_name}: {recognized_text}"
            memory = append_history(memory, opponent_message=formatted_user_msg, chat_id=chat_id)
            save_memory(chat_id, memory)

            # Генерируем текстовый ответ ИИ
            response_text = await generate_response(chat_id, thread_id, PROMPT, text_to_ai, user_id=user_id)
            
            # Озвучиваем ответ генератором речи
            voice_bytes = await text_to_voice_bytes_silero(response_text)
            voice_file = BufferedInputFile(voice_bytes.read(), filename="pups_answer.ogg")

            # Отправляем ответное ГОЛОСОВОЕ сообщение
            await bot.send_voice(
                chat_id=chat_id,
                voice=voice_file,
                caption=f"{response_text}" if len(response_text) < 1000 else None, # Опционально: текстовая расшифровка под ГС
                message_thread_id=thread_id,
                reply_to_message_id=message.message_id
            )

            # Сохраняем в лог для саммари
            if is_summary_enabled(chat_id):
                log_data = get_chat_log(chat_id)
                log_data.append({
                    "user": user_name,
                    "text": f"[Голосовое] {recognized_text}",
                    "time": time.strftime("%H:%M")
                })
                log_data.append({
                    "user": CHAT_TRIGGER_WORD.capitalize(),
                    "text": f"[Голосовое] {response_text}",
                    "time": time.strftime("%H:%M")
                })
                save_chat_log(chat_id, log_data)

        except Exception as e:
            print(f"Ошибка при формировании голосового ответа: {e}")
            await send_log_to_telegram(f"{e}", "Voice Handler", chat_id, thread_id, user_name)
            
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
async def handle_set_key(message: aiogram_types.Message):
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
async def handle_set_key(message: aiogram_types.Message):
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