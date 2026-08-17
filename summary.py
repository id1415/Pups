import os
import json
import asyncio
import aiohttp
from datetime import datetime
import pytz
from aiogram.types import BufferedInputFile
from utils import get_chat_log, clear_chat_log
from variables import PROMPT, CHAT_TRIGGER_WORD, bot, WORKER_URL, WORKER_API_KEY
import gemini

SUMMARY_CONFIG_FILE = "summary_config.json"

def get_summary_config() -> dict:
    if not os.path.exists(SUMMARY_CONFIG_FILE):
        return {}
    with open(SUMMARY_CONFIG_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def is_summary_enabled(chat_id: int) -> bool:
    config = get_summary_config()
    chat_cfg = config.get(str(chat_id))
    if isinstance(chat_cfg, dict):
        return chat_cfg.get("enabled", False)
    return bool(chat_cfg)

def set_summary_state(chat_id: int, state: bool, thread_id: int = None, target_time: str = "21:00"):
    config = get_summary_config()
    cid = str(chat_id)
    
    if state:
        config[cid] = {
            "enabled": True,
            "thread_id": thread_id,
            "time": target_time
        }
    else:
        if cid in config:
            config[cid]["enabled"] = False

    with open(SUMMARY_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

async def generate_worker_image(prompt_text: str):
    """Отправляет запрос к Cloudflare Worker и возвращает байты картинки."""
    if not WORKER_URL or not WORKER_API_KEY:
        print("Ошибка: WORKER_URL или WORKER_API_KEY не заданы в .env")
        return None

    headers = {
        "Authorization": f"Bearer {WORKER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt_text,
        "width": 1024,
        "height": 1024
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(WORKER_URL, json=payload, headers=headers, timeout=500) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    error_text = await resp.text()
                    print(f"Ошибка Worker API ({resp.status}): {error_text}")
                    return None
    except Exception as e:
        print(f"Исключение при запросе к Worker: {e}")
        return None

async def process_pupps_summary(chat_id: int, log_data: list, thread_id: int = None):
    print('----------------------------')
    print(f"Обработка саммари для чата {chat_id}, топик {thread_id}")
    full_history_text = "\n".join([f"{m['user']}: {m['text']}" for m in log_data])
    summary_system_prompt = (
        "Тебя зовут Пупс. Дерзи, стебись, прикалывайся, подкалывай, матерись. Отвечай с юмором. "
        "Ты против сексуальных меньшинств. "
        "Тебе подкинули лог базара из этого чата. "
        "Сделай развёрнутый, резкий, дерзкий и угарный пересказ того, о чем эти персонажи тут обсуждали. "
        "Ограничение - 3000 символов. "
        f"Вот их писанина:\n\n{full_history_text}"
    )
    
    try:
        summary = await gemini.priem_summary(summary_system_prompt)
        print(summary)
        
        await bot.send_message(
            chat_id=chat_id, 
            message_thread_id=thread_id,
            text=f"🔥 **{CHAT_TRIGGER_WORD.upper()} ПОЯСНЯЕТ ЗА ПРОШЕДШИЙ БАЗАР:**\n\n{summary}"
        )
        clear_chat_log(chat_id)
    except Exception as e:
        print(f"Ошибка саммари: {e}")
        print('----------------------------')
        return
        
    image_prompt_system = (
        f"{PROMPT}\n\n"
        "Забудь про русский язык и свой стиль диалога. Твоя задача — создать подробный промт на английском языке для генерации картинки (Flux model).\n"
        "Прочитай саммари ниже и сгенерируй детальный описательный промт (English image generation prompt) с 1-2 ключевыми героями/персонажами из этого пересказа в смешной, яркой или абсурдной обстановке на основе произошедшего.\n"
        "ВЫДАВАЙ ТОЛЬКО АНГЛИЙСКИЙ ПРОМТ, БЕЗ ЛИШНЕГО ТЕКСТА И ВВОДНЫХ СЛОВ!\n\n"
        f"Вот текст саммари:\n{summary}"
    )

    try:
        image_prompt = await gemini.priem_summary(image_prompt_system)
        image_prompt = image_prompt.strip()
        print(f"Сгенерированный промт для картинки: {image_prompt}")

        # Запрашиваем генерацию у Cloudflare Worker и отправляем результат
        image_bytes = await generate_worker_image(image_prompt)
        if image_bytes:
            buffered_file = BufferedInputFile(image_bytes, filename="summary_illustration.png")
            await bot.send_photo(
                chat_id=chat_id,
                message_thread_id=thread_id,
                photo=buffered_file,
                caption=f"🎨 **Иллюстрация прошедшего дня:**\n`{image_prompt}`"
            )
        else:
            print("Не удалось получить изображение от Cloudflare Worker.")
    except Exception as e:
        print(f"Ошибка при генерации/отправке картинки для саммари: {e}")

    print('----------------------------')
        
async def daily_summary_executor():
    """Вызывается каждую минуту планировщиком и проверяет время отправки саммари."""
    config = get_summary_config()
    if not config:
        return

    # Получаем текущее время по Москве в формате HH:MM
    moscow_tz = pytz.timezone('Europe/Moscow')
    current_time = datetime.now(moscow_tz).strftime("%H:%M")

    for chat_id_str, settings in list(config.items()):
        # Совместимость со старым форматом boolean
        if isinstance(settings, bool):
            continue
            
        if settings.get("enabled") and settings.get("time") == current_time:
            chat_id = int(chat_id_str)
            thread_id = settings.get("thread_id")
            log_data = get_chat_log(chat_id)
            
            if not log_data:
                set_summary_state(chat_id, False)
                try:
                    await bot.send_message(
                        chat_id=chat_id, 
                        message_thread_id=thread_id,
                        text=f'Слышь, вы че, вымерли тут все? За весь день ни одной живой души. Короче, я тушу пересказ, ловить тут нечего. Если будет базар, админ введи команду "{CHAT_TRIGGER_WORD} вкл пересказ" заново. А пока идите траву трогайте, мудаки!'
                    )
                except Exception as e:
                    print(f"Не удалось отправить уведомление о пустом логе в чат {chat_id}: {e}")
                continue
            
            try:
                await process_pupps_summary(chat_id, log_data, thread_id=thread_id)
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Ошибка при обработке очереди саммари для {chat_id}: {e}")
                continue