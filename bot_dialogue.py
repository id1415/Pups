import re
import logging
import pytz
import os
import json
import asyncio
import os
from datetime import datetime
from aiogram import Router, Bot, types as aiogram_types

dialogue_router = Router()
logger = logging.getLogger(__name__)

# Интервал отправки сообщений (в минутах)
DEFAULT_INTERVAL_MINUTES = 60
DIALOGUE_CONFIG_FILE = "dialogue_config.json"

def load_dialogue_config() -> dict:
    """Загружает конфигурацию авто-сообщений из файла."""
    if not os.path.exists(DIALOGUE_CONFIG_FILE):
        return {}
    with open(DIALOGUE_CONFIG_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка чтения {DIALOGUE_CONFIG_FILE}: {e}")
            return {}

def save_dialogue_config(config: dict):
    """Сохраняет конфигурацию авто-сообщений в файл."""
    try:
        with open(DIALOGUE_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Ошибка записи в {DIALOGUE_CONFIG_FILE}: {e}")

def load_dialogue_state():
    """
    Динамически воссоздает структуру состояния на основе активных задач в планировщике.
    Необходимо для обратной совместимости с HistoryMiddleware.
    """
    from bot11_pups import scheduler
    state = {"dialogues": {}}
    try:
        for job in scheduler.get_jobs():
            if job.id.startswith("dialogue_"):
                parts = job.id.split("_")
                if len(parts) == 3:
                    chat_id, topic_id = parts[1], parts[2]
                    dialog_key = f"{chat_id}_{topic_id}"
                    state["dialogues"][dialog_key] = {"is_active": True}
    except Exception as e:
        logger.error(f"Ошибка при чтении задач планировщика: {e}")
    return state
    
async def restore_dialogues(bot: Bot):
    """Восстанавливает авто-сообщения из конфига при запуске бота."""
    from bot11_pups import scheduler
    config = load_dialogue_config()
    moscow_tz = pytz.timezone('Europe/Moscow')
    
    count = 0
    for job_id, info in config.items():
        if info.get("is_active"):
            chat_id = info["chat_id"]
            topic_id = info["topic_id"]
            interval_minutes = info.get("interval_minutes", DEFAULT_INTERVAL_MINUTES)
            
            scheduler.add_job(
                send_dialogue_message,
                'interval',
                minutes=interval_minutes,
                id=job_id,
                args=[chat_id, topic_id, bot],
                replace_existing=True,
                timezone=moscow_tz
            )
            count += 1
            
    if count > 0:
        logger.info(f"Успешно восстановлено авто-задач: {count}")

async def send_dialogue_message(chat_id: int, topic_id: int, bot: Bot):
    """Фоновая задача: запрашивает ответ у ИИ и отправляет его в текущий топик/чат"""
    from bot11_pups import generate_response
    import prompt

    try:
        # Запрашиваем генерацию. Контекст автоматически подтягивается из chat_memories.
        response_text = await generate_response(
            chat_id=chat_id,
            thread_id=topic_id,
            system_prompt=prompt.PUPPS5,  # У Деда в его папке здесь будет свой prompt.DED
            current_user_message=""
        )

        if response_text:
            # Отправляем сообщение обратно в тот же топик (если topic_id=None, отправится в корень чата)
            await bot.send_message(chat_id, response_text, message_thread_id=topic_id)
            
    except Exception as e:
        logger.error(f"Ошибка авто режима в чате {chat_id}, топик {topic_id}: {e}")

# --- Новые фильтры команд без ID топика ---

def start_command_filter(message: aiogram_types.Message) -> bool:
    if not message.text:
        return False
    from bot11_pups import CHAT_TRIGGER_WORD
    trigger = CHAT_TRIGGER_WORD.lower()
    text = message.text.lower().strip()
    return bool(re.match(rf'^({trigger})\s+start(?:\s+(\d+))?$', text))

def stop_command_filter(message: aiogram_types.Message) -> bool:
    if not message.text:
        return False
    from bot11_pups import CHAT_TRIGGER_WORD
    trigger = CHAT_TRIGGER_WORD.lower()
    text = message.text.lower().strip()
    return bool(re.match(rf'^({trigger})\s+stop$', text))

# --- Хэндлеры команд ---

@dialogue_router.message(start_command_filter)
async def start_dialogue_cmd(message: aiogram_types.Message, bot: Bot):
    from bot11_pups import scheduler, CHAT_TRIGGER_WORD
    
    chat_id = message.chat.id
    topic_id = message.message_thread_id 
    trigger = CHAT_TRIGGER_WORD.lower()
    match = re.match(rf'^({trigger})\s+start(?:\s+(\d+))?$', message.text.lower().strip())
    interval_minutes = DEFAULT_INTERVAL_MINUTES
    
    if match and match.group(2): # Если число было передано
        interval_minutes = int(match.group(2))
        
        # Защита: проверяем, чтобы интервал был не меньше 30 минут
        if interval_minutes < 30:
            await message.reply("❌ Ошибка! Временной интервал не может быть меньше 30 минут, иначе боты быстро исчерпают лимиты API.")
            return
    
    # --- КРИТИЧЕСКАЯ ПРОВЕРКА: Ищем любую активную задачу в ЭТОМ чате ---
    active_job = None
    for job in scheduler.get_jobs():
        if job.id.startswith(f"dialogue_{chat_id}_"):
            active_job = job
            break
    
    if active_job:
        await message.reply("🔄 Режим автоматических сообщений уже запущен на этом канале.")
        return
    
    '''if active_job:
        # Извлекаем из ID задачи топик, в котором она запущена
        running_topic_id = int(active_job.id.split("_")[2])
        
        if running_topic_id == (topic_id or 0):
            await message.reply("🔄 Режим автоматических сообщений уже запущен на этом канале.")
        else:
            await message.reply("🔄 Режим автоматических сообщений уже запущен на этом канале.")
        return'''
    
    # Для ID задачи используем 0, если это корень чата, чтобы не ломать форматирование строки
    job_id = f"dialogue_{chat_id}_{topic_id or 0}"
        
    # Регистрируем интервальную задачу строго для этого топика/канала
    moscow_tz = pytz.timezone('Europe/Moscow')
    scheduler.add_job(
        send_dialogue_message,
        'interval',
        minutes=interval_minutes,
        id=job_id,
        args=[chat_id, topic_id, bot],
        replace_existing=True,
        timezone=moscow_tz
    )
    
    config = load_dialogue_config()
    config[job_id] = {
        "chat_id": chat_id,
        "topic_id": topic_id,
        "interval_minutes": interval_minutes,
        "is_active": True
    }
    save_dialogue_config(config)
    
    bot_name = CHAT_TRIGGER_WORD.capitalize()
        
    await message.reply(f"✅ Режим авто сообщений активирован! {bot_name} будет писать сюда каждые {interval_minutes} мин.")
    
    # Сразу генерируем и отправляем первую реплику
    await send_dialogue_message(chat_id, topic_id, bot)

@dialogue_router.message(stop_command_filter)
async def stop_dialogue_cmd(message: aiogram_types.Message):
    from bot11_pups import scheduler
    
    chat_id = message.chat.id
    topic_id = message.message_thread_id
    job_id = f"dialogue_{chat_id}_{topic_id or 0}"
    config = load_dialogue_config()
    removed = False
    
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
        if job_id in config:
            config[job_id]["is_active"] = False
        removed = True
    else:
        jobs = scheduler.get_jobs()
        for job in jobs:
            if job.id.startswith(f"dialogue_{chat_id}_"):
                scheduler.remove_job(job.id)
                if job.id in config:
                    config[job.id]["is_active"] = False
                removed = True
                break
                
    if removed:
        save_dialogue_config(config)
        await message.reply("🛑 Режим авто сообщений остановлен.")
    else:
        await message.reply("В данном чате режим авто сообщений не был запущен.")