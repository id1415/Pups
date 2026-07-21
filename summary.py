import os
import json
import asyncio

from utils import get_chat_log, clear_chat_log
from variables import PROMPT, CHAT_TRIGGER_WORD, bot
import gemini

SUMMARY_CONFIG_FILE = "summary_config.json"

def is_summary_enabled(chat_id: int) -> bool:
    if not os.path.exists(SUMMARY_CONFIG_FILE):
        return False
    with open(SUMMARY_CONFIG_FILE, 'r') as f:
        try:
            config = json.load(f)
            return config.get(str(chat_id), False)
        except:
            return False

def set_summary_state(chat_id: int, state: bool):
    config = {}
    if os.path.exists(SUMMARY_CONFIG_FILE):
        with open(SUMMARY_CONFIG_FILE, 'r') as f:
            try: config = json.load(f)
            except: config = {}
    config[str(chat_id)] = state
    with open(SUMMARY_CONFIG_FILE, 'w') as f:
        json.dump(config, f)

async def process_pupps_summary(chat_id, log_data):
    print('----------------------------')
    print(chat_id)
    full_history_text = "\n".join([f"{m['user']}: {m['text']}" for m in log_data])
    summary_system_prompt = (
        f"{PROMPT}\n\n"
        "СЛУШАЙ СЮДА: Тебе подкинули лог базара из этого чата. "
        "Сделай резкий, дерзкий и угарный пересказ того, о чем эти персонажи тут обсуждали. "
        "Стебись, выделяй главных героев. Пиши в стиле Пупса. Максимум 3000 символов. "
        f"Вот их писанина:\n\n{full_history_text}" # [-60000:]
    )
    
    try:
        summary = await gemini.priem_summary(summary_system_prompt)
        
        await bot.send_message(
            chat_id, 
            f"🔥 **{CHAT_TRIGGER_WORD.upper()} ПОЯСНЯЕТ ЗА ПРОШЕДШИЙ БАЗАР:**\n\n{summary}"
        )
        clear_chat_log(chat_id)
    except Exception as e:
        print(f"Ошибка саммари: {e}")
        #clear_chat_log(chat_id)
    print('----------------------------')
        
async def daily_summary_executor():
    # Проходит по всем чатам и отправляет саммари, если оно включено
    if not os.path.exists(SUMMARY_CONFIG_FILE):
        return

    with open(SUMMARY_CONFIG_FILE, 'r') as f:
        try:
            config = json.load(f)
        except:
            return

    for chat_id_str, enabled in list(config.items()):
        if enabled:
            chat_id = int(chat_id_str)
            log_data = get_chat_log(chat_id)
            
            if not log_data:
                set_summary_state(chat_id, False)
                try:
                    await bot.send_message(
                        chat_id, 
                        f'Слышь, вы че, вымерли тут все? За весь день ни одной живой души. Короче, я тушу пересказ, ловить тут нечего. Если будет базар, админ введи команду "{CHAT_TRIGGER_WORD} вкл пересказ" заново. А пока идите траву трогайте, мудаки!'
                    )
                except Exception as e:
                    print(f"Не удалось отправить уведомление о пустом логе в чат {chat_id}: {e}")
                continue
            
            try:
                await process_pupps_summary(chat_id, log_data)
                await asyncio.sleep(30)
            except Exception as e:
                print(f"Ошибка при обработке очереди саммари для {chat_id}: {e}")
                continue