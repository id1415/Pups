import os
import json
from datetime import datetime
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import pytz

load_dotenv()

MEMORY_DIR = "chat_memories"
KEYS_DIR = "user_keys"
LOGS_DIR = "chat_logs"
MASTER_KEY = os.getenv("MASTER_CRYPTO_KEY")
CHAT_SETTINGS_FILE = "chat_settings.json"
chat_settings_cache = {}
VISION_SETTINGS_FILE = "vision_settings.json"
vision_settings_cache = {}
IMAGE_SETTINGS_FILE = "image_settings.json"
image_settings_cache = {}
MUSIC_SETTINGS_FILE = "music_settings.json"
music_settings_cache = {}

def _get_fernet():
    """Инициализирует объект шифрования, если ключ задан."""
    if not MASTER_KEY:
        # Если забыли добавить ключ в .env, бот упадет с понятной ошибкой
        raise ValueError("Критическая ошибка: MASTER_CRYPTO_KEY не задан в .env файле!")
    return Fernet(MASTER_KEY.encode())

def _get_key_path(user_id: int) -> str:
    return os.path.join(KEYS_DIR, f"{user_id}_key.json")

def load_user_key(user_id: int) -> str:
    """Загружает, расшифровывает и возвращает API-ключ пользователя."""
    file_path = _get_key_path(user_id)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            encrypted_key = data.get("api_key")
            
            if not encrypted_key:
                return None
                
            # Расшифровываем
            cipher = _get_fernet()
            decrypted_bytes = cipher.decrypt(encrypted_key.encode())
            return decrypted_bytes.decode('utf-8')
            
    except Exception as e:
        print(f"Ошибка загрузки/расшифровки ключа пользователя {user_id}: {e}")
        return None

def save_user_key(user_id: int, api_key: str):
    """Шифрует и сохраняет персональный ключ пользователя."""
    if not os.path.exists(KEYS_DIR):
        os.makedirs(KEYS_DIR)
    file_path = _get_key_path(user_id)
    try:
        if api_key is None:
            # Если ключ равен None (команда удаления), просто записываем пустую структуру или удаляем файл
            if os.path.exists(file_path):
                os.remove(file_path)
            return True

        # Шифруем ключ перед сохранением
        cipher = _get_fernet()
        encrypted_bytes = cipher.encrypt(api_key.encode())
        encrypted_string = encrypted_bytes.decode('utf-8')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "user_id": user_id, 
                "api_key": encrypted_string  # В файл улетает зашифрованная абракадабра
            }, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Ошибка шифрования/сохранения ключа пользователя {user_id}: {e}")
        return False

def _get_memory_path(chat_id: int) -> str:
    return os.path.join(MEMORY_DIR, f"{chat_id}_memory.json")

def load_memory(chat_id: int):
    file_path = _get_memory_path(chat_id)
    default_memory = {
        "chat_id": chat_id, 
        "history": [] 
    }
    if not os.path.exists(file_path):
        return default_memory
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки памяти: {e}")
        return default_memory

def save_memory(chat_id: int, chat_memory_data: dict):
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)
    file_path = _get_memory_path(chat_id)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chat_memory_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Ошибка при сохранении памяти: {e}")

def append_history(memory_data: dict, opponent_message: str = None, my_response: str = None, max_history: int = 150) -> dict:
    moscow_tz = pytz.timezone('Europe/Moscow')
    timestamp = datetime.now(moscow_tz).strftime("[%Y-%m-%d %H:%M:%S]")
    
    # Добавляем сообщение пользователя
    if opponent_message:
        user_content = f"{timestamp} {opponent_message}"
        memory_data["history"].append({"role": "user", "content": user_content})
    
    # Добавляем ответ ассистента только если он есть
    if my_response:
        memory_data["history"].append({"role": "assistant", "content": my_response})
    
    # Если сообщений стало больше лимита — отрезаем старые
    if len(memory_data["history"]) > max_history:
        memory_data["history"] = memory_data["history"][-max_history:]
            
    return memory_data
    
def clear_memory(chat_id: int):
    """Удаляет файл памяти чата, если он существует."""
    file_path = _get_memory_path(chat_id)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Ошибка при удалении файла памяти: {e}")
            return False
    return True # Если файла нет, считаем, что память и так чиста
    
def get_chat_model(chat_id: int) -> str:
    cid = str(chat_id)
    if cid in chat_settings_cache:
        return chat_settings_cache[cid].get("model_name", "unmoderated-gpt")
    return "unmoderated-gpt"  # Модель по умолчанию, если чата нет в базе
    
def get_image_model(chat_id: int) -> str:
    cid = str(chat_id)
    if cid in image_settings_cache:
        return image_settings_cache[cid].get("model_name", "flux-2-pro")
    return "flux-2-klein-9b"  # Модель по умолчанию, если чата нет в базе
    
def get_vision_model(chat_id: int) -> str:
    cid = str(chat_id)
    if cid in vision_settings_cache:
        return vision_settings_cache[cid].get("model_name", "grok-4.20-beta")
    return "grok-4.20-beta"  # Модель по умолчанию, если чата нет в базе
    
def get_music_model(chat_id: int) -> str:
    cid = str(chat_id)
    if cid in music_settings_cache:
        return music_settings_cache[cid].get("model_name", "suno-v5.5")
    return "suno-v5.5"  # Модель по умолчанию, если чата нет в базе
    
def save_chat_model(chat_id: int, model_name: str):
    cid = str(chat_id)
    if cid not in chat_settings_cache:
        chat_settings_cache[cid] = {"trigger_word": "пупс"}
    
    chat_settings_cache[cid]["model_name"] = model_name.strip()
    with open(CHAT_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_settings_cache, f, ensure_ascii=False, indent=4)
        
def save_image_model(chat_id: int, model_name: str):
    cid = str(chat_id)
    if cid not in image_settings_cache:
        image_settings_cache[cid] = {"trigger_word": "пупс"}
    
    image_settings_cache[cid]["model_name"] = model_name.strip()
    with open(IMAGE_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(image_settings_cache, f, ensure_ascii=False, indent=4)
        
def save_vision_model(chat_id: int, model_name: str):
    cid = str(chat_id)
    if cid not in vision_settings_cache:
        vision_settings_cache[cid] = {"trigger_word": "пупс"}
    
    vision_settings_cache[cid]["model_name"] = model_name.strip()
    with open(VISION_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(vision_settings_cache, f, ensure_ascii=False, indent=4)
    
def save_music_model(chat_id: int, model_name: str):
    cid = str(chat_id)
    if cid not in music_settings_cache:
        music_settings_cache[cid] = {"trigger_word": "пупс"}
    
    music_settings_cache[cid]["model_name"] = model_name.strip()
    with open(MUSIC_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(music_settings_cache, f, ensure_ascii=False, indent=4)

def load_chat_settings():
    global chat_settings_cache
    if os.path.exists(CHAT_SETTINGS_FILE):
        try:
            with open(CHAT_SETTINGS_FILE, "r", encoding="utf-8") as f:
                chat_settings_cache = json.load(f)
        except Exception:
            chat_settings_cache = {}
    else:
        chat_settings_cache = {}
        
def load_image_settings():
    global image_settings_cache
    if os.path.exists(IMAGE_SETTINGS_FILE):
        try:
            with open(IMAGE_SETTINGS_FILE, "r", encoding="utf-8") as f:
                image_settings_cache = json.load(f)
        except Exception:
            image_settings_cache = {}
    else:
        image_settings_cache = {}

def load_vision_settings():
    global vision_settings_cache
    if os.path.exists(VISION_SETTINGS_FILE):
        try:
            with open(VISION_SETTINGS_FILE, "r", encoding="utf-8") as f:
                vision_settings_cache = json.load(f)
        except Exception:
            vision_settings_cache = {}
    else:
        vision_settings_cache = {}
        
def load_music_settings():
    global music_settings_cache
    if os.path.exists(MUSIC_SETTINGS_FILE):
        try:
            with open(MUSIC_SETTINGS_FILE, "r", encoding="utf-8") as f:
                music_settings_cache = json.load(f)
        except Exception:
            music_settings_cache = {}
    else:
        music_settings_cache = {}
        
def get_log_path(chat_id: int) -> str:
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    return os.path.join(LOGS_DIR, f"log_{chat_id}.json")

def get_chat_log(chat_id: int):
    path = get_log_path(chat_id)
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except:
            return []

def save_chat_log(chat_id: int, log_data: list):
    moscow_tz = pytz.timezone('Europe/Moscow')
    now = datetime.now(moscow_tz).strftime("[%Y-%m-%d %H:%M:%S]")
    
    if log_data and isinstance(log_data[-1], str):
        log_data[-1] = f"{log_data[-1]}"
        
    with open(get_log_path(chat_id), 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

def clear_chat_log(chat_id: int):
    path = get_log_path(chat_id)
    if os.path.exists(path):
        os.remove(path)