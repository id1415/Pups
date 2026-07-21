# gemini-3.1-flash-lite-preview
import os
import base64
import asyncio
import prompt
from google import genai
from google.genai import types
from google.genai import Client
from variables import API_KEY_GEMINI, PROMPT
from utils import load_memory, save_memory, append_history

client = genai.Client(
    api_key=API_KEY_GEMINI,
)

def _get_gemini_client(user_key: str = None) -> genai.Client:
    """Возвращает клиент Gemini с пользовательским или системным ключом."""
    active_key = user_key if user_key else API_KEY_GEMINI
    return genai.Client(api_key=active_key)
    
async def priem_summary(prompt_text: str, user_key: str = None) -> str:
    client = _get_gemini_client(user_key)
    
    config = types.GenerateContentConfig(
        temperature=0.9,
    )
    
    for attempt in range(5):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt_text,
                config=config
            )
            
            if response and response.text and response.text.strip():
                return response.text.strip()
                
        except Exception as e:
            print(f"❌ Ошибка Gemini Summary (попытка {attempt + 1}): {e}")
            
        if attempt < 4:
            await asyncio.sleep(5)
            
    raise RuntimeError("Не удалось сгенерировать саммари.")

async def priem(chat_id: int, current_user_message: str, user_key: str = None) -> str:
    client = _get_gemini_client(user_key)
    memory = load_memory(chat_id)
    
    contents = []
    for item in memory.get("history", []):
        role = item.get("role")
        content = item.get("content")
        if not content and "parts" in item:
            content = item["parts"][0]["text"]
            
        if role in ["user", "assistant"] and content:
            gemini_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                )
            )

    config = types.GenerateContentConfig(
        temperature=0.9,
        system_instruction=PROMPT
    )
    
    for attempt in range(5):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=contents,
                config=config
            )
            
            if response and response.text and response.text.strip():
                final_content = response.text.strip()
                updated_memory = append_history(memory, my_response=final_content)
                save_memory(chat_id, updated_memory)
                print(final_content)
                return final_content
                
        except Exception as e:
            print(f"❌ Ошибка Gemini API (попытка {attempt + 1}): {e}")
            
        if attempt < 4:
            await asyncio.sleep(5)
            
    raise RuntimeError("Не удалось получить ответ от Gemini.")


async def priem_vision(chat_id: int, current_user_message: str, base64_image: str, user_key: str = None) -> str:
    client = _get_gemini_client(user_key)
    memory = load_memory(chat_id)
    
    contents = []
    for item in memory.get("history", []):
        role = item.get("role")
        content = item.get("content")
        if not content and "parts" in item:
            content = item["parts"][0]["text"]
            
        if role in ["user", "assistant"] and content:
            gemini_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                )
            )
    
    image_bytes = base64.b64decode(base64_image)
    
    if contents and contents[-1].role == "user":
        contents.pop()
    
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=current_user_message)
            ]
        )
    )

    config = types.GenerateContentConfig(
        temperature=0.9,
        system_instruction=PROMPT
    )

    for attempt in range(5):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=contents,
                config=config
            )
            
            if response and response.text and response.text.strip():
                final_content = response.text.strip()
                updated_memory = append_history(memory, my_response=final_content)
                save_memory(chat_id, updated_memory)
                return final_content
                
        except Exception as e:
            print(f"❌ Ошибка Gemini Vision (попытка {attempt + 1}): {e}")
            
        if attempt < 4:
            await asyncio.sleep(5)
            
    raise RuntimeError("Gemini Vision не смог обработать изображение.")


async def priem_video(chat_id: int, current_user_message: str, base64_video: str, mime_type: str = "video/mp4", user_key: str = None) -> str:
    client = _get_gemini_client(user_key)
    memory = load_memory(chat_id)
    
    contents = []
    for item in memory.get("history", []):
        role = item.get("role")
        content = item.get("content")
        if not content and "parts" in item:
            content = item["parts"][0]["text"]
            
        if role in ["user", "assistant"] and content:
            gemini_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                )
            )
    
    video_bytes = base64.b64decode(base64_video)
    
    if contents and contents[-1].role == "user":
        contents.pop()
    
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
                types.Part.from_text(text=current_user_message)
            ]
        )
    )

    config = types.GenerateContentConfig(
        temperature=0.9,
        system_instruction=PROMPT
    )

    for attempt in range(5):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=contents,
                config=config
            )
            
            if response and response.text and response.text.strip():
                final_content = response.text.strip()
                updated_memory = append_history(memory, my_response=final_content)
                save_memory(chat_id, updated_memory)
                return final_content
                
        except Exception as e:
            print(f"❌ Ошибка Gemini Video (попытка {attempt + 1}): {e}")
            
        if attempt < 4:
            await asyncio.sleep(5)
            
    raise RuntimeError("Gemini не смог разобрать видео.")