import os
import asyncio
import prompt
import httpx
from google import genai
from google.genai import types
from google.genai import Client

from variables import API_KEY_GEMINI

client = genai.Client(
    api_key=API_KEY_GEMINI,
)

async def priem(text):
    for attempt in range(5):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview", # gemini-3-flash-preview
                contents=text,
                config=types.GenerateContentConfig(
                    temperature=0.9
                    )
                )
            # Проверяем на None и пустую строку
            if response and response.text and response.text.strip():
                print(f"Успешно на попытке {attempt + 1}")
                print(response.text)
                return response.text
            print(f"Попытка {attempt + 1}: Получен пустой ответ.")
            
        except Exception as e:
            # Если произошла сетевая ошибка или ошибка API
            print(f"Попытка {attempt + 1}: Ошибка при вызове API: {e}")
        if attempt < 4:
            print("Ждем 30 секунд перед следующей попыткой...")
            await asyncio.sleep(30)
            
    print("Не удалось получить ответ после 5 попыток.")
    return None