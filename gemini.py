# gemini-3.1-flash-lite-preview
import io
from pydub import AudioSegment
import os
import base64
import asyncio
import prompt
from google import genai
from google.genai import types
from google.genai import Client
from variables import API_KEY_GEMINI, PROMPT
from utils import load_memory, save_memory, append_history
import speech_recognition as sr

MODEL_ID = "gemini-2.5-flash-native-audio-latest"
CHUNK_SIZE = 1024

def _format_history_context(history: list, max_items: int = 100) -> str:
    """Форматирует последние сообщения из памяти для контекста модели."""
    if not history:
        return ""
    
    # Берем последние max_items сообщений
    recent_history = history[-max_items:]
    lines = []
    for item in recent_history:
        role = "Пользователь" if item.get("role") == "user" else "Ты (ассистент)"
        content = item.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
            
    return "\n".join(lines)

def _recognize_speech_pydub(audio_seg: AudioSegment) -> str:
    """Приводит аудио к 16kHz WAV и распознает речь через SpeechRecognition."""
    try:
        audio_16k = audio_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        wav_io = io.BytesIO()
        audio_16k.export(wav_io, format="wav")
        wav_io.seek(0)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="ru-RU")
            return text.strip()
    except sr.UnknownValueError:
        print("⚠️ SpeechRecognition не смог разобрать речь в ответе.")
    except Exception as e:
        print(f"⚠️ Ошибка SpeechRecognition: {e}")
    return ""

async def process_voice_turn(
    ogg_bytes: bytes, 
    system_prompt: str, 
    api_key: str, 
    voice_name: str = "Puck",
    history: list = None,
    timeout: float = 120.0
) -> tuple[io.BytesIO, str]:
    if not ogg_bytes:
        raise ValueError("Входной файл пуст.")

    # 1. Формируем расширенный системный промт с историей переписки
    full_prompt = system_prompt
    history_text = _format_history_context(history) if history else ""
    if history_text:
        full_prompt += f"\n\n--- ИСТОРИЯ ПОСЛЕДНЕГО ДИАЛОГА В ЧАТЕ ---\n{history_text}\n--- КОНЕЦ ИСТОРИИ (учитывай её при ответе на текущее голосовое) ---"

    # 2. Конвертируем входной OGG в PCM 16000Hz Mono 16-bit
    input_audio = AudioSegment.from_file(io.BytesIO(ogg_bytes), format="ogg")
    input_audio = input_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    pcm_input = input_audio.raw_data

    client = genai.Client(
        api_key=api_key,
        http_options={'api_version': 'v1alpha'}
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=full_prompt)]
        ),
    )

    audio_chunks = []
    turn_completed_event = asyncio.Event()

    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
        
        async def receiver():
            try:
                async for response in session.receive():
                    server_content = response.server_content
                    if server_content is not None:
                        if server_content.model_turn:
                            for part in server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    audio_chunks.append(part.inline_data.data)
                        
                        if server_content.turn_complete:
                            turn_completed_event.set()
                            break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Ошибка в Live receiver: {e}")
                turn_completed_event.set()

        async def sender():
            for i in range(0, len(pcm_input), CHUNK_SIZE):
                chunk = pcm_input[i:i + CHUNK_SIZE]
                await session.send(input={"data": chunk, "mime_type": "audio/pcm"}, end_of_turn=False)
                await asyncio.sleep(0.01)

            silence_chunk = b'\x00' * CHUNK_SIZE
            for _ in range(48):
                await session.send(input={"data": silence_chunk, "mime_type": "audio/pcm"}, end_of_turn=False)
                await asyncio.sleep(0.02)

        receiver_task = asyncio.create_task(receiver())
        sender_task = asyncio.create_task(sender())

        try:
            await asyncio.wait_for(turn_completed_event.wait(), timeout=timeout)
        finally:
            receiver_task.cancel()
            sender_task.cancel()

    if not audio_chunks:
        raise RuntimeError("Модель завершила ответ без аудио.")

    raw_audio_bytes = b"".join(audio_chunks)
    output_audio = AudioSegment(
        data=raw_audio_bytes,
        sample_width=2,
        frame_rate=24000,
        channels=1
    )

    out_io = io.BytesIO()
    output_audio.export(out_io, format="ogg", codec="libopus")
    out_io.seek(0)

    try:
        response_text = await asyncio.to_thread(_recognize_speech_pydub, output_audio)
    except Exception as e:
        print(e)
    if not response_text:
        response_text = "🎙️ [Голосовой ответ]"

    return out_io, str(response_text)

client = genai.Client(
    api_key=API_KEY_GEMINI
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
                model="gemini-3.5-flash-lite",
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

async def priem(chat_id: int, current_user_message: str, user_key: str = None, system_prompt: str = PROMPT) -> str:
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
        system_instruction=system_prompt  # <-- ИЗМЕНЕНО: было жёстко PROMPT
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


async def priem_vision(chat_id: int, current_user_message: str, base64_image: str, user_key: str = None, system_prompt: str = PROMPT) -> str:
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
    
    #image_bytes = base64.b64decode(base64_image)
    images = base64_image if isinstance(base64_image, list) else [base64_image]
    
    if contents and contents[-1].role == "user":
        contents.pop()
    
    parts = []
    for img in images:
        parts.append(types.Part.from_bytes(data=base64.b64decode(img), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=current_user_message))

    contents.append(types.Content(role="user", parts=parts))

    config = types.GenerateContentConfig(
        temperature=0.9,
        system_instruction=system_prompt
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