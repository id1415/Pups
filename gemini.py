# gemini-3.1-flash-lite-preview
import io
from pydub import AudioSegment
import os
import base64
import asyncio
import time
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

# ===================== НАСТРОЙКИ ДЛЯ ВИДЕО =====================
# Таймаут надо менять ЗДЕСЬ (в клиенте Gemini), а не в handle_video_vision:
# таймаут в handle_video_vision касается только скачивания видео из Telegram,
# а TimeoutError прилетал из aiohttp-клиента google-genai, у которого
# был лимит 300 сек на ОДИН HTTP-запрос к API (через WARP не хватало).
VIDEO_TIMEOUT_MS = 600_000                    # 600 сек = 10 минут на каждый HTTP-запрос с видео
VIDEO_FILES_API_MIN_BYTES = 8 * 1024 * 1024   # видео больше 8 МБ отправляем через Files API
VIDEO_INLINE_MAX_BYTES = 18 * 1024 * 1024     # у inline-запроса лимит ~20 МБ, больше 18 МБ туда не влезет
VIDEO_PROCESSING_TIMEOUT_SEC = 180            # сколько ждём, пока Google обработает видео после загрузки


def _get_gemini_client(user_key: str = None, timeout_ms: int = 300_000) -> genai.Client:
    """Возвращает клиент Gemini с пользовательским или системным ключом.

    timeout_ms — таймаут КАЖДОГО HTTP-запроса к API в миллисекундах
    (по умолчанию 300 сек = 5 минут; для видео передаётся 600 сек).
    """
    active_key = user_key if user_key else API_KEY_GEMINI
    http_options = types.HttpOptions(
        timeout=timeout_ms
    )

    return genai.Client(
        api_key=active_key,
        http_options=http_options
    )
    
async def priem_summary(prompt_text: str, user_key: str = None) -> str:
    client = _get_gemini_client(user_key)
    
    config = types.GenerateContentConfig(
        temperature=0.9,
    )
    
    last_error = None
    for attempt in range(5):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt_text,
                config=config
            )
            
            if response and response.text and response.text.strip():
                return response.text.strip()
            else:
                last_error = RuntimeError("Gemini вернул пустой ответ")
                print(f"⚠️ Gemini Summary (попытка {attempt + 1}): пустой ответ")
                
        except Exception as e:
            last_error = e
            print(f"❌ Ошибка Gemini Summary (попытка {attempt + 1}): {type(e).__name__}: {e}")
            
        if attempt < 4:
            await asyncio.sleep(5)
            
    raise RuntimeError(
        f"Не удалось сгенерировать саммари после 5 попыток. "
        f"Последняя ошибка — {type(last_error).__name__}: {last_error}"
    )

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

    # --- ФИКС 400 INVALID_ARGUMENT "Requests ending with a model turn are not supported" ---
    # Gemini API требует, чтобы запрос заканчивался репликой ПОЛЬЗОВАТЕЛЯ (role="user").
    # В авто-режиме current_user_message пустой, а история из chat_memories часто
    # заканчивается ответом бота (role="model") — на этом API падает с 400.
    # Решение: добавляем служебную реплику пользователя с просьбой продолжить диалог.
    if not contents:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=current_user_message.strip() or
                    "Напиши любое сообщение в этот чат от своего имени: "
                    "поделись мыслью, идеей или темой для обсуждения."
                )]
            )
        )
    elif contents[-1].role == "model":
        continuation_prompt = current_user_message.strip() or (
            "Твоя очередь написать сообщение в этот чат от своего имени: "
            "поделись мыслью, идеей или темой для обсуждения. "
            "Не отвечай на это служебное сообщение и не упоминай его."
        )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=continuation_prompt)]
            )
        )
    # ----------------------------------------------------------------------------------------

    last_error = None
    for attempt in range(5):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=contents,
                config=config
            )
            
            if response and response.text and response.text.strip():
                final_content = response.text.strip()
                updated_memory = append_history(memory, my_response=final_content)
                save_memory(chat_id, updated_memory)
                print(final_content)
                return final_content
            else:
                last_error = RuntimeError("Gemini вернул пустой ответ")
                print(f"⚠️ Gemini API (попытка {attempt + 1}): пустой ответ")
                
        except Exception as e:
            last_error = e
            print(f"❌ Ошибка Gemini API (попытка {attempt + 1}): {type(e).__name__}: {e}")
            
        if attempt < 4:
            await asyncio.sleep(5)
            
    # Пробрасываем НАСТОЯЩУЮ причину, а не общую фразу — так в логе авто-режима
    # видно, что именно сломалось (429, 400, таймаут, ключ и т.д.)
    raise RuntimeError(
        f"Не удалось получить ответ от Gemini после 5 попыток. "
        f"Последняя ошибка — {type(last_error).__name__}: {last_error}"
    )


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

    last_error = None
    for attempt in range(5):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=contents,
                config=config
            )
            
            if response and response.text and response.text.strip():
                final_content = response.text.strip()
                updated_memory = append_history(memory, my_response=final_content)
                save_memory(chat_id, updated_memory)
                return final_content
            else:
                last_error = RuntimeError("Gemini вернул пустой ответ")
                print(f"⚠️ Gemini Vision (попытка {attempt + 1}): пустой ответ")
                
        except Exception as e:
            last_error = e
            print(f"❌ Ошибка Gemini Vision (попытка {attempt + 1}): {type(e).__name__}: {e}")
            
        if attempt < 4:
            await asyncio.sleep(5)
            
    raise RuntimeError(
        f"Gemini Vision не смог обработать изображение после 5 попыток. "
        f"Последняя ошибка — {type(last_error).__name__}: {last_error}"
    )


# ---------- Хелперы для работы с видео через Files API ----------

async def _upload_video_and_wait(client: genai.Client, video_bytes: bytes, mime_type: str):
    """Загружает видео в Gemini Files API и ждёт, пока оно обработается (state=ACTIVE).

    Files API передаёт файл в бинарном виде (без +33% раздувания от base64)
    и позволяет дальше слать ЛЁГКИЕ запросы со ссылкой на файл, вместо того
    чтобы тащить ~17 МБ base64 в каждом запросе (что через WARP и вызывало
    TimeoutError: aiohttp не успевал отправить тело запроса за отведённый таймаут).
    """
    uploaded = await client.aio.files.upload(
        file=io.BytesIO(video_bytes),
        config=types.UploadFileConfig(mime_type=mime_type)
    )

    if not uploaded or not getattr(uploaded, "name", None):
        raise RuntimeError("Files API не вернул имя загруженного файла.")

    current = uploaded
    state = getattr(current, "state", None)
    deadline = time.monotonic() + VIDEO_PROCESSING_TIMEOUT_SEC

    # Ждём, пока Google обработает (транскодирует) видео
    while state == types.FileState.PROCESSING:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Видео обрабатывается в Files API дольше {VIDEO_PROCESSING_TIMEOUT_SEC} сек."
            )
        await asyncio.sleep(3)
        current = await client.aio.files.get(name=uploaded.name)
        state = getattr(current, "state", None)

    if state == types.FileState.FAILED:
        file_error = getattr(current, "error", None)
        raise RuntimeError(f"Files API не смог обработать видео: {file_error or 'причина неизвестна'}")

    if not getattr(current, "uri", None):
        raise RuntimeError("Files API не вернул uri файла после обработки.")

    return current


async def _delete_uploaded_file(client: genai.Client, uploaded_file) -> None:
    """Удаляет файл из Files API (best effort — ошибки удаления не критичны)."""
    try:
        await client.aio.files.delete(name=uploaded_file.name)
    except Exception as e:
        print(f"⚠️ Не удалось удалить файл {getattr(uploaded_file, 'name', '?')} из Files API: {type(e).__name__}: {e}")


async def priem_video(chat_id: int, current_user_message: str, video_bytes: bytes, mime_type: str = "video/mp4", user_key: str = None) -> str:
    # Для видео даём 600 сек на запрос: ролики по 10-20 МБ через WARP грузятся
    # медленно, и старый таймаут 300 сек падал по TimeoutError ещё на отправке.
    client = _get_gemini_client(user_key, timeout_ms=VIDEO_TIMEOUT_MS)
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

    if contents and contents[-1].role == "user":
        contents.pop()

    config = types.GenerateContentConfig(
        temperature=0.9,
        system_instruction=PROMPT
    )

    model_id = "gemini-3.1-flash-lite-preview"
    last_error = None
    uploaded_file = None
    use_files_api = len(video_bytes) > VIDEO_FILES_API_MIN_BYTES

    def _push_user_video_part(parts):
        # Убираем предыдущую user-реплику (если есть) и ставим свежую с видео
        if contents and contents[-1].role == "user":
            contents.pop()
        contents.append(types.Content(role="user", parts=parts))

    # ---------------- ПУТЬ 1: Files API (для видео > 8 МБ) ----------------
    if use_files_api:
        for upload_attempt in range(3):
            try:
                size_mb = len(video_bytes) / 1024 / 1024
                print(f"📤 Загрузка видео ({size_mb:.1f} МБ) в Files API (попытка {upload_attempt + 1}/3)...")
                uploaded_file = await _upload_video_and_wait(client, video_bytes, mime_type)
                print(f"✅ Видео загружено и обработано: {uploaded_file.name}")
                break
            except Exception as e:
                last_error = e
                uploaded_file = None
                print(f"❌ Ошибка загрузки видео в Files API (попытка {upload_attempt + 1}): {type(e).__name__}: {e}")
                if upload_attempt < 2:
                    await asyncio.sleep(5)

    if uploaded_file is not None:
        # -------- Генерация по ссылке на файл: в запросе больше НЕТ мегабайт данных,
        # поэтому повторные попытки НЕ перекачивают видео заново --------
        try:
            for attempt in range(5):
                try:
                    _push_user_video_part([
                        types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=mime_type),
                        types.Part.from_text(text=current_user_message)
                    ])
                    response = await client.aio.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=config
                    )

                    if response and response.text and response.text.strip():
                        final_content = response.text.strip()
                        updated_memory = append_history(memory, my_response=final_content)
                        save_memory(chat_id, updated_memory)
                        return final_content
                    else:
                        last_error = RuntimeError("Gemini вернул пустой ответ")
                        print(f"⚠️ Gemini Video (попытка {attempt + 1}): пустой ответ")

                except Exception as e:
                    last_error = e
                    print(f"❌ Ошибка Gemini Video (попытка {attempt + 1}): {type(e).__name__}: {repr(e)}")

                if attempt < 4:
                    await asyncio.sleep(5)
        finally:
            await _delete_uploaded_file(client, uploaded_file)

        if len(video_bytes) > VIDEO_INLINE_MAX_BYTES:
            # Для такого размера inline-запрос невозможен — отдаём ошибку как есть
            raise RuntimeError(
                f"Gemini не смог разобрать видео после 5 попыток. "
                f"Последняя ошибка — {type(last_error).__name__}: {last_error}"
            )
        print("⚠️ Путь через Files API не удался, пробуем запасной вариант: видео прямо в запросе (inline).")

    elif use_files_api and len(video_bytes) > VIDEO_INLINE_MAX_BYTES:
        # Files API так и не заработал, а inline для большого видео невозможен
        raise RuntimeError(
            f"Не удалось загрузить видео в Files API после 3 попыток. "
            f"Последняя ошибка — {type(last_error).__name__}: {last_error}"
        )
    elif use_files_api:
        print("⚠️ Files API недоступен, пробуем запасной вариант: видео прямо в запросе (inline).")

    # ---------------- ПУТЬ 2 (запасной): inline-видео прямо в запросе ----------------
    for attempt in range(5):
        try:
            _push_user_video_part([
                types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
                types.Part.from_text(text=current_user_message)
            ])
            response = await client.aio.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )

            if response and response.text and response.text.strip():
                final_content = response.text.strip()
                updated_memory = append_history(memory, my_response=final_content)
                save_memory(chat_id, updated_memory)
                return final_content
            else:
                last_error = RuntimeError("Gemini вернул пустой ответ")
                print(f"⚠️ Gemini Video (попытка {attempt + 1}): пустой ответ")

        except Exception as e:
            last_error = e
            print(f"❌ Ошибка Gemini Video (попытка {attempt + 1}): {type(e).__name__}: {repr(e)}")

        if attempt < 4:
            await asyncio.sleep(5)

    raise RuntimeError(
        f"Gemini не смог разобрать видео после 5 попыток. "
        f"Последняя ошибка — {type(last_error).__name__}: {last_error}"
    )
