import aiohttp
import asyncio
from variables import CHAT_TRIGGER_WORD

async def get_chat_models_for_telegram():
    url = "https://api.airforce/v1/models"

    translations = {
        "free": "бесплатно",
        "paid": "платно",
        "partial_outage": "частичный сбой",
        "major_outage": "серьёзный сбой",
        "degraded": "замедлено",
        "operational": "работает",  # добавили на случай, если status вернет operational
        "unstable": "нестабильно",
    }

    try:
        # Асинхронный запрос вместо requests
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=120) as response:
                if response.status != 200:
                    return f"Ошибка API: код {response.status}", []

                payload = await response.json()

        models = payload.get("data", [])

        telegram_lines = []
        raw_ids = []  # Сюда будем собирать чистые ID без оформления
        models = sorted(models, key=lambda x: x.get("id", "").lower())
        models.sort(key=lambda x: x.get("tier", ""))

        for model in models:
            if model.get("supports_chat") is True:
                model_id = model.get("id")

                # Добавляем чистый ID (в нижнем регистре для удобства сравнения)
                if model_id:
                    raw_ids.append(model_id.lower())

                tier = model.get("tier")
                status = model.get("status")

                translated_tier = translations.get(tier, tier)
                translated_status = translations.get(status, status)

                line = f"`{CHAT_TRIGGER_WORD} chat {model_id}`, {translated_tier}, {translated_status}"
                telegram_lines.append(line)

        final_message = "\n".join(telegram_lines)

        # Возвращаем СРАЗУ ДВА значения: строку для вывода и список ID
        return final_message, raw_ids

    except Exception as e:
        return f"Ошибка при получении моделей: {e}", []
        
async def get_music_models_for_telegram():
    url = "https://api.airforce/v1/models"

    translations = {
        "free": "бесплатно",
        "paid": "платно",
        "partial_outage": "частичный сбой",
        "major_outage": "серьёзный сбой",
        "degraded": "замедлено",
        "operational": "работает",  # добавили на случай, если status вернет operational
        "unstable": "нестабильно",
    }

    try:
        # Асинхронный запрос вместо requests
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=120) as response:
                if response.status != 200:
                    return f"Ошибка API: код {response.status}", []

                payload = await response.json()

        models = payload.get("data", [])

        telegram_lines = []
        raw_ids = []  # Сюда будем собирать чистые ID без оформления
        models = sorted(models, key=lambda x: x.get("id", "").lower())
        models.sort(key=lambda x: x.get("tier", ""))

        for model in models:
            if model.get("media_type") == "audio":
                model_id = model.get("id")

                # Добавляем чистый ID (в нижнем регистре для удобства сравнения)
                if model_id:
                    raw_ids.append(model_id.lower())

                tier = model.get("tier")
                status = model.get("status")

                translated_tier = translations.get(tier, tier)
                translated_status = translations.get(status, status)

                # Оформляем для вывода. Не ставим точку в начале,
                # так как вы её добавляете в хэндлере через `• {m}`
                line = f"`{CHAT_TRIGGER_WORD} music {model_id}`, {translated_tier}, {translated_status}"
                telegram_lines.append(line)

        final_message = "\n".join(telegram_lines)

        # Возвращаем СРАЗУ ДВА значения: строку для вывода и список ID
        return final_message, raw_ids

    except Exception as e:
        return f"Ошибка при получении моделей: {e}", []
        
async def get_vision_models_for_telegram():
    url = "https://api.airforce/v1/models"

    translations = {
        "free": "бесплатно",
        "paid": "платно",
        "partial_outage": "частичный сбой",
        "major_outage": "серьёзный сбой",
        "degraded": "замедлено",
        "operational": "работает",  # добавили на случай, если status вернет operational
        "unstable": "нестабильно",
    }

    try:
        # Асинхронный запрос вместо requests
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=120) as response:
                if response.status != 200:
                    return f"Ошибка API: код {response.status}", []

                payload = await response.json()

        models = payload.get("data", [])

        telegram_lines = []
        raw_ids = []  # Сюда будем собирать чистые ID без оформления
        models = sorted(models, key=lambda x: x.get("id", "").lower())
        models.sort(key=lambda x: x.get("tier", ""))

        for model in models:
            if model.get("supports_vision") is True:
                model_id = model.get("id")

                # Добавляем чистый ID (в нижнем регистре для удобства сравнения)
                if model_id:
                    raw_ids.append(model_id.lower())

                tier = model.get("tier")
                status = model.get("status")

                translated_tier = translations.get(tier, tier)
                translated_status = translations.get(status, status)

                # Оформляем для вывода. Не ставим точку в начале,
                # так как вы её добавляете в хэндлере через `• {m}`
                line = f"`{CHAT_TRIGGER_WORD} vision {model_id}`, {translated_tier}, {translated_status}"
                telegram_lines.append(line)

        final_message = "\n".join(telegram_lines)

        # Возвращаем СРАЗУ ДВА значения: строку для вывода и список ID
        return final_message, raw_ids

    except Exception as e:
        return f"Ошибка при получении моделей: {e}", []
        
async def get_image_models_for_telegram():
    url = "https://api.airforce/v1/models"

    translations = {
        "free": "бесплатно",
        "paid": "платно",
        "partial_outage": "частичный сбой",
        "major_outage": "серьёзный сбой",
        "degraded": "замедлено",
        "operational": "работает",  # добавили на случай, если status вернет operational
        "unstable": "нестабильно",
    }

    try:
        # Асинхронный запрос вместо requests
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=120) as response:
                if response.status != 200:
                    return f"Ошибка API: код {response.status}", []

                payload = await response.json()

        models = payload.get("data", [])

        telegram_lines = []
        raw_ids = []  # Сюда будем собирать чистые ID без оформления
        models = sorted(models, key=lambda x: x.get("id", "").lower())
        models.sort(key=lambda x: x.get("tier", ""))

        for model in models:
            if model.get("supports_images") is True and model.get("media_type") == 'image':
                model_id = model.get("id")

                # Добавляем чистый ID (в нижнем регистре для удобства сравнения)
                if model_id:
                    raw_ids.append(model_id.lower())

                tier = model.get("tier")
                status = model.get("status")

                translated_tier = translations.get(tier, tier)
                translated_status = translations.get(status, status)

                # Оформляем для вывода. Не ставим точку в начале,
                # так как вы её добавляете в хэндлере через `• {m}`
                line = f"`{CHAT_TRIGGER_WORD} image {model_id}`, {translated_tier}, {translated_status}"
                telegram_lines.append(line)

        final_message = "\n".join(telegram_lines)

        # Возвращаем СРАЗУ ДВА значения: строку для вывода и список ID
        return final_message, raw_ids

    except Exception as e:
        return f"Ошибка при получении моделей: {e}", []