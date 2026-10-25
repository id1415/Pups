"""
Помощники для надёжной отправки сообщений в Telegram.

Закрывает две проблемы:
1. Сетевые сбои (ClientConnectorError: Cannot connect to host api.telegram.org)
   -> делается несколько попыток отправки с паузой.
2. Ошибки Markdown-разметки ("Can't parse entities...")
   -> три ступени отправки:
      0) обычная отправка с лёгким ремонтом разметки (незакрытый блок ``` и т.п.);
      1) экранирование спецсимволов Markdown — текст отображается "как есть",
         без форматирования, но без потери символов;
      2) грубая зачистка символов разметки (последний рубеж).

parse_mode=ParseMode.MARKDOWN в variables.py (DefaultBotProperties) остаётся как был —
фолбэки применяются только к конкретной отправке, ничего глобально не меняется.
"""
import asyncio
import logging
import re

import aiohttp
from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.types import Message

logger = logging.getLogger(__name__)

# Сколько попыток отправки при сетевых сбоях и пауза между ними (в секундах)
SEND_RETRIES = 4
SEND_RETRY_DELAY = 3

# Спецсимволы Markdown, из-за которых чаще всего падает разметка
_MD_ENTITY_RE = re.compile(r'([_*\[\]\\`])')


def repair_markdown(text: str) -> str:
    """Мелкий ремонт разметки: закрывает незакрытый блок кода ```."""
    if isinstance(text, str) and text.count("```") % 2 == 1:
        text = text.rstrip() + "\n```"
    return text


def escape_markdown(text: str) -> str:
    """Экранирует спецсимволы Markdown — Telegram покажет текст 'как есть'."""
    return _MD_ENTITY_RE.sub(r'\\\1', text)


def strip_markdown(text: str) -> str:
    """Последний рубеж: выкидывает символы разметки из текста."""
    return text.replace("*", "").replace("_", "").replace("`", "")


async def _send_with_protection(attempt_send, retries: int, retry_delay: float):
    """
    Универсальный конверт отправки.

    attempt_send(stage: int) -> Message, где stage:
        0 — обычная отправка (разметка бота по умолчанию),
        1 — текст с экранированными спецсимволами,
        2 — текст без символов разметки.
    """
    stage = 0
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            return await attempt_send(stage)

        except TelegramRetryAfter as e:
            # Flood control: Telegram сам сказал, сколько ждать
            last_exc = e
            wait = float(getattr(e, "retry_after", 0) or 0) + retry_delay
            logger.warning(f"TG flood control: жду {wait:.0f} сек (попытка {attempt}/{retries}).")
            await asyncio.sleep(wait)

        except TelegramBadRequest as e:
            msg = str(e).lower()
            # 400: Can't parse entities и т.п. — пробуем следующий режим
            if ("parse" in msg or "entity" in msg) and stage < 2:
                logger.warning(f"Ошибка Markdown в сообщении ({e}) — отправляю без форматирования.")
                stage += 1
                continue
            raise  # проблема не в разметке (пустой/длинный текст и т.п.) — наверх

        except (TelegramNetworkError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            # Cannot connect to host api.telegram.org и прочие сетевые сбои
            last_exc = e
            logger.warning(
                f"Сетевая ошибка отправки в Telegram (попытка {attempt}/{retries}): {e!r}"
            )
            if attempt < retries:
                await asyncio.sleep(retry_delay)

    raise last_exc


async def send_message_safe(
    bot: Bot,
    chat_id: int,
    text: str,
    message_thread_id: int = None,
    retries: int = SEND_RETRIES,
    retry_delay: float = SEND_RETRY_DELAY,
    **kwargs,
) -> Message:
    """
    Отправка текстового сообщения с защитой от сетевых сбоев и ошибок Markdown.

    Остальные kwargs (reply_to_message_id и т.п.) передаются в bot.send_message как есть.
    """
    text = str(text)

    async def _attempt(stage: int) -> Message:
        if stage == 0:
            payload = repair_markdown(text)
        elif stage == 1:
            payload = escape_markdown(text)
        else:
            payload = strip_markdown(text) or "…"

        return await bot.send_message(
            chat_id=chat_id,
            text=payload,
            message_thread_id=message_thread_id,
            **kwargs,
        )

    return await _send_with_protection(_attempt, retries, retry_delay)


async def send_photo_safe(
    bot: Bot,
    chat_id: int,
    photo,
    caption: str = None,
    message_thread_id: int = None,
    retries: int = SEND_RETRIES,
    retry_delay: float = SEND_RETRY_DELAY,
    **kwargs,
) -> Message:
    """Отправка фото с теми же защитами, что и send_message_safe (применяются к подписи)."""

    async def _attempt(stage: int) -> Message:
        cap = caption
        if cap is not None:
            cap = str(cap)
            if stage == 0:
                cap = repair_markdown(cap)
            elif stage == 1:
                cap = escape_markdown(cap)
            else:
                cap = strip_markdown(cap) or None

        return await bot.send_photo(
            chat_id=chat_id,
            photo=photo,
            caption=cap,
            message_thread_id=message_thread_id,
            **kwargs,
        )

    return await _send_with_protection(_attempt, retries, retry_delay)
