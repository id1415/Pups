"""
Помощники для надёжной отправки сообщений в Telegram.

Закрывает две проблемы:
1. Сетевые сбои (ClientConnectorError: Cannot connect to host api.telegram.org)
   -> делается несколько попыток отправки с паузой.
2. Ошибки Markdown-разметки ("Can't parse entities: Can't find end of the entity...")
   -> три ступени отправки:
      0) обычная отправка с лёгким ремонтом разметки (незакрытый блок ``` и т.п.);
      1) текст КАК ЕСТЬ и с ЯВНЫМ parse_mode=None — Телега не парсит его вообще;
      2) зачистка символов разметки + тоже parse_mode=None (последний рубеж).

ВАЖНО: в ступенях 1-2 parse_mode=None передаётся ЯВНО, а не «просто не передаётся».
У бота в variables.py стоит DefaultBotProperties(parse_mode=ParseMode.MARKDOWN):
если parse_mode не передать вовсе, aiogram подставит дефолт бота — и Телега
снова будет парсить Markdown. Явный None перекрывает дефолт, поэтому текст
доходит целиком: и [Verse 1], и _подчёркивания_, и бэктики из лирики.

Почему прежняя схема (escape/strip с дефолтным parse_mode) не работала:
- escape_markdown ставил "\\" перед спецсимволами, но LEGACY Markdown (в отличие
  от MarkdownV2) экранирование через "\\" НЕ поддерживает — сообщение ломалось так же;
- strip_markdown выкидывал * _ ` но НЕ скобки [ ] — а в лирике повсюду
  [Verse 1] / [Chorus] / [Instrumental]: незакрытая "[" для legacy Markdown —
  это незакрытая ссылка, отсюда "Can't find end of the entity starting at ...".

ПЛЮС ГЛАВНОЕ: в variables.py (DefaultBotProperties) теперь parse_mode=None —
глобального Markdown-парсинга больше нет вообще, поэтому ЛЮБАЯ отправка в проекте
(bot_dialogue.py, summary.py, message.reply без parse_mode и т.д.) больше не может
упасть с "can't parse entities". Форматирование возвращается только там, где оно
нужно: ступень 0 send_message_safe шлёт с ЯВНЫМ "Markdown" и при ошибке разметки
автоматически повторяет без неё.
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
    """Экранирует спецсимволы Markdown.

    ВНИМАНИЕ: в фолбэках больше НЕ используется — legacy Markdown не понимает
    экранирование через "\\" (это умеет только MarkdownV2), поэтому экранированный
    текст ломался так же, как исходный. Оставлена для совместимости.
    """
    return _MD_ENTITY_RE.sub(r'\\\1', text)


def strip_markdown(text: str) -> str:
    """Последний рубеж: выкидывает символы разметки из текста."""
    return text.replace("*", "").replace("_", "").replace("`", "")


async def _send_with_protection(attempt_send, retries: int, retry_delay: float):
    """
    Универсальный конверт отправки.

    attempt_send(stage: int) -> Message, где stage:
        0 — текст с ЯВНЫМ parse_mode="Markdown" (бывший дефолт бота),
        1 — текст как есть, parse_mode=None (Телега не парсит вообще),
        2 — текст без символов разметки, parse_mode=None.
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


def _fallback_kwargs(kwargs: dict, stage: int) -> dict:
    """Копия kwargs для отправки.

    Ступень 0 — parse_mode="Markdown" ЯВНО (глобального дефолта в variables.py
    больше нет: там parse_mode=None). Явный parse_mode вызывающего уважается.
    Ступени 1-2 — parse_mode=None ЯВНО: Телега вообще не парсит текст, сломаться
    нечему. Копия нужна, чтобы словарь вызывающего не мутировал между попытками.
    """
    extra = dict(kwargs)
    if stage == 0:
        extra.setdefault("parse_mode", "Markdown")
    else:
        extra["parse_mode"] = None
    return extra


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
            payload = text  # как есть — парсинга нет, ломаться нечему
        else:
            payload = strip_markdown(text) or "…"

        return await bot.send_message(
            chat_id=chat_id,
            text=payload,
            message_thread_id=message_thread_id,
            **_fallback_kwargs(kwargs, stage),
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
                cap = cap  # как есть, парсинг выключен ниже
            else:
                cap = strip_markdown(cap) or None

        return await bot.send_photo(
            chat_id=chat_id,
            photo=photo,
            caption=cap,
            message_thread_id=message_thread_id,
            **_fallback_kwargs(kwargs, stage),
        )

    return await _send_with_protection(_attempt, retries, retry_delay)
