# -*- coding: utf-8 -*-
"""
Telegram-бот с комбинированной памятью: короткая (диалог) + долгая (RAG по документам).
Интеграция с ProxyAPI через aiohttp (формат OpenAI Chat Completions).

Запуск: установите зависимости (requirements.txt), заполните .env, выполните: python bot.py
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import aiohttp
import faiss  # type: ignore
import httpx
import numpy as np
from aiohttp import ClientTimeout
from docx import Document as DocxDocument
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import Conflict
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Загрузка окружения и базовые настройки
# ─────────────────────────────────────────────────────────────────────────────
# Явный путь: при запуске из IDE cwd может быть не каталог проекта — токены не подхватывались.
_ENV_FILE = Path(__file__).resolve().parent / ".env"
# Значения из .env должны иметь приоритет над уже выставленными в ОС переменными
# (иначе после смены токена в файле продолжит использоваться старый BOT_TOKEN).
load_dotenv(_ENV_FILE, override=True)

# Токены и параметры API (можно переопределить через .env)
BOT_TOKEN = (
    os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or ""
).strip()


def _normalize_proxy_api_url(url: str) -> str:
    """
    Исправление частых опечаток в .env:
    • \"https.proxyapi.ru/…\" вместо \"https://…\";
    • \"https://proxyapi.ru\" вместо \"https://api.proxyapi.ru\".
    """
    u = url.strip().rstrip("/")
    if u.startswith("https.") and not u.startswith("https://"):
        u = "https://" + u[len("https.") :]
    if u.startswith("https://proxyapi.ru"):
        u = "https://api.proxyapi.ru" + u[len("https://proxyapi.ru") :]
    return u


PROXY_API_URL = _normalize_proxy_api_url(
    os.getenv("PROXY_API_URL", "https://api.proxyapi.ru/openai/v1")
)
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
try:
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS") or "2048")
except ValueError:
    DEFAULT_MAX_TOKENS = 2048

# Параметры короткой памяти и RAG
SHORT_MEMORY_MAX_MESSAGES = int(os.getenv("SHORT_MEMORY_MAX_MESSAGES", "10"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Ограничение размера файла (20 МБ по умолчанию)
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024

# Путь для сохранения индексов FAISS
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# Логирование
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_PATH = Path(os.getenv("LOG_FILE", "bot.log"))

# Таймаут HTTP к ProxyAPI (секунды)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))

# Таймауты HTTP к Telegram (python-telegram-bot использует httpx).
# При httpx.ConnectTimeout / telegram.error.TimedOut на get_me или polling —
# увеличьте TELEGRAM_CONNECT_TIMEOUT и проверьте доступ к api.telegram.org (VPN/прокси).
TELEGRAM_CONNECT_TIMEOUT = float(os.getenv("TELEGRAM_CONNECT_TIMEOUT", "60"))
TELEGRAM_READ_TIMEOUT = float(os.getenv("TELEGRAM_READ_TIMEOUT", "30"))
TELEGRAM_WRITE_TIMEOUT = float(os.getenv("TELEGRAM_WRITE_TIMEOUT", "60"))
TELEGRAM_POOL_TIMEOUT = float(os.getenv("TELEGRAM_POOL_TIMEOUT", "30"))
# Long polling должен держать read дольше, чем интервал сервера
TELEGRAM_GET_UPDATES_CONNECT_TIMEOUT = float(
    os.getenv("TELEGRAM_GET_UPDATES_CONNECT_TIMEOUT", "60")
)
TELEGRAM_GET_UPDATES_READ_TIMEOUT = float(
    os.getenv("TELEGRAM_GET_UPDATES_READ_TIMEOUT", "50")
)
TELEGRAM_GET_UPDATES_WRITE_TIMEOUT = float(
    os.getenv("TELEGRAM_GET_UPDATES_WRITE_TIMEOUT", "60")
)
TELEGRAM_GET_UPDATES_POOL_TIMEOUT = float(
    os.getenv("TELEGRAM_GET_UPDATES_POOL_TIMEOUT", "30")
)
try:
    TELEGRAM_BOOTSTRAP_RETRIES = int(os.getenv("TELEGRAM_BOOTSTRAP_RETRIES", "10"))
except ValueError:
    TELEGRAM_BOOTSTRAP_RETRIES = 10

# Свой экземпляр Bot API (если официальный api недоступен и поднят telegram-bot-api)
TELEGRAM_BASE_URL = os.getenv("TELEGRAM_BASE_URL", "").strip() or None


def esc(value: object) -> str:
    """Экранирование для Telegram HTML (&, <, >)."""
    return html.escape(str(value), quote=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Эмодзи и текстовые шаблоны (UX)
# ─────────────────────────────────────────────────────────────────────────────
E = type(
    "E",
    (),
    {
        "ok": "✅",
        "warn": "⚠️",
        "err": "❌",
        "think": "🤔",
        "book": "📚",
        "chat": "💬",
        "robot": "🤖",
        "hourglass": "⏳",
        "trash": "🗑️",
        "page": "📄",
        "stats": "📊",
        "pdf": "📕",
        "docx": "📘",
        "txt": "📝",
        "typing": "✍️",
        "gear": "⚙️",
    },
)


WELCOME_TEXT = textwrap.dedent(
    f"""
    {E.robot} Привет! Я бот с <b>комбинированной памятью</b>:
    • {E.chat} <b>Короткая память</b>: последние {SHORT_MEMORY_MAX_MESSAGES} реплик диалога.
    • {E.book} <b>Долгая память</b>: загрузите PDF / DOCX / TXT — я проиндексирую и буду отвечать с опорой на документ.

    <b>Команды</b>
    /help — справка
    /clear — очистить историю диалога
    /cleardoc — удалить документ и векторный индекс
    /stats — статистика памяти
    /model название_модели — сменить модель LLM для вас

    Отправьте текст или документ — отвечу с учётом памяти.
    """
).strip()

HELP_TEXT = textwrap.dedent(
    f"""
    {E.book} <b>Документы</b>: поддерживаются <code>.pdf</code>, <code>.docx</code>, <code>.txt</code> (до {MAX_FILE_SIZE_BYTES // (1024 * 1024)} МБ).
    После загрузки строится FAISS-индекс (эмбеддинги <code>{esc(EMBEDDING_MODEL_NAME)}</code>), чанки: {CHUNK_SIZE} символов, overlap {CHUNK_OVERLAP}.

    {E.chat} <b>Диалог</b>: в контекст LLM попадают последние {SHORT_MEMORY_MAX_MESSAGES} сообщений.

    <b>Команды</b>: /clear — очистить диалог; /cleardoc — удалить документ и индекс; /stats — статистика; /model — модель LLM.

    Приоритет: если документ есть — сначала подставляются <b>top-{RAG_TOP_K}</b> релевантных фрагмента, затем история.

    ProxyAPI URL: <code>{esc(PROXY_API_URL)}</code> (совместимо с OpenAI Chat Completions).
    """
).strip()


def setup_logging() -> None:
    """Настройка записи логов в файл и консоль."""
    handlers: List[logging.Handler] = []

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    handlers.append(fh)

    sh = logging.StreamHandler()
    sh.setLevel(LOG_LEVEL)
    handlers.append(sh)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


log = logging.getLogger("memory_bot")


# ─────────────────────────────────────────────────────────────────────────────
# 3) Короткая память (сессия пользователя, deque в RAM)
# ─────────────────────────────────────────────────────────────────────────────
class ShortTermMemory:
    """
    Хранит последние N сообщений на пользователя.
    Каждое сообщение — словарь в формате ролей OpenAI: role + content.
    """

    def __init__(self, max_messages: int = SHORT_MEMORY_MAX_MESSAGES) -> None:
        # user_id (int) -> deque сообщений
        self._store: Dict[int, Deque[Dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_messages)
        )
        self._max = max_messages

    def append(self, user_id: int, role: str, content: str) -> None:
        self._store[user_id].append({"role": role, "content": content})

    def get_messages(self, user_id: int) -> List[Dict[str, str]]:
        return list(self._store[user_id])

    def clear(self, user_id: int) -> None:
        self._store[user_id].clear()

    def count(self, user_id: int) -> int:
        return len(self._store[user_id])


# Глобальные структуры (один процесс бота)
short_memory = ShortTermMemory(SHORT_MEMORY_MAX_MESSAGES)
# Предпочитаемая модель LLM на пользователя (если не задана — DEFAULT_MODEL)
user_models: Dict[int, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# 4) Долгая память: извлечение текста, чанки, эмбеддинги, FAISS, диск
# ─────────────────────────────────────────────────────────────────────────────
_encoder: Optional[SentenceTransformer] = None
_encoder_lock = asyncio.Lock()


def _get_encoder() -> SentenceTransformer:
    """Ленивая загрузка модели эмбеддингов (один раз на процесс)."""
    global _encoder
    if _encoder is None:
        log.info("Загрузка модели эмбеддингов: %s", EMBEDDING_MODEL_NAME)
        _encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _encoder


async def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Асинхронная обёртка: encode в пуле потоков, чтобы не блокировать event loop.
    Возвращает float32 матрицу (n, dim).
    """

    def _sync_encode() -> np.ndarray:
        model = _get_encoder()
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # для косинусной близости через inner product
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    return await asyncio.to_thread(_sync_encode)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Разбиение текста на чанки по символам с перекрытием.
    Границы стараемся подвинуть к пробелам (мягкий перенос).
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    step = max(1, size - overlap)

    while start < n:
        end = min(start + size, n)
        piece = text[start:end]

        if end < n:
            # ищем последний пробел в окне, чтобы не рвать слова
            split_at = piece.rfind(" ")
            if split_at > size // 3:
                piece = piece[:split_at]
                end = start + split_at

        piece = piece.strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break
        start = max(end - overlap, start + step)

    return chunks


def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    parts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception as ex:  # noqa: BLE001 — защита от битых страниц
            log.warning("PDF: ошибка извлечения страницы: %s", ex)
            t = ""
        parts.append(t)
    return "\n".join(parts).strip()


def extract_text_from_docx(data: bytes) -> str:
    bio = BytesIO(data)
    doc = DocxDocument(bio)
    return "\n".join(p.text for p in doc.paragraphs if p.text).strip()


def extract_text_from_txt(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return data.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace").strip()


def extract_text_by_filename(filename: str, data: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if lower.endswith(".docx"):
        return extract_text_from_docx(data)
    if lower.endswith(".txt"):
        return extract_text_from_txt(data)
    raise ValueError(f"Неподдерживаемый формат файла: {filename}")


def emoji_for_filename(name: str) -> str:
    l = name.lower()
    if l.endswith(".pdf"):
        return E.pdf
    if l.endswith(".docx"):
        return E.docx
    if l.endswith(".txt"):
        return E.txt
    return E.page


@dataclass
class UserRagState:
    """Состояние RAG одного пользователя + пути сохранения на диск."""

    user_id: int
    chunks: List[str] = field(default_factory=list)
    index: Optional[Any] = None  # faiss.IndexFlatIP
    source_name: Optional[str] = None

    @property
    def user_dir(self) -> Path:
        return DATA_DIR / "users" / str(self.user_id)

    @property
    def index_path(self) -> Path:
        return self.user_dir / "vectors.faiss"

    @property
    def meta_path(self) -> Path:
        return self.user_dir / "meta.json"

    def has_index(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def clear_memory(self) -> None:
        self.chunks.clear()
        self.index = None
        self.source_name = None

    def save_to_disk(self) -> None:
        """Сохраняем индекс FAISS и метаданные (тексты чанков)."""
        if self.index is None:
            return
        self.user_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        meta = {
            "user_id": self.user_id,
            "source_name": self.source_name,
            "chunks": self.chunks,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_from_disk(self) -> bool:
        """Восстановление с диска. True если успешно."""
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        try:
            self.index = faiss.read_index(str(self.index_path))
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.chunks = list(meta.get("chunks", []))
            self.source_name = meta.get("source_name")
            return self.has_index()
        except Exception as ex:  # noqa: BLE001
            log.exception("Не удалось загрузить индекс user=%s: %s", self.user_id, ex)
            self.clear_memory()
            return False

    def delete_from_disk(self) -> None:
        for p in (self.index_path, self.meta_path):
            try:
                if p.exists():
                    p.unlink()
            except OSError as ex:
                log.warning("Не удалось удалить %s: %s", p, ex)
        try:
            if self.user_dir.exists() and not any(self.user_dir.iterdir()):
                self.user_dir.rmdir()
        except OSError:
            pass


class RagManager:
    """Пул RAG-состояний по user_id."""

    def __init__(self) -> None:
        self._by_user: Dict[int, UserRagState] = {}

    def get(self, user_id: int) -> UserRagState:
        if user_id not in self._by_user:
            self._by_user[user_id] = UserRagState(user_id=user_id)
        return self._by_user[user_id]

    async def try_autoload(self, user_id: int) -> None:
        state = self.get(user_id)
        if not state.has_index():
            await asyncio.to_thread(state.load_from_disk)


rag_manager = RagManager()


async def build_index_for_user(
    user_id: int,
    filename: str,
    raw_text: str,
    progress_callback: Optional[Any] = None,
) -> Tuple[int, int]:
    """
    Строим FAISS IndexFlatIP по нормализованным эмбеддингам (косинусная близость).
    progress_callback(done, total, stage_message) — для «прогресс-бара» в Telegram.
    """
    total_steps = 3
    if progress_callback:
        await progress_callback(1, total_steps, "Разбиение на чанки…")

    chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("После разбиения на чанки текст пуст — проверьте содержимое файла.")

    if progress_callback:
        await progress_callback(2, total_steps, "Генерация эмбеддингов…")

    dim = _get_encoder().get_sentence_embedding_dimension()
    embeddings = await encode_texts(chunks)

    if progress_callback:
        await progress_callback(3, total_steps, "Индекс FAISS и сохранение…")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    state = rag_manager.get(user_id)
    state.chunks = chunks
    state.index = index
    state.source_name = filename

    await asyncio.to_thread(state.save_to_disk)

    return len(chunks), dim


async def rag_search(user_id: int, query: str) -> List[str]:
    """Top-K наиболее релевантных чанков по запросу."""
    await rag_manager.try_autoload(user_id)
    state = rag_manager.get(user_id)
    if not state.has_index() or not query.strip():
        return []

    q = await encode_texts([query])
    scores, ids = await asyncio.to_thread(state.index.search, q, RAG_TOP_K)

    results: List[str] = []
    for idx in ids[0]:
        ii = int(idx)
        if 0 <= ii < len(state.chunks):
            results.append(state.chunks[ii])
    return results


async def clear_user_rag(user_id: int) -> None:
    state = rag_manager.get(user_id)
    state.clear_memory()
    await asyncio.to_thread(state.delete_from_disk)


def render_progress_bar(step: int, total: int, width: int = 12) -> str:
    """Простой текстовый прогресс-бар для сообщения Telegram."""
    total = max(1, total)
    step = max(0, min(step, total))
    filled = int(round(width * step / total))
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * step / total)
    return f"[{bar}] {pct}%"


# ─────────────────────────────────────────────────────────────────────────────
# 5) ProxyAPI — асинхронные запросы (OpenAI Chat Completions)
# ─────────────────────────────────────────────────────────────────────────────
def chat_completions_url() -> str:
    return f"{PROXY_API_URL}/chat/completions"


async def call_proxy_api(
    session: aiohttp.ClientSession,
    messages: List[Dict[str, str]],
    model: str,
) -> str:
    """
    Вызов chat/completions. Возвращает текст ответа ассистента.
    Ошибки пробрасываются наверх для локализованного текста пользователю.
    """
    if not PROXY_API_KEY:
        raise RuntimeError("Не задан PROXY_API_KEY в окружении.")

    headers = {
        "Authorization": f"Bearer {PROXY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }

    timeout = ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with session.post(
        chat_completions_url(), headers=headers, json=payload, timeout=timeout
    ) as resp:
        body = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")

        data = json.loads(body)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as ex:
            raise RuntimeError(f"Неожиданный ответ API: {body[:500]}") from ex


# ─────────────────────────────────────────────────────────────────────────────
# 6) Комбинированный промпт: RAG → история → текущий вопрос
# ─────────────────────────────────────────────────────────────────────────────
def build_system_prompt(rag_chunks: List[str]) -> str:
    """
    Системная инструкция + контекст из документа (если есть).
    Пользователь требовал: при наличии документа сначала релевантные фрагменты.
    """
    base = (
        "Ты полезный ассистент в Telegram. Отвечай ясно и по делу. "
        "Если в контексте есть выдержки из документа пользователя, опирайся на них; "
        "если информации в документе недостаточно, честно скажи об этом."
    )
    if not rag_chunks:
        return base

    joined = "\n\n---\n\n".join(
        f"Фрагмент {i+1}:\n{c}" for i, c in enumerate(rag_chunks)
    )
    return (
        f"{base}\n\n"
        f"Ниже — наиболее релевантные фрагменты из загруженного документа пользователя "
        f"(отсортированы по релевантности к текущему запросу):\n\n{joined}"
    )


def build_messages_for_llm(
    user_id: int,
    user_text: str,
    rag_chunks: List[str],
) -> List[Dict[str, str]]:
    """
    Итоговый список сообщений для Chat Completions:
    system (RAG + правила) + короткая история (без текущего сообщения) + текущий user.
    """
    system_content = build_system_prompt(rag_chunks)
    history = short_memory.get_messages(user_id)

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def user_model(user_id: int) -> str:
    return user_models.get(user_id, DEFAULT_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# 7) Обработчики Telegram
# ─────────────────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id if update.effective_user else 0
    log.info("/start от user_id=%s username=%s", uid, update.effective_user.username)
    await update.message.reply_text(WELCOME_TEXT, parse_mode=ParseMode.HTML)


async def log_every_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отладка: видно в логе, доходят ли обновления до этого процесса (важно при втором экземпляре бота)."""
    hint = ""
    if update.message and update.message.text is not None:
        hint = f" text={update.message.text[:100]!r}"
    elif update.message:
        hint = " message(без текстового поля / вложение)"
    log.info("Telegram update_id=%s%s", update.update_id, hint)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.info("/help от user_id=%s", update.effective_user.id)
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    short_memory.clear(uid)
    log.info("/clear: очищена короткая память user_id=%s", uid)
    await update.message.reply_text(f"{E.trash} История диалога очищена.")


async def cmd_cleardoc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    await clear_user_rag(uid)
    log.info("/cleardoc: удалён RAG user_id=%s", uid)
    await update.message.reply_text(
        f"{E.trash} Документ и векторный индекс удалены (включая файлы на диске)."
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    await rag_manager.try_autoload(uid)
    rag = rag_manager.get(uid)
    n_short = short_memory.count(uid)
    has_doc = rag.has_index()
    n_chunks = len(rag.chunks) if has_doc else 0
    src = rag.source_name or "—"
    model = user_model(uid)

    text = (
        f"{E.stats} <b>Статистика памяти</b>\n"
        f"• Сообщений в короткой памяти: <b>{n_short}</b> / {SHORT_MEMORY_MAX_MESSAGES}\n"
        f"• Документ в долгой памяти: <b>{'да' if has_doc else 'нет'}</b>\n"
    )
    if has_doc:
        text += (
            f"• Файл: <code>{esc(src)}</code>\n"
            f"• Чанков в индексе: <b>{n_chunks}</b>\n"
        )
    text += f"• Модель LLM: <code>{esc(model)}</code>\n"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    args = context.args or []
    if not args:
        await update.message.reply_text(
            f"{E.warn} Укажите модель: <code>/model gpt-4o-mini</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    name = " ".join(args).strip()
    if not name:
        await update.message.reply_text(f"{E.err} Пустое имя модели.")
        return
    user_models[uid] = name
    log.info("user_id=%s сменил модель на %s", uid, name)
    await update.message.reply_text(
        f"{E.gear} Модель установлена: <code>{esc(name)}</code>",
        parse_mode=ParseMode.HTML,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    uid = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        await update.message.reply_text(f"{E.warn} Сообщение пустое.")
        return

    log.info("Текст от user_id=%s: %s", uid, text[:200])

    await update.message.chat.send_action(ChatAction.TYPING)

    await rag_manager.try_autoload(uid)
    rag_chunks = await rag_search(uid, text)

    messages = build_messages_for_llm(uid, text, rag_chunks)
    session: aiohttp.ClientSession = context.application.bot_data["http_session"]

    try:
        reply = await call_proxy_api(session, messages, user_model(uid))
    except asyncio.TimeoutError:
        log.warning("Таймаут ProxyAPI user_id=%s", uid)
        await update.message.reply_text(
            f"{E.err} Таймаут запроса к API. Попробуйте позже или смените модель."
        )
        return
    except aiohttp.ClientError as ex:
        log.exception("Сетевая ошибка ProxyAPI: %s", ex)
        await update.message.reply_text(
            f"{E.err} Сетевая ошибка при обращении к API: <code>{esc(ex)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    except Exception as ex:  # noqa: BLE001
        log.exception("Ошибка ProxyAPI: %s", ex)
        await update.message.reply_text(
            f"{E.err} Ошибка API: {esc(str(ex)[:800])}",
            parse_mode=ParseMode.HTML,
        )
        return

    short_memory.append(uid, "user", text)
    short_memory.append(uid, "assistant", reply)

    # Разбиваем длинные ответы (лимит Telegram ~4096 символов)
    for chunk in split_telegram_message(reply):
        await update.message.reply_text(chunk)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return

    doc = update.message.document
    uid = update.effective_user.id
    fname = doc.file_name or "file.dat"

    log.info("Документ от user_id=%s: %s (%s байт)", uid, fname, doc.file_size)

    if doc.file_size is not None and doc.file_size > MAX_FILE_SIZE_BYTES:
        await update.message.reply_text(
            f"{E.err} Файл слишком большой: {doc.file_size} байт. "
            f"Максимум: {MAX_FILE_SIZE_BYTES} байт."
        )
        return

    lower = fname.lower()
    if not (lower.endswith(".pdf") or lower.endswith(".docx") or lower.endswith(".txt")):
        await update.message.reply_text(
            f"{E.warn} Поддерживаются только PDF, DOCX, TXT. Ваш файл: <code>{esc(fname)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    status = await update.message.reply_text(
        f"{emoji_for_filename(fname)} {E.hourglass} Загружаю <code>{esc(fname)}</code>…",
        parse_mode=ParseMode.HTML,
    )

    try:
        tg_file = await doc.get_file()
        file_bytes = await tg_file.download_as_bytearray()
    except Exception as ex:  # noqa: BLE001
        log.exception("Ошибка скачивания файла: %s", ex)
        await status.edit_text(f"{E.err} Не удалось скачать файл из Telegram.")
        return

    try:
        raw = extract_text_by_filename(fname, bytes(file_bytes))
    except Exception as ex:  # noqa: BLE001
        log.exception("Ошибка извлечения текста: %s", ex)
        await status.edit_text(f"{E.err} Не удалось прочитать документ: <code>{esc(ex)}</code>", parse_mode=ParseMode.HTML)
        return

    if not raw or len(raw.strip()) < 10:
        await status.edit_text(
            f"{E.warn} Документ пустой или текст слишком короткий — индексация отменена."
        )
        return

    total_steps = 3

    async def progress_cb(done: int, total: int, label: str) -> None:
        bar = render_progress_bar(done, total)
        try:
            await status.edit_text(
                f"{emoji_for_filename(fname)} {E.hourglass} <b>{esc(label)}</b>\n{bar}",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            # Telegram может запретить слишком частые edit — игнорируем мелкие сбои
            pass

    try:
        async with _encoder_lock:
            n_chunks, dim = await build_index_for_user(
                uid, fname, raw, progress_callback=progress_cb
            )
    except Exception as ex:  # noqa: BLE001
        log.exception("Ошибка индексации: %s", ex)
        await status.edit_text(
            f"{E.err} Ошибка индексации: <code>{esc(ex)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    user_dir = rag_manager.get(uid).user_dir
    await status.edit_text(
        f"{E.ok} Документ проиндексирован!\n"
        f"• Файл: <code>{esc(fname)}</code>\n"
        f"• Чанков: <b>{n_chunks}</b>, размерность вектора: <b>{dim}</b>\n"
        f"• Индекс сохранён на диск в <code>{esc(user_dir)}</code>",
        parse_mode=ParseMode.HTML,
    )
    log.info(
        "Индексация OK user_id=%s file=%s chunks=%s dim=%s", uid, fname, n_chunks, dim
    )


def split_telegram_message(text: str, limit: int = 4000) -> List[str]:
    """Грубое разбиение длинного ответа на части."""
    text = text or ""
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    start = 0
    while start < len(text):
        parts.append(text[start : start + limit])
        start += limit
    return parts


def _is_telegram_transport_failure(err: BaseException) -> bool:
    """
    Временные сбои связи до api.telegram.org: DNS (Win WSAHOST_NOT_FOUND 11001),
    httpx.ConnectError, таймаут соединения. PTB следующим long poll обычно восстанавливается.
    """
    cur: Optional[BaseException] = err
    seen: set[int] = set()
    for _ in range(12):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))

        if isinstance(cur, httpx.ConnectError):
            return True
        if isinstance(cur, httpx.TimeoutException):
            return True

        if isinstance(cur, OSError):
            # Windows: 11001 — getaddrinfo / имя узла не найдено
            if getattr(cur, "winerror", None) == 11001:
                return True
            if getattr(cur, "errno", None) == 11001:
                return True

        tname = type(cur).__name__
        if "ConnectError" in tname or "ConnectTimeout" in tname:
            return True

        cur = getattr(cur, "__cause__", None)

    return False


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if isinstance(err, Conflict):
        log.error(
            "Telegram 409 Conflict: уже идёт getUpdates этим же токеном (второй python, "
            "другой ПК или не снятый webhook). Остановите лишний процесс. Детали: %s",
            err,
        )
        return
    if _is_telegram_transport_failure(err):
        log.warning(
            "Краткий сбой связи с Telegram (%s). Обычно проходит само; если часто "
            "(Errno 11001) — проверьте интернет, DNS или VPN до api.telegram.org.",
            err,
        )
        return
    log.exception("Необработанное исключение в обработчике: %s", err)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            f"{E.err} Внутренняя ошибка бота. Администратор уведомлён логом."
        )


async def post_init(application: Application) -> None:
    """Создаём общий aiohttp ClientSession на всё приложение."""
    application.bot_data["http_session"] = aiohttp.ClientSession()
    log.info("HTTP session создана.")


async def post_shutdown(application: Application) -> None:
    session = application.bot_data.get("http_session")
    if session and not session.closed:
        await session.close()
        log.info("HTTP session закрыта.")


# ─────────────────────────────────────────────────────────────────────────────
# 8) Точка входа
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not BOT_TOKEN:
        raise RuntimeError(
            f"Укажите BOT_TOKEN в файле {_ENV_FILE} "
            "(или переменную окружения TELEGRAM_BOT_TOKEN)."
        )

    log.info(
        ".env файл: %s | bot id по токену: %s",
        _ENV_FILE,
        BOT_TOKEN.split(":", 1)[0] if ":" in BOT_TOKEN else "?",
    )
    log.info("Старт бота. DEFAULT_MODEL=%s PROXY=%s", DEFAULT_MODEL, PROXY_API_URL)
    if TELEGRAM_BASE_URL:
        log.info("Кастомный TELEGRAM_BASE_URL=%s", TELEGRAM_BASE_URL)
    log.info(
        "Polling: убедитесь, что нет второго запущенного процесса с тем же BOT_TOKEN "
        "(иначе getUpdates приходит сюда пустым, а ответы уходят из другого места)."
    )

    builder = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .connect_timeout(TELEGRAM_CONNECT_TIMEOUT)
        .read_timeout(TELEGRAM_READ_TIMEOUT)
        .write_timeout(TELEGRAM_WRITE_TIMEOUT)
        .pool_timeout(TELEGRAM_POOL_TIMEOUT)
        .get_updates_connect_timeout(TELEGRAM_GET_UPDATES_CONNECT_TIMEOUT)
        .get_updates_read_timeout(TELEGRAM_GET_UPDATES_READ_TIMEOUT)
        .get_updates_write_timeout(TELEGRAM_GET_UPDATES_WRITE_TIMEOUT)
        .get_updates_pool_timeout(TELEGRAM_GET_UPDATES_POOL_TIMEOUT)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
    )
    if TELEGRAM_BASE_URL is not None:
        builder = builder.base_url(TELEGRAM_BASE_URL)
    application = builder.build()

    application.add_error_handler(on_error)

    # Сначала логируем любой update (group=-1 обрабатывается раньше group=0)
    application.add_handler(TypeHandler(Update, log_every_update), group=-1)

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("clear", cmd_clear))
    application.add_handler(CommandHandler("cleardoc", cmd_cleardoc))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("model", cmd_model))

    application.add_handler(
        MessageHandler(filters.Document.ALL, handle_document)
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        bootstrap_retries=TELEGRAM_BOOTSTRAP_RETRIES,
    )


if __name__ == "__main__":
    main()
