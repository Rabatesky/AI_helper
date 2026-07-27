"""Подключение к SQLite и схема базы.

SQLite не требует установки: это не сервер, а библиотека внутри Python.
Вся база — один файл в data/, который создаётся при первом запуске.

Обращения синхронные, хотя бот асинхронный. Формально блокирующий вызов в
событийном цикле — грех, но запрос к локальному файлу при одном пользователе
занимает доли миллисекунды. Если когда-нибудь упрёмся, здесь появится aiosqlite,
и менять придётся только этот модуль.
"""

import logging
import sqlite3

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "assistant.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    chat_id        INTEGER NOT NULL,
    text           TEXT    NOT NULL,
    -- Время храним строкой ISO 8601 и всегда в UTC. Читаемо при отладке
    -- и сравнивается обычным «меньше/больше», потому что формат
    -- лексикографически упорядочен.
    fire_at        TEXT    NOT NULL,
    created_at     TEXT    NOT NULL,
    -- pending — ждёт срабатывания, sent — отправлено, cancelled — отменено
    status         TEXT    NOT NULL DEFAULT 'pending',
    -- Правило повторения в JSON или NULL для одноразовых напоминаний.
    -- У повторяющихся fire_at хранит ближайшее срабатывание и сдвигается
    -- вперёд после каждой отправки.
    repeat_rule    TEXT
);

-- Индекс под главный запрос планировщика: «что уже пора отправить».
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (status, fire_at);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Добавляет колонку, если её ещё нет.

    CREATE TABLE IF NOT EXISTS не трогает уже созданную таблицу, поэтому новые
    поля к существующей базе надо доводить отдельно. Способ примитивный, но при
    одном файле и одном пользователе полноценная система миграций избыточна.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in existing:
        return

    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    conn.commit()
    logger.info("Добавлена колонка %s.%s", table, column)


def connect(path=DB_PATH) -> sqlite3.Connection:
    """Открывает базу, создавая файл и схему при необходимости."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: соединение создаётся при старте, а используется
    # из задач событийного цикла, которые формально могут оказаться в другом
    # потоке. Одновременных запросов у нас нет, гонки не будет.
    conn = sqlite3.connect(path, check_same_thread=False)

    # Обращаться к колонкам по имени удобнее и надёжнее, чем по номеру.
    conn.row_factory = sqlite3.Row

    # WAL: чтение не блокирует запись. При одном пользователе избыточно,
    # но стоит одну строку и убирает целый класс проблем в будущем.
    conn.execute("PRAGMA journal_mode=WAL")

    conn.executescript(SCHEMA)
    conn.commit()

    # Доводим схему для баз, созданных предыдущими версиями бота.
    _ensure_column(conn, "reminders", "repeat_rule", "TEXT")

    logger.info("База готова: %s", path)
    return conn
