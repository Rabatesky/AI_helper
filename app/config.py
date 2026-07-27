"""Чтение и проверка конфигурации.

Все секреты и настройки живут в .env и читаются ровно в одном месте — здесь.
Модуль намеренно падает при импорте, если чего-то не хватает: лучше
получить внятную ошибку при старте, чем странное поведение в рантайме.
"""

import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

# Корень проекта = папка на уровень выше app/.
# Считаем путь от __file__, а не от текущей рабочей директории,
# чтобы бот одинаково запускался и из PyCharm, и из терминала, и из контейнера.
BASE_DIR = Path(__file__).resolve().parent.parent

# Папка для данных приложения (позже — файл SQLite с напоминаниями).
# В Docker её подключим как volume, чтобы данные переживали пересборку образа.
DATA_DIR = BASE_DIR / "data"

# load_dotenv кладёт содержимое .env в переменные окружения процесса.
# В Docker файла .env может не быть — переменные придут напрямую от compose,
# поэтому отсутствие файла не считаем ошибкой.
load_dotenv(BASE_DIR / ".env")


class ConfigError(RuntimeError):
    """Конфигурация неполная или некорректная."""


def _require(name: str) -> str:
    """Достаёт обязательную переменную окружения или падает с понятным текстом."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Не задана переменная {name}. "
            f"Скопируй .env.example в .env и заполни значения."
        )
    return value


def _int_option(name: str, default: int, minimum: int = 1) -> int:
    """Читает числовую настройку с значением по умолчанию и нижней границей."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должен быть целым числом, а получено: {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} не может быть меньше {minimum} (получено {value})")
    return value


def _parse_user_ids(raw: str) -> frozenset[int]:
    """Превращает строку '123, 456' в множество {123, 456}.

    Множество, а не список — проверка вхождения O(1) и дубликаты отсеиваются сами.
    """
    ids: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError as exc:
            raise ConfigError(
                f"ALLOWED_USER_IDS должен содержать числовые Telegram ID через запятую, "
                f"а получено: {chunk!r}"
            ) from exc

    if not ids:
        raise ConfigError(
            "ALLOWED_USER_IDS пуст — бот не отвечал бы вообще никому. "
            "Узнай свой ID у @userinfobot и впиши его в .env."
        )
    return frozenset(ids)


def _parse_timezone(name: str) -> ZoneInfo:
    """Проверяет, что часовой пояс существует, и возвращает готовый объект.

    Ошибку ловим здесь, а не при первом напоминании: неверный пояс должен
    ронять бота на старте, а не через неделю в момент срабатывания.
    """
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ConfigError(
            f"Неизвестный часовой пояс {name!r}. Нужен идентификатор IANA, "
            f"например Europe/Moscow или Asia/Yekaterinburg."
        ) from exc


# --- Telegram ----------------------------------------------------------------
BOT_TOKEN: str = _require("BOT_TOKEN")
ALLOWED_USER_IDS: frozenset[int] = _parse_user_ids(_require("ALLOWED_USER_IDS"))

# --- Время -------------------------------------------------------------------
# Часовой пояс пользователя. Внутри всё считаем в UTC, а к этому поясу приводим
# только на границах: при показе времени и при разборе фраз вроде «завтра в 9».
TIMEZONE_NAME: str = os.getenv("TZ", "UTC").strip() or "UTC"
TIMEZONE: ZoneInfo = _parse_timezone(TIMEZONE_NAME)

# --- LLM ---------------------------------------------------------------------
LLM_API_KEY: str = _require("GROQ_API_KEY")

# Groq говорит на протоколе OpenAI, поэтому клиент один и тот же —
# отличается только адрес. Смена провайдера = смена этих двух строк.
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b").strip()

# Разброс ответов: 0 — предсказуемо и сухо, 1 — творчески и непредсказуемо.
# Для ассистента нужна умеренность, особенно когда добавим вызов инструментов.
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))

# Сколько ждать ответа модели, прежде чем сдаться и извиниться перед пользователем.
LLM_TIMEOUT_SECONDS: int = _int_option("LLM_TIMEOUT_SECONDS", default=60)

# --- Распознавание речи ------------------------------------------------------
# Whisper доступен на том же ключе, что и чат. Вариант turbo быстрее и дешевле
# по лимитам, обычный whisper-large-v3 чуть точнее на сложном аудио.
STT_MODEL: str = os.getenv("STT_MODEL", "whisper-large-v3-turbo").strip()

# Язык голосовых. Явное указание заметно повышает точность; пустое значение
# включает автоопределение — на случай, если говоришь на нескольких языках.
STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "ru").strip()

# Предел длительности голосового. Защита от случайной отправки часовой записи:
# такой файл долго качается, долго распознаётся и съедает суточный лимит.
MAX_VOICE_SECONDS: int = _int_option("MAX_VOICE_SECONDS", default=300)

# --- Напоминания ---------------------------------------------------------------
# Как часто планировщик спрашивает у базы «что уже пора отправить».
# Точность напоминаний равна этому интервалу — секунды здесь не важны.
REMINDER_TICK_SECONDS: int = _int_option("REMINDER_TICK_SECONDS", default=30)

# Если бот лежал и напоминание просрочено, его всё равно отправят при запуске —
# но только если опоздание меньше этого срока. Прошлогоднее «позвонить маме»
# уже не помогает, а только сбивает с толку.
MISSED_REMINDER_GRACE_DAYS: int = _int_option("MISSED_REMINDER_GRACE_DAYS", default=7)

# Сколько раз подряд модель может попросить вызвать инструмент в рамках одного
# сообщения. Ограничение страхует от зацикливания: «посмотрел список -> отменил
# -> снова посмотрел -> ...».
MAX_TOOL_ROUNDS: int = _int_option("MAX_TOOL_ROUNDS", default=5)

# --- Диалог ------------------------------------------------------------------
# Через сколько минут молчания разговор считается законченным и начинается новый.
SESSION_TTL_MINUTES: int = _int_option("SESSION_TTL_MINUTES", default=120)

# Сколько последних реплик отправлять модели. Ограничение нужно, потому что
# контекст модели конечен, а длинная история замедляет и удорожает каждый запрос.
MAX_HISTORY_MESSAGES: int = _int_option("MAX_HISTORY_MESSAGES", default=20, minimum=2)
