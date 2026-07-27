"""Точка входа бота.

Шаг 3: бот ведёт диалог с языковой моделью, помнит контекст разговора и
понимает голосовые сообщения. Голосовое распознаётся в текст и дальше идёт
ровно по тому же пути, что и набранное руками.
"""

import asyncio
import logging
from collections import defaultdict
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.access import WhitelistMiddleware
from app.agent import respond
from app.api_errors import ProviderError
from app.config import (
    ALLOWED_USER_IDS,
    BOT_TOKEN,
    LLM_MODEL,
    MAX_HISTORY_MESSAGES,
    MAX_VOICE_SECONDS,
    REMINDER_TICK_SECONDS,
    SESSION_TTL_MINUTES,
    STT_MODEL,
)
from app.db import connect
from app.history import HistoryStore
from app.llm import LLMClient
from app.reminders import ReminderScheduler, ReminderStore
from app.stt import SpeechToText
from app.tools import ToolBox, ToolContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Telegram не принимает сообщения длиннее 4096 символов. Режем с запасом,
# чтобы поместилась служебная пометка о новом диалоге.
TELEGRAM_LIMIT = 4000

# Пометка, которой бот сообщает, что прежний разговор истёк по времени.
# Без неё пользователь не поймёт, почему бот вдруг «забыл» предыдущий вопрос.
NEW_SESSION_NOTICE = "— новый диалог —"

# Dispatcher — маршрутизатор: получает апдейты и решает, какой обработчик вызвать.
dp = Dispatcher()

# Вешаем проверку доступа на уровень апдейта. outer_middleware срабатывает
# ДО подбора обработчика, то есть чужие сообщения отсекаются самыми первыми.
dp.update.outer_middleware(WhitelistMiddleware(ALLOWED_USER_IDS))

# По замку на пользователя. aiogram обрабатывает апдейты параллельно, и без
# замка два быстро отправленных сообщения полезли бы в одну историю
# одновременно — реплики перемешались бы, а модель получила бы кашу.
_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Режет длинный ответ на части, влезающие в одно сообщение Telegram.

    Стараемся рвать по границе абзаца, а если абзац сам длиннее лимита —
    по границе строки. Это лучше, чем резать посреди слова.
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""
    for block in text.split("\n"):
        # +1 — символ перевода строки, который вернётся при склейке.
        if len(current) + len(block) + 1 > limit:
            if current:
                parts.append(current)
                current = ""
            # Абзац не помещается целиком — режем его механически.
            while len(block) > limit:
                parts.append(block[:limit])
                block = block[limit:]
        current = f"{current}\n{block}" if current else block

    if current:
        parts.append(current)
    return parts


async def send_long(message: Message, text: str) -> None:
    """Отправляет ответ, при необходимости разбив на несколько сообщений."""
    for part in split_text(text):
        await message.answer(part)


@dp.message(Command("start"))
async def handle_start(message: Message) -> None:
    """Приветствие: что за бот, что умеет сейчас и куда развивается.

    Текст без Markdown-разметки: сообщения отправляются как есть, и звёздочки
    с решётками отобразились бы буквально.
    """
    await message.answer(
        "Привет! Я твой личный ассистент.\n"
        "Живу в Telegram, отвечаю только тебе — посторонние для меня не существуют.\n"
        "\n"
        "ЧТО УМЕЮ СЕЙЧАС\n"
        "\n"
        "💬 Разговор\n"
        "Отвечаю на вопросы и помню, о чём мы говорили. "
        f"Если молчишь дольше {SESSION_TTL_MINUTES} минут, начинаю новый диалог — "
        "чтобы утренний разговор не мешался с вечерним.\n"
        "\n"
        "🎤 Голосовые\n"
        "Наговори сообщение — распознаю речь и отвечу. Сначала покажу, что именно "
        "услышал, чтобы ты сразу увидел, если я ошибся.\n"
        "\n"
        "⏰ Напоминания\n"
        "Разовые: «напомни завтра в 9 позвонить маме», "
        "«не забыть бы в пятницу забрать посылку».\n"
        "Повторяющиеся: «каждый день в 8 выпить таблетки», «по будням в 7:30 про "
        "зарядку», «каждые полчаса размять глаза», «первого числа оплатить интернет».\n"
        "Отдельной команды не нужно — просто скажи словами. Список покажу, любое отменю.\n"
        "\n"
        "КОМАНДЫ\n"
        "/new — начать разговор с чистого листа\n"
        "/reminders — список активных напоминаний\n"
        "/status — на какой модели работаю и что помню\n"
        "\n"
        "ЧТО ДАЛЬШЕ\n"
        "Меня продолжают развивать. В планах: долгая память о фактах (чтобы не "
        "напоминать каждый раз, кто ты и что любишь), поиск в интернете, поиск мест "
        "поблизости, сводки из внешних сервисов и утренний дайджест.\n"
        "\n"
        "Если чего-то не хватает — скажи, это лучший способ определить, "
        "что делать следующим."
    )


@dp.message(Command("new"))
async def handle_new(message: Message, history: HistoryStore) -> None:
    """Принудительный сброс контекста.

    Нужен, когда тема сменилась, а таймаут ещё не истёк: писать про SQL
    в контексте разговора про историю Москвы модели только мешает.
    """
    had_content = history.reset(message.from_user.id)
    await message.answer(
        "Начали новый диалог, прошлый контекст забыт."
        if had_content
        else "Контекст и так пуст — можно просто писать."
    )


@dp.message(Command("status"))
async def handle_status(
    message: Message,
    history: HistoryStore,
    reminders: ReminderStore,
) -> None:
    """Диагностика: видно, что бот помнит и какой моделью отвечает."""
    session, _ = history.get_or_start(message.from_user.id)
    pending = reminders.list_pending(message.from_user.id)
    await message.answer(
        f"Модель: {LLM_MODEL}\n"
        f"Распознавание речи: {STT_MODEL}\n"
        f"Реплик в текущем диалоге: {len(session.messages)}\n"
        f"В запрос уходят последние: {MAX_HISTORY_MESSAGES}\n"
        f"Диалог протухает после {SESSION_TTL_MINUTES} мин молчания\n"
        f"Активных напоминаний: {len(pending)}"
    )


@dp.message(Command("reminders"))
async def handle_reminders(message: Message, reminders: ReminderStore) -> None:
    """Список напоминаний напрямую из базы, без обращения к модели.

    Дублирует то, что умеет сама модель, но работает мгновенно и не тратит
    лимит запросов — удобно, чтобы просто свериться.
    """
    pending = reminders.list_pending(message.from_user.id)
    if not pending:
        await message.answer("Активных напоминаний нет.")
        return

    lines = [f"#{r.id}: {r.text}\n     {r.schedule_description()}" for r in pending]
    await message.answer("Активные напоминания:\n\n" + "\n".join(lines))


async def reply_with_llm(
    message: Message,
    history: HistoryStore,
    llm: LLMClient,
    toolbox: ToolBox,
    text: str,
) -> None:
    """Общий путь для любого сообщения, доведённого до текста.

    Сюда попадают и набранное вручную, и распознанное из голосового: с точки
    зрения диалога разницы нет, поэтому логика одна на оба случая.
    """
    user_id = message.from_user.id

    # Замок на пользователя: пока обрабатываем одно сообщение, следующее ждёт.
    async with _locks[user_id]:
        session, expired = history.get_or_start(user_id)

        try:
            # ChatActionSender показывает статус «печатает…» и сам продлевает
            # его, пока мы ждём модель: Telegram гасит этот статус через 5 секунд.
            async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
                # Историю передаём БЕЗ нового сообщения — respond добавит его сам.
                # Так при ошибке в истории не останется висящей реплики без ответа,
                # и повтор запроса не создаст дубликат.
                answer = await respond(
                    llm=llm,
                    toolbox=toolbox,
                    ctx=ToolContext(user_id=user_id, chat_id=message.chat.id),
                    history=history.llm_messages(session),
                    user_text=text,
                )
        except ProviderError as exc:
            # Текст такой ошибки писался с расчётом на показ пользователю.
            await message.answer(str(exc))
            return
        except Exception:
            # Непредвиденное. Пользователю — общая фраза, разработчику — трейсбек,
            # иначе бот молча «проглотит» сообщение, и причина останется загадкой.
            logger.exception("Непредвиденная ошибка при обработке сообщения")
            await message.answer("Что-то пошло не так. Загляни в логи бота.")
            return

        # Сохраняем обе реплики только после успешного ответа.
        history.add_message(session, "user", text)
        history.add_message(session, "assistant", answer)

    if expired:
        answer = f"{NEW_SESSION_NOTICE}\n\n{answer}"

    await send_long(message, answer)


@dp.message(F.text)
async def handle_text(
    message: Message,
    history: HistoryStore,
    llm: LLMClient,
    toolbox: ToolBox,
) -> None:
    """Обычное текстовое сообщение.

    history, llm и toolbox приезжают сюда автоматически: aiogram подставляет в
    аргументы хендлера значения, положенные в диспетчер при старте (см. main).
    """
    await reply_with_llm(message, history, llm, toolbox, message.text)


@dp.message(F.voice)
async def handle_voice(
    message: Message,
    history: HistoryStore,
    llm: LLMClient,
    toolbox: ToolBox,
    stt: SpeechToText,
) -> None:
    """Голосовое сообщение: распознаём речь и отвечаем как на обычный текст."""
    voice = message.voice

    if voice.duration > MAX_VOICE_SECONDS:
        await message.answer(
            f"Голосовое слишком длинное ({voice.duration} с). "
            f"Максимум — {MAX_VOICE_SECONDS} с."
        )
        return

    if voice.duration < 1:
        # Случайное касание кнопки микрофона. Провайдер на такой файл вернёт
        # ошибку «слишком короткое аудио» — лучше отсечь до загрузки.
        await message.answer("Запись слишком короткая — я ничего не услышал.")
        return

    try:
        # Качаем файл в память, а не на диск: голосовые небольшие, а временные
        # файлы пришлось бы чистить и думать о правах внутри контейнера.
        buffer = BytesIO()
        await message.bot.download(voice, destination=buffer)

        # «Печатает…» показываем и на время распознавания — пользователю не
        # важно, на каком именно этапе бот занят, важно что он не завис.
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            text = await stt.transcribe(buffer.getvalue())
    except ProviderError as exc:
        await message.answer(str(exc))
        return
    except Exception:
        logger.exception("Не удалось обработать голосовое сообщение")
        await message.answer("Не смог обработать голосовое. Загляни в логи бота.")
        return

    if not text:
        await message.answer("Не разобрал ни слова — попробуй записать ещё раз.")
        return

    # Показываем распознанный текст отдельным сообщением. Whisper иногда
    # ошибается, и лучше сразу видеть, что именно бот услышал, чем гадать,
    # почему ответ не по теме.
    await message.answer(f"🎤 {text}")

    await reply_with_llm(message, history, llm, toolbox, text)


@dp.message()
async def handle_unsupported(message: Message) -> None:
    """Всё остальное: фото, стикеры, документы.

    Хендлер без фильтров ловит то, что не подошло предыдущим. Порядок важен:
    aiogram проверяет обработчики сверху вниз и берёт первый подходящий.
    """
    await message.answer("Пока я понимаю только текст и голосовые сообщения.")


async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    llm = LLMClient()
    stt = SpeechToText()
    history = HistoryStore(
        ttl_minutes=SESSION_TTL_MINUTES,
        max_messages=MAX_HISTORY_MESSAGES,
    )

    # База создаётся при первом запуске, отдельная установка не нужна.
    conn = connect()
    reminders = ReminderStore(conn)
    toolbox = ToolBox(reminders)
    scheduler = ReminderScheduler(reminders, bot, interval_seconds=REMINDER_TICK_SECONDS)

    # Кладём зависимости в диспетчер. aiogram передаст их в те хендлеры,
    # которые объявили аргументы с такими именами. Это удобнее глобальных
    # переменных: в тестах можно подсунуть другую реализацию.
    dp["history"] = history
    dp["llm"] = llm
    dp["stt"] = stt
    dp["reminders"] = reminders
    dp["toolbox"] = toolbox

    # Узнаём, под каким аккаунтом мы подключились. Заодно это первая проверка
    # токена: неверный токен даст здесь внятную ошибку, а не тишину.
    me = await bot.get_me()
    logger.info(
        "Бот @%s запущен. Модель: %s, распознавание речи: %s. Разрешённые ID: %s",
        me.username,
        LLM_MODEL,
        STT_MODEL,
        sorted(ALLOWED_USER_IDS),
    )

    # Планировщик стартует до опроса Telegram: если бот лежал и что-то
    # просрочено, оно уйдёт сразу, не дожидаясь первого сообщения.
    scheduler.start()

    try:
        # Long polling: бот сам периодически спрашивает Telegram о новых апдейтах.
        # Для разработки удобнее webhook'ов — не нужен публичный HTTPS-адрес.
        await dp.start_polling(bot)
    finally:
        # Корректно закрываем соединения при остановке (Ctrl+C).
        await scheduler.stop()
        conn.close()
        await llm.close()
        await stt.close()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
