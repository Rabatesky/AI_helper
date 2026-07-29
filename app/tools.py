"""Инструменты, которые модель может попросить выполнить.

Здесь две связанные вещи: описания функций для модели (TOOL_SPECS) и их
исполнение (ToolBox). Описание — это тоже промпт: от его точности напрямую
зависит, поймёт ли модель, когда функцию звать, а когда нет. Формулировки
здесь выстраданы на тестах, менять их стоит осознанно.

Результат выполнения возвращается модели обычным текстом — в том числе текст
ошибки. Это штатный путь: получив «время в прошлом», модель переспросит или
исправится сама, без нашего участия.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import TIMEZONE
from app.prompts import format_now
from app.recurrence import Recurrence, RecurrenceError
from app.reminders import ReminderStore
from app.websearch import SearchError, WebSearch

logger = logging.getLogger(__name__)

# Ограничение на длину текста напоминания: модель иногда норовит записать
# в него целый абзац вместо сути.
MAX_REMINDER_TEXT = 300


@dataclass(frozen=True)
class ToolContext:
    """Кто именно попросил. Модель этих значений не видит и подделать не может."""

    user_id: int
    chat_id: int


TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": (
                "Поставить одноразовое напоминание на конкретный момент времени. "
                "Использовать, когда пользователь просит напомнить о будущем деле "
                "или говорит, что боится его забыть. "
                "НЕ использовать, если пользователь задаёт вопрос, просит что-то "
                "объяснить, рассказать или показать — это обычный разговор. "
                "НИКОГДА не вызывай этот инструмент несколько раз подряд, чтобы "
                "изобразить повторяющееся напоминание: для повторов есть "
                "create_recurring_reminder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "when": {
                        "type": "string",
                        "description": (
                            "Момент срабатывания в формате ISO 8601 со смещением "
                            "часового пояса, например 2026-07-28T09:00:00+03:00. "
                            "Вычисляется от текущего времени из системного сообщения."
                        ),
                    },
                    "what": {
                        "type": "string",
                        "description": (
                            "О чём напомнить — коротко, от лица пользователя. "
                            "Например: «позвонить маме»."
                        ),
                    },
                },
                "required": ["when", "what"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_recurring_reminder",
            "description": (
                "Поставить повторяющееся напоминание. Использовать всегда, когда "
                "пользователь просит напоминать регулярно: каждый день, по "
                "определённым дням недели, раз в месяц или через равные "
                "промежутки времени («каждые полчаса», «каждые два часа»). "
                "Для однократного напоминания используй create_reminder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["daily", "weekly", "monthly", "interval"],
                        "description": (
                            "daily — каждый день в указанное время; "
                            "weekly — по указанным дням недели; "
                            "monthly — раз в месяц в указанное число; "
                            "interval — через равные промежутки времени"
                        ),
                    },
                    "at": {
                        "type": "string",
                        "description": (
                            "Время срабатывания в формате ЧЧ:ММ, например 09:00. "
                            "Нужно для daily, weekly и monthly; для interval не указывается."
                        ),
                    },
                    "every_minutes": {
                        "type": "integer",
                        "description": (
                            "Только для kind=interval: длина промежутка в минутах, "
                            "от 5 до 1440. Полчаса это 30, два часа — 120."
                        ),
                    },
                    "what": {
                        "type": "string",
                        "description": "О чём напоминать — коротко, от лица пользователя",
                    },
                    "weekdays": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Только для kind=weekly: дни недели, где 1 — понедельник, "
                            "7 — воскресенье. Будни это [1,2,3,4,5]."
                        ),
                    },
                    "day_of_month": {
                        "type": "integer",
                        "description": (
                            "Только для kind=monthly: число месяца от 1 до 31. "
                            "В коротких месяцах сработает в последний день."
                        ),
                    },
                },
                "required": ["kind", "at", "what"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            # Имя намеренно не «search» и не «search_web»: модели gpt-oss
            # обучены со встроенным браузером, где поиск идёт в два шага
            # (search, затем open с полями id и cursor). Похожее имя будило
            # этот заученный сценарий, и модель слала аргументы от чужого
            # инструмента вместо нашего query.
            "name": "lookup_current_info",
            "description": (
                "Найти актуальную информацию в интернете. Использовать всегда, "
                "когда ответ зависит от сегодняшнего дня или от данных, которых "
                "нет в твоих знаниях: погода, курсы валют, новости, цены, часы "
                "работы заведений, результаты событий. "
                "НЕ использовать для общих знаний, которые ты и так знаешь, — "
                "определения, объяснения, история, программирование. "
                "Инструмент сразу возвращает готовые выдержки из найденных "
                "страниц: открывать страницы отдельным вызовом не нужно и нельзя. "
                "Единственный аргумент — query. "
                "Если в выдержках нет нужного, НЕ повторяй похожий запрос по "
                "нескольку раз: либо задай принципиально другой запрос, либо "
                "ответь по тому, что нашлось, честно оговорив неполноту."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Поисковый запрос — так, как его набрал бы человек "
                            "в поисковике. Например: «погода Казань сегодня». "
                            "Не пиши вопрос целиком фразой из диалога."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": (
                "Показать активные напоминания пользователя. Использовать, когда "
                "он спрашивает, что у него запланировано или о чём ему напомнят."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": (
                "Отменить ранее созданное напоминание по его номеру. Для "
                "повторяющегося напоминания отменяет всю серию. Если номер "
                "неизвестен, сначала вызови list_reminders и уточни у пользователя, "
                "какое именно напоминание отменить."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "integer",
                        "description": "Номер напоминания из списка",
                    },
                },
                "required": ["reminder_id"],
            },
        },
    },
]


class ToolBox:
    """Исполняет инструменты, о которых просит модель.

    Метод execute асинхронный, хотя работа с базой синхронная: сетевые навыки
    (поиск, а в будущем обращения к внешним API) обязаны быть асинхронными,
    иначе на время запроса встанет обработка всех остальных сообщений.
    """

    def __init__(self, store: ReminderStore, search: WebSearch) -> None:
        self._store = store
        self._search = search

    async def execute(self, name: str, arguments: dict, ctx: ToolContext) -> str:
        """Выполняет инструмент и возвращает текстовый результат для модели."""
        handler = {
            "create_reminder": self._create_reminder,
            "create_recurring_reminder": self._create_recurring_reminder,
            "lookup_current_info": self._search_web,
            "list_reminders": self._list_reminders,
            "cancel_reminder": self._cancel_reminder,
        }.get(name)

        if handler is None:
            # Модель выдумала функцию, которой нет. Сообщаем ей об этом —
            # обычно после такого ответа она выкручивается сама.
            logger.warning("Модель запросила неизвестный инструмент: %s", name)
            return f"Ошибка: инструмента {name} не существует."

        try:
            return await handler(arguments, ctx)
        except Exception:
            logger.exception("Ошибка при выполнении инструмента %s", name)
            return "Ошибка: не удалось выполнить действие."

    # --- Отдельные инструменты ------------------------------------------------

    async def _search_web(self, args: dict, ctx: ToolContext) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "Ошибка: не указан поисковый запрос."

        try:
            results = await self._search.search(query)
        except SearchError as exc:
            return f"Ошибка: {exc}"

        if not results:
            return f"По запросу «{query}» ничего не нашлось. Попробуй переформулировать."

        # Отдаём модели пронумерованный список: заголовок, выдержка, адрес.
        # Модель сама выберет нужное и сформулирует ответ человеку.
        lines = [f"Результаты поиска по запросу «{query}»:"]
        for number, item in enumerate(results, start=1):
            lines.append(f"{number}. {item.title}\n   {item.snippet}\n   Источник: {item.url}")
        return "\n".join(lines)

    async def _create_reminder(self, args: dict, ctx: ToolContext) -> str:
        raw_when = str(args.get("when", "")).strip()
        what = str(args.get("what", "")).strip()

        if not what:
            return "Ошибка: не указано, о чём напомнить."
        what = what[:MAX_REMINDER_TEXT]

        fire_at = self._parse_when(raw_when)
        if fire_at is None:
            return (
                f"Ошибка: не удалось разобрать время {raw_when!r}. "
                f"Нужен формат ISO 8601, например 2026-07-28T09:00:00+03:00."
            )

        now = datetime.now(timezone.utc)
        if fire_at <= now:
            # Модель ошиблась с датой — сообщаем текущее время, чтобы ей было
            # от чего пересчитать, и она переспросила или исправилась.
            return (
                f"Ошибка: {fire_at.astimezone(TIMEZONE):%d.%m.%Y %H:%M} уже прошло. "
                f"Сейчас {format_now(datetime.now(TIMEZONE))}. "
                f"Уточни у пользователя, когда именно напомнить."
            )

        reminder = self._store.create(ctx.user_id, ctx.chat_id, what, fire_at)
        logger.info("Создано напоминание %s на %s", reminder.id, reminder.fire_at)
        return (
            f"Напоминание #{reminder.id} создано: «{reminder.text}» "
            f"на {reminder.local_time()}."
        )

    async def _create_recurring_reminder(self, args: dict, ctx: ToolContext) -> str:
        what = str(args.get("what", "")).strip()
        if not what:
            return "Ошибка: не указано, о чём напоминать."
        what = what[:MAX_REMINDER_TEXT]

        try:
            rule = Recurrence.build(
                kind=str(args.get("kind", "")),
                at=str(args.get("at", "")),
                weekdays=args.get("weekdays"),
                day_of_month=args.get("day_of_month"),
                every_minutes=args.get("every_minutes"),
            )
        except RecurrenceError as exc:
            # Текст ошибки уходит модели — она исправится или переспросит.
            return f"Ошибка: {exc}."

        first_fire = rule.next_after(datetime.now(timezone.utc))
        reminder = self._store.create(ctx.user_id, ctx.chat_id, what, first_fire, repeat=rule)
        logger.info(
            "Создано повторяющееся напоминание %s: %s", reminder.id, rule.describe()
        )
        return (
            f"Повторяющееся напоминание #{reminder.id} создано: «{reminder.text}», "
            f"{rule.describe()}. Первое сработает {reminder.local_time()}."
        )

    async def _list_reminders(self, args: dict, ctx: ToolContext) -> str:
        reminders = self._store.list_pending(ctx.user_id)
        if not reminders:
            return "Активных напоминаний нет."

        lines = [
            f"#{r.id}: «{r.text}» — {r.schedule_description()}"
            for r in reminders
        ]
        return "Активные напоминания:\n" + "\n".join(lines)

    async def _cancel_reminder(self, args: dict, ctx: ToolContext) -> str:
        raw_id = args.get("reminder_id")
        try:
            reminder_id = int(raw_id)
        except (TypeError, ValueError):
            return f"Ошибка: номер напоминания должен быть числом, а получено {raw_id!r}."

        cancelled = self._store.cancel(reminder_id, ctx.user_id)
        if cancelled is None:
            # Либо номера не существует, либо он чужой, либо напоминание уже
            # сработало. Для модели разница несущественна.
            return (
                f"Ошибка: активного напоминания #{reminder_id} нет. "
                f"Покажи пользователю список через list_reminders."
            )

        logger.info("Отменено напоминание %s", reminder_id)
        return f"Напоминание #{reminder_id} («{cancelled.text}») отменено."

    @staticmethod
    def _parse_when(raw: str) -> datetime | None:
        """Разбирает время из ответа модели.

        Модель обычно присылает ISO со смещением, но иногда забывает часовой
        пояс. В этом случае считаем, что время указано в поясе пользователя, —
        так интуитивно правильнее, чем принять его за UTC и ошибиться на часы.
        """
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=TIMEZONE)
        return parsed.astimezone(timezone.utc)
