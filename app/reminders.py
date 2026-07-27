"""Напоминания: хранение в SQLite и планировщик отправки.

Планировщик устроен максимально просто: фоновая задача раз в N секунд
спрашивает у базы «что уже пора» и отправляет найденное.

Готовые библиотеки (APScheduler и подобные) держат собственный список задач,
который надо синхронизировать с нашей таблицей — два источника правды, которые
рано или поздно разъедутся. Здесь источник один: база. Переживание перезапуска
получается бесплатно — бот поднялся, заглянул в таблицу, увидел просроченное.
Цена — точность порядка интервала опроса, что для напоминаний несущественно.
"""

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from aiogram import Bot

from app.config import MISSED_REMINDER_GRACE_DAYS, TIMEZONE
from app.prompts import format_now
from app.recurrence import Recurrence

logger = logging.getLogger(__name__)

# Статусы напоминания.
PENDING = "pending"      # ждёт срабатывания
SENT = "sent"            # отправлено
CANCELLED = "cancelled"  # отменено пользователем
EXPIRED = "expired"      # просрочено настолько, что отправлять уже бессмысленно


@dataclass(frozen=True)
class Reminder:
    id: int
    user_id: int
    chat_id: int
    text: str
    fire_at: datetime  # всегда с часовым поясом, всегда UTC
    created_at: datetime
    status: str
    # None — одноразовое напоминание. Для повторяющегося fire_at означает
    # ближайшее срабатывание и сдвигается вперёд после каждой отправки.
    repeat: Recurrence | None = None

    @property
    def is_recurring(self) -> bool:
        return self.repeat is not None

    def local_time(self) -> str:
        """Время срабатывания словами, в часовом поясе пользователя."""
        return format_now(self.fire_at.astimezone(TIMEZONE))

    def schedule_description(self) -> str:
        """Как описать расписание пользователю."""
        if self.repeat is None:
            return self.local_time()
        return f"{self.repeat.describe()} (ближайшее — {self.local_time()})"


def _row_to_reminder(row: sqlite3.Row) -> Reminder:
    return Reminder(
        id=row["id"],
        user_id=row["user_id"],
        chat_id=row["chat_id"],
        text=row["text"],
        fire_at=datetime.fromisoformat(row["fire_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        status=row["status"],
        repeat=Recurrence.from_json(row["repeat_rule"]),
    )


class ReminderStore:
    """Работа с таблицей напоминаний."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(
        self,
        user_id: int,
        chat_id: int,
        text: str,
        fire_at: datetime,
        repeat: Recurrence | None = None,
    ) -> Reminder:
        """Создаёт напоминание. fire_at обязано быть с часовым поясом."""
        fire_at_utc = fire_at.astimezone(timezone.utc)
        created_at = datetime.now(timezone.utc)

        cursor = self._conn.execute(
            "INSERT INTO reminders (user_id, chat_id, text, fire_at, created_at, status, repeat_rule) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                chat_id,
                text,
                fire_at_utc.isoformat(),
                created_at.isoformat(),
                PENDING,
                repeat.to_json() if repeat else None,
            ),
        )
        self._conn.commit()

        return Reminder(
            id=cursor.lastrowid,
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            fire_at=fire_at_utc,
            created_at=created_at,
            status=PENDING,
            repeat=repeat,
        )

    def list_pending(self, user_id: int) -> list[Reminder]:
        """Активные напоминания пользователя, ближайшие первыми."""
        rows = self._conn.execute(
            "SELECT * FROM reminders WHERE user_id = ? AND status = ? ORDER BY fire_at",
            (user_id, PENDING),
        ).fetchall()
        return [_row_to_reminder(row) for row in rows]

    def cancel(self, reminder_id: int, user_id: int) -> Reminder | None:
        """Отменяет напоминание. Возвращает отменённое или None, если нечего.

        user_id в условии — не паранойя, а защита от того, что модель
        подставит идентификатор из прошлого разговора или просто выдумает его.
        """
        row = self._conn.execute(
            "SELECT * FROM reminders WHERE id = ? AND user_id = ? AND status = ?",
            (reminder_id, user_id, PENDING),
        ).fetchone()
        if row is None:
            return None

        self._conn.execute(
            "UPDATE reminders SET status = ? WHERE id = ?", (CANCELLED, reminder_id)
        )
        self._conn.commit()
        return _row_to_reminder(row)

    def due(self, now: datetime) -> list[Reminder]:
        """Напоминания, время которых уже наступило."""
        rows = self._conn.execute(
            "SELECT * FROM reminders WHERE status = ? AND fire_at <= ? ORDER BY fire_at",
            (PENDING, now.astimezone(timezone.utc).isoformat()),
        ).fetchall()
        return [_row_to_reminder(row) for row in rows]

    def set_status(self, reminder_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE reminders SET status = ? WHERE id = ?", (status, reminder_id)
        )
        self._conn.commit()

    def reschedule(self, reminder_id: int, next_fire_at: datetime) -> None:
        """Сдвигает повторяющееся напоминание на следующее срабатывание.

        Запись остаётся в статусе pending: серия не заканчивается, просто
        наступает следующий круг.
        """
        self._conn.execute(
            "UPDATE reminders SET fire_at = ? WHERE id = ?",
            (next_fire_at.astimezone(timezone.utc).isoformat(), reminder_id),
        )
        self._conn.commit()


class ReminderScheduler:
    """Фоновая задача, отправляющая напоминания в назначенное время."""

    def __init__(
        self,
        store: ReminderStore,
        bot: Bot,
        interval_seconds: int = 30,
        grace_days: int = MISSED_REMINDER_GRACE_DAYS,
    ) -> None:
        self._store = store
        self._bot = bot
        self._interval = interval_seconds
        # Насколько поздно ещё имеет смысл отправлять пропущенное напоминание.
        self._grace = timedelta(days=grace_days)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="reminder-scheduler")
        logger.info("Планировщик запущен, интервал %s с", self._interval)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        logger.info("Планировщик остановлен")

    async def _run(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Планировщик обязан пережить любую ошибку: если задача умрёт,
                # напоминания перестанут приходить молча.
                logger.exception("Ошибка в цикле планировщика")
            await asyncio.sleep(self._interval)

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)

        for reminder in self._store.due(now):
            delay = now - reminder.fire_at
            too_late = delay > self._grace

            if too_late:
                # Бот пролежал слишком долго: напоминание «позвонить маме»
                # недельной давности уже не полезно, а только сбивает с толку.
                logger.info(
                    "Напоминание %s просрочено на %s — не отправляем", reminder.id, delay
                )
            else:
                text = f"🔔 {reminder.text}"
                if delay > timedelta(minutes=1):
                    # Опоздали — честно говорим об этом, иначе выглядит как сбой.
                    text += f"\n\n(должно было сработать: {reminder.local_time()})"

                try:
                    await self._bot.send_message(reminder.chat_id, text)
                    logger.info(
                        "Отправлено напоминание %s пользователю %s",
                        reminder.id,
                        reminder.user_id,
                    )
                except Exception:
                    # Пользователь заблокировал бота, чат удалён и т.п. Не даём
                    # записи остаться в очереди, иначе будем биться об неё каждый круг.
                    logger.exception("Не удалось отправить напоминание %s", reminder.id)

            self._finish(reminder, now)

    def _finish(self, reminder: Reminder, now: datetime) -> None:
        """Закрывает разовое напоминание или переводит серию на следующий круг."""
        if not reminder.is_recurring:
            self._store.set_status(reminder.id, SENT if now - reminder.fire_at <= self._grace else EXPIRED)
            return

        # Считаем следующее срабатывание от «сейчас», а не от прошлого времени:
        # если бот лежал неделю, пользователь получит одно напоминание, а не
        # семь подряд за каждый пропущенный день.
        next_at = reminder.repeat.next_after(now)
        self._store.reschedule(reminder.id, next_at)
        logger.info("Напоминание %s повторится %s", reminder.id, next_at.isoformat())
