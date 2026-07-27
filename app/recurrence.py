"""Правила повторения напоминаний и вычисление следующего срабатывания.

Поддерживаем четыре вида правил: каждый день, по определённым дням недели,
раз в месяц и через равные промежутки времени. Набор возможных значений
остаётся настолько узким, что модель почти не может ошибиться, а мы можем
всё проверить.

Интервальный вид появился не сразу, и его отсутствие вылезло характерным
образом: на просьбу «напоминай каждые полчаса» модель, не найдя подходящего
инструмента, принялась ставить разовые напоминания одно за другим, пока не
упёрлась в предохранитель. Это общее свойство работы с инструментами — модель
не говорит «не умею», а выкручивается тем, что есть, поэтому пробел в
возможностях проявляется как странное поведение.

Полноценные календарные правила (RRULE) или cron-строки дали бы больше свободы
ценой чужой библиотеки и целого класса краевых случаев вроде «пятая пятница
месяца». Когда понадобится — заменится здесь, не трогая остальной код.

Все вычисления идут в часовом поясе пользователя и только в конце переводятся
в UTC. Обратный порядок дал бы сдвиг времени при смене летнего времени.
"""

import calendar
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from app.config import TIMEZONE

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
INTERVAL = "interval"

# Границы интервального повторения. Нижняя защищает от превращения бота в
# спамера, верхняя не нужна технически, но всё, что реже суток, естественнее
# выражается ежедневным или ежемесячным правилом.
MIN_INTERVAL_MINUTES = 5
MAX_INTERVAL_MINUTES = 24 * 60

# Дни недели по индексам datetime.weekday(): 0 — понедельник.
WEEKDAYS_PLURAL = (
    "понедельникам", "вторникам", "средам", "четвергам",
    "пятницам", "субботам", "воскресеньям",
)

WORKDAYS = frozenset({0, 1, 2, 3, 4})
WEEKEND = frozenset({5, 6})


class RecurrenceError(ValueError):
    """Правило повторения задано некорректно."""


def _plural(number: int, one: str, few: str, many: str) -> str:
    """Русская форма слова при числе: 1 час, 2 часа, 5 часов."""
    if 11 <= number % 100 <= 14:
        return many
    last = number % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many


@dataclass(frozen=True)
class Recurrence:
    """Правило повторения: что именно и в котором часу."""

    kind: str                     # daily | weekly | monthly | interval
    hour: int = 0
    minute: int = 0
    weekdays: frozenset[int] = frozenset()  # для weekly, 0 — понедельник
    day_of_month: int = 1                   # для monthly
    every_minutes: int = 0                  # для interval

    # --- Создание и проверка -------------------------------------------------

    @classmethod
    def build(
        cls,
        kind: str,
        at: str = "",
        weekdays: list | None = None,
        day_of_month: int | None = None,
        every_minutes: int | None = None,
    ) -> "Recurrence":
        """Собирает правило из аргументов модели, проверяя каждое значение."""
        kind = (kind or "").strip().lower()
        if kind not in (DAILY, WEEKLY, MONTHLY, INTERVAL):
            raise RecurrenceError(
                f"неизвестный вид повторения {kind!r}, "
                f"ожидается {DAILY}, {WEEKLY}, {MONTHLY} или {INTERVAL}"
            )

        if kind == INTERVAL:
            # У интервального правила нет времени суток: оно отсчитывается
            # от момента создания, а дальше — от каждого срабатывания.
            return cls(kind=kind, every_minutes=cls._parse_interval(every_minutes))

        hour, minute = cls._parse_time(at)

        if kind == WEEKLY:
            parsed_days = cls._parse_weekdays(weekdays)
            return cls(kind=kind, hour=hour, minute=minute, weekdays=parsed_days)

        if kind == MONTHLY:
            if day_of_month is None:
                raise RecurrenceError("для ежемесячного повторения нужно число месяца")
            try:
                day = int(day_of_month)
            except (TypeError, ValueError):
                raise RecurrenceError(f"число месяца должно быть числом, а получено {day_of_month!r}")
            if not 1 <= day <= 31:
                raise RecurrenceError(f"число месяца должно быть от 1 до 31, а получено {day}")
            return cls(kind=kind, hour=hour, minute=minute, day_of_month=day)

        return cls(kind=kind, hour=hour, minute=minute)

    @staticmethod
    def _parse_time(at: str) -> tuple[int, int]:
        """Разбирает время вида «09:00»."""
        raw = (at or "").strip()
        try:
            parsed = time.fromisoformat(raw)
        except ValueError:
            raise RecurrenceError(f"время {raw!r} не распознано, нужен формат ЧЧ:ММ")
        return parsed.hour, parsed.minute

    @staticmethod
    def _parse_interval(every_minutes: int | None) -> int:
        """Проверяет длину интервала в минутах."""
        if every_minutes is None:
            raise RecurrenceError("для интервального повторения нужно указать every_minutes")
        try:
            minutes = int(every_minutes)
        except (TypeError, ValueError):
            raise RecurrenceError(f"интервал должен быть числом минут, а получено {every_minutes!r}")

        if not MIN_INTERVAL_MINUTES <= minutes <= MAX_INTERVAL_MINUTES:
            raise RecurrenceError(
                f"интервал должен быть от {MIN_INTERVAL_MINUTES} минут до "
                f"{MAX_INTERVAL_MINUTES // 60} часов, а получено {minutes} минут"
            )
        return minutes

    @staticmethod
    def _parse_weekdays(weekdays: list | None) -> frozenset[int]:
        """Приводит список дней недели к номерам 0..6."""
        if not weekdays:
            raise RecurrenceError("для еженедельного повторения нужен список дней недели")

        result: set[int] = set()
        for item in weekdays:
            try:
                number = int(item)
            except (TypeError, ValueError):
                raise RecurrenceError(f"день недели должен быть числом 1..7, а получено {item!r}")
            if not 1 <= number <= 7:
                raise RecurrenceError(f"день недели должен быть от 1 до 7, а получено {number}")
            # Модели удобнее «1 — понедельник», внутри считаем от нуля.
            result.add(number - 1)
        return frozenset(result)

    # --- Вычисление следующего срабатывания ----------------------------------

    def next_after(self, moment: datetime) -> datetime:
        """Ближайшее срабатывание строго позже указанного момента, в UTC."""
        if self.kind == INTERVAL:
            # Отсчитываем от переданного момента. Часовой пояс тут ни при чём:
            # «каждые полчаса» одинаковы в любой точке мира.
            return (moment + timedelta(minutes=self.every_minutes)).astimezone(timezone.utc)

        local = moment.astimezone(TIMEZONE)

        if self.kind == DAILY:
            candidate = self._at_time(local.date())
            if candidate <= local:
                candidate = self._at_time(local.date() + timedelta(days=1))
            return candidate.astimezone(timezone.utc)

        if self.kind == WEEKLY:
            # Проверяем сегодня и следующие семь дней — этого всегда достаточно.
            for offset in range(8):
                day = local.date() + timedelta(days=offset)
                if day.weekday() not in self.weekdays:
                    continue
                candidate = self._at_time(day)
                if candidate > local:
                    return candidate.astimezone(timezone.utc)
            raise RecurrenceError("не удалось вычислить следующий день недели")

        # MONTHLY: перебираем текущий месяц и следующие.
        year, month = local.year, local.month
        for _ in range(13):
            candidate = self._at_time(self._clamp_day(year, month))
            if candidate > local:
                return candidate.astimezone(timezone.utc)
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        raise RecurrenceError("не удалось вычислить следующий месяц")

    def _at_time(self, day: date) -> datetime:
        """Дата + время правила в часовом поясе пользователя."""
        return datetime.combine(day, time(self.hour, self.minute), tzinfo=TIMEZONE)

    def _clamp_day(self, year: int, month: int) -> date:
        """Подгоняет число месяца под его длину.

        «31 числа каждого месяца» в феврале означает последний день февраля,
        а не пропуск месяца: пользователь скорее имел в виду «в конце месяца».
        """
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, min(self.day_of_month, last_day))

    # --- Представление -------------------------------------------------------

    def describe(self) -> str:
        """Человекочитаемое описание правила."""
        if self.kind == INTERVAL:
            return self._describe_interval()

        at = f"{self.hour:02d}:{self.minute:02d}"

        if self.kind == DAILY:
            return f"каждый день в {at}"

        if self.kind == MONTHLY:
            return f"{self.day_of_month} числа каждого месяца в {at}"

        if self.weekdays == WORKDAYS:
            return f"по будням в {at}"
        if self.weekdays == WEEKEND:
            return f"по выходным в {at}"

        names = [WEEKDAYS_PLURAL[d] for d in sorted(self.weekdays)]
        if len(names) > 1:
            listed = ", ".join(names[:-1]) + " и " + names[-1]
        else:
            listed = names[0]
        return f"по {listed} в {at}"

    def _describe_interval(self) -> str:
        """«каждые 30 минут», «каждый час», «каждые 2 часа»."""
        minutes = self.every_minutes

        if minutes % 60 == 0:
            hours = minutes // 60
            if hours == 1:
                return "каждый час"
            return f"каждые {hours} {_plural(hours, 'час', 'часа', 'часов')}"

        if minutes == 1:
            return "каждую минуту"
        return f"каждые {minutes} {_plural(minutes, 'минуту', 'минуты', 'минут')}"

    # --- Хранение ------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "hour": self.hour,
                "minute": self.minute,
                "weekdays": sorted(self.weekdays),
                "day_of_month": self.day_of_month,
                "every_minutes": self.every_minutes,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str | None) -> "Recurrence | None":
        if not raw:
            return None
        data = json.loads(raw)
        return cls(
            kind=data["kind"],
            # Значения по умолчанию нужны для записей, созданных до появления
            # соответствующих полей: старые правила в базе их не содержат.
            hour=data.get("hour", 0),
            minute=data.get("minute", 0),
            weekdays=frozenset(data.get("weekdays", [])),
            day_of_month=data.get("day_of_month", 1),
            every_minutes=data.get("every_minutes", 0),
        )
