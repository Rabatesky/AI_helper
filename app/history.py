"""Хранилище диалогов: сессии, протухание по бездействию, обрезка контекста.

Ключевая идея — разговор не бесконечен. Если спросить утром про историю Москвы,
а вечером про SQL, это два разных разговора, и мешать их в одну кучу не нужно:
модель начнёт притягивать неуместный контекст, а лишние сообщения будут
съедать лимиты.

Сейчас всё лежит в памяти процесса и теряется при перезапуске. Это осознанный
компромисс на текущем шаге. Раскладка данных при этом уже такая, какой она
будет в SQLite (сессия -> сообщения), поэтому позже поменяется только
реализация хранилища, а не логика бота.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

# Роли в диалоге — те же, что понимает API модели.
# "user" — реплики человека, "assistant" — ответы модели.
Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """Одна реплика диалога."""

    role: Role
    content: str
    created_at: datetime


@dataclass
class Session:
    """Один разговор: непрерывная последовательность реплик."""

    session_id: int
    user_id: int
    started_at: datetime
    last_activity_at: datetime
    messages: list[ChatMessage] = field(default_factory=list)


class HistoryStore:
    """Диалоги всех пользователей.

    Времена храним в UTC. Приводить к часовому поясу пользователя будем только
    при показе — так не придётся думать о том, где физически стоит сервер.
    """

    def __init__(self, ttl_minutes: int, max_messages: int) -> None:
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_messages = max_messages
        self._sessions: dict[int, Session] = {}
        self._next_session_id = 1

    def get_or_start(self, user_id: int, now: datetime | None = None) -> tuple[Session, bool]:
        """Возвращает активную сессию и признак «прежний разговор протух».

        Второе значение — именно про протухание, а не про «сессия новая».
        Разница важна для интерфейса: сообщить «начал новый диалог» нужно
        тому, у кого предыдущий разговор оборвался по таймауту, и не нужно
        тому, кто пишет боту впервые или только что сам вызвал /new.

        Проверка протухания делается здесь, в момент обращения. Фоновый таймер
        не нужен: пока пользователь молчит, состояние никого не интересует.
        """
        now = now or datetime.now(timezone.utc)
        previous = self._sessions.get(user_id)

        if previous is not None and now - previous.last_activity_at <= self._ttl:
            return previous, False

        expired = previous is not None
        session = Session(
            session_id=self._next_session_id,
            user_id=user_id,
            started_at=now,
            last_activity_at=now,
        )
        self._next_session_id += 1
        self._sessions[user_id] = session
        return session, expired

    def add_message(
        self,
        session: Session,
        role: Role,
        content: str,
        now: datetime | None = None,
    ) -> None:
        """Добавляет реплику в сессию и продлевает её жизнь."""
        now = now or datetime.now(timezone.utc)
        session.messages.append(ChatMessage(role=role, content=content, created_at=now))
        session.last_activity_at = now

    def reset(self, user_id: int) -> bool:
        """Принудительно завершает текущий разговор (команда /new).

        Возвращает True, если было что завершать, — чтобы бот мог по-разному
        ответить на «начали новый диалог» и «а он и так пустой».
        """
        session = self._sessions.pop(user_id, None)
        return session is not None and bool(session.messages)

    def llm_messages(self, session: Session) -> list[dict[str, str]]:
        """Готовит историю в формате, который ждёт API модели.

        Берём только последние max_messages реплик: окно контекста модели
        конечно, а каждый лишний токен — это время ответа и расход лимита.
        В самой сессии при этом остаётся полная история — она пригодится,
        когда захотим складывать старые разговоры в архив.

        Системный промпт сюда не входит: он собирается заново на каждый
        запрос, потому что содержит текущее время.
        """
        recent = session.messages[-self._max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]
