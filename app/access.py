"""Контроль доступа: бот реагирует только на пользователей из белого списка.

Реализовано как outer middleware — прослойка, через которую проходит КАЖДЫЙ
апдейт от Telegram до того, как aiogram начнёт подбирать обработчик.
Почему именно так, а не фильтром на каждом хендлере:

  * правило описано один раз и автоматически действует на всё, что мы добавим
    дальше (голосовые, кнопки, команды) — невозможно случайно забыть фильтр;
  * посторонний апдейт отбрасывается максимально рано, до любой логики.
"""

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

logger = logging.getLogger(__name__)


class WhitelistMiddleware(BaseMiddleware):
    """Пропускает дальше только апдейты от разрешённых пользователей."""

    def __init__(self, allowed_user_ids: frozenset[int]) -> None:
        self.allowed_user_ids = allowed_user_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # event здесь — объект Update (контейнер апдейта). Его свойство .event
        # возвращает то, что лежит внутри: Message, CallbackQuery и т.д.
        inner_event = event.event if isinstance(event, Update) else event
        user = getattr(inner_event, "from_user", None)

        if user is None or user.id not in self.allowed_user_ids:
            # Молча выходим, не вызывая handler. Для чужого человека бот выглядит
            # неотвечающим — это лучше, чем сообщение «у вас нет доступа»,
            # которое подтверждает, что бот жив, и провоцирует продолжать.
            logger.warning(
                "Отклонён апдейт от user_id=%s (username=%s)",
                getattr(user, "id", "unknown"),
                getattr(user, "username", None),
            )
            return None

        # Пользователь свой — передаём управление дальше по цепочке.
        return await handler(event, data)
