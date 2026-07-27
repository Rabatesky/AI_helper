"""Цикл «модель просит — мы выполняем — модель формулирует ответ».

Это то место, где решается, разговор перед нами или просьба о действии. Точнее,
решает модель: получив вместе с сообщением описания инструментов, она возвращает
либо текст, либо просьбу вызвать функцию. Мы лишь исполняем и отдаём результат
обратно, чтобы она превратила его в человеческую фразу.

Кругов может быть несколько подряд: «посмотреть список» -> «отменить нужное» ->
ответ. Их число ограничено, чтобы модель не могла зациклиться.
"""

import logging

from app.config import MAX_TOOL_ROUNDS
from app.llm import LLMClient, LLMReply
from app.prompts import build_system_prompt
from app.tools import TOOL_SPECS, ToolBox, ToolContext

logger = logging.getLogger(__name__)


def _assistant_message(reply: LLMReply) -> dict:
    """Собирает реплику модели для отправки обратно в следующем запросе.

    Формируем вручную, а не пересылаем ответ провайдера целиком: разные
    провайдеры добавляют свои поля, и часть из них API потом не принимает.
    """
    return {
        "role": "assistant",
        "content": reply.text or "",
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            }
            for call in reply.tool_calls
        ],
    }


async def respond(
    *,
    llm: LLMClient,
    toolbox: ToolBox,
    ctx: ToolContext,
    history: list[dict[str, str]],
    user_text: str,
) -> str:
    """Возвращает финальный текст ответа, выполнив нужные инструменты по пути."""
    messages: list[dict] = [*history, {"role": "user", "content": user_text}]

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        reply = await llm.complete(
            system=build_system_prompt(),
            messages=messages,
            tools=TOOL_SPECS,
        )

        if not reply.wants_tools:
            return reply.text or "Модель вернула пустой ответ. Попробуй переформулировать."

        # Модель попросила действие. Её реплику обязательно кладём в переписку:
        # без неё результаты вызовов повиснут в воздухе и API отвергнет запрос.
        messages.append(_assistant_message(reply))

        for call in reply.tool_calls:
            logger.info(
                "Круг %s: вызываем %s(%s)", round_number, call.name, call.arguments
            )
            result = toolbox.execute(call.name, call.arguments, ctx)

            # Роль "tool" — ответ на конкретный вызов, связь по tool_call_id.
            messages.append(
                {"role": "tool", "tool_call_id": call.call_id, "content": result}
            )

    # Круги кончились, а модель всё просит инструменты. Скорее всего зациклилась.
    logger.warning("Исчерпан лимит кругов вызова инструментов (%s)", MAX_TOOL_ROUNDS)
    return (
        "Что-то я запутался и не смог довести действие до конца. "
        "Попробуй сформулировать иначе."
    )
