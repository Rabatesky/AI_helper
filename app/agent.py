"""Цикл «модель просит — мы выполняем — модель формулирует ответ».

Это то место, где решается, разговор перед нами или просьба о действии. Точнее,
решает модель: получив вместе с сообщением описания инструментов, она возвращает
либо текст, либо просьбу вызвать функцию. Мы лишь исполняем и отдаём результат
обратно, чтобы она превратила его в человеческую фразу.

Кругов может быть несколько подряд: «посмотреть список» -> «отменить нужное» ->
ответ. Их число ограничено, чтобы модель не могла зациклиться.
"""

import logging

from app.api_errors import ProviderError
from app.config import MAX_TOOL_ROUNDS
from app.llm import LLMClient, LLMReply, ToolCallSchemaError
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


def _call_key(name: str, arguments: dict) -> str:
    """Отпечаток вызова, чтобы узнавать повторы в рамках одного сообщения."""
    parts = sorted(f"{k}={str(v).strip().lower()}" for k, v in arguments.items())
    return f"{name}({','.join(parts)})"


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

    # Уже выполненные вызовы. Модель иногда повторяет почти тот же поиск,
    # надеясь на другой результат, и так расходует все круги впустую.
    executed: set[str] = set()

    # Всё добытое инструментами — пригодится, если круги закончатся.
    collected: list[str] = []

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        try:
            reply = await llm.complete(
                system=build_system_prompt(),
                messages=messages,
                tools=TOOL_SPECS,
            )
        except ToolCallSchemaError as exc:
            # Модель вызвала инструмент с неподходящими аргументами, и провайдер
            # отклонил вызов, не доведя его до нас. Роняли бы ответ целиком —
            # вместо этого объясняем промах и даём следующий круг на исправление.
            logger.warning("Круг %s: вызов отклонён по схеме", round_number)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Предыдущий вызов инструмента отклонён: аргументы не "
                        "соответствуют схеме. Неверный вызов: "
                        f"{exc.failed_generation[:300]}. "
                        "Вызови инструмент заново, указав ровно те поля, которые "
                        "описаны в его схеме, и ничего сверх них."
                    ),
                }
            )
            continue

        if not reply.wants_tools:
            return reply.text or "Модель вернула пустой ответ. Попробуй переформулировать."

        # Модель попросила действие. Её реплику обязательно кладём в переписку:
        # без неё результаты вызовов повиснут в воздухе и API отвергнет запрос.
        messages.append(_assistant_message(reply))

        for call in reply.tool_calls:
            key = _call_key(call.name, call.arguments)
            if key in executed:
                # Повтор того же вызова. Выполнять заново бессмысленно —
                # вместо результата отдаём прямое указание закончить.
                logger.info("Круг %s: повторный вызов %s — пропускаем", round_number, key)
                result = (
                    "Этот вызов уже выполнялся в рамках текущего сообщения, "
                    "результат был выше. Не повторяй его и не пробуй похожие "
                    "варианты: ответь пользователю по уже собранным данным, "
                    "а если их не хватает — честно скажи об этом и уточни детали."
                )
            else:
                executed.add(key)
                logger.info(
                    "Круг %s: вызываем %s(%s)", round_number, call.name, call.arguments
                )
                result = await toolbox.execute(call.name, call.arguments, ctx)
                collected.append(result)

            # Роль "tool" — ответ на конкретный вызов, связь по tool_call_id.
            messages.append(
                {"role": "tool", "tool_call_id": call.call_id, "content": result}
            )

    # Круги кончились, а модель всё просит инструменты. К этому моменту она
    # обычно уже собрала что-то полезное, и выбрасывать это — худшее, что можно
    # сделать: пользователь получал «запутался» вместо почти готового ответа.
    # Спрашиваем последний раз, но без инструментов: ответить текстом придётся.
    logger.warning(
        "Исчерпан лимит кругов (%s), просим ответить по собранным данным",
        MAX_TOOL_ROUNDS,
    )
    return await _answer_from_collected(llm, user_text, collected)


async def _answer_from_collected(
    llm: LLMClient,
    user_text: str,
    collected: list[str],
) -> str:
    """Просит модель ответить по уже добытым данным.

    Запрос собирается с нуля: только вопрос и собранные сведения, без описаний
    инструментов и без предыдущих вызовов в переписке. Это принципиально.
    Модели gpt-oss обучены со встроенным браузером, и, увидев в переписке
    цепочку вызовов, продолжают её даже когда инструменты не переданы —
    в логах это выглядело как попытка вызвать несуществующий browser.open,
    из-за чего провайдер отклонял запрос, а пользователь оставался без ответа.
    Чистый запрос такого повода не даёт.
    """
    fallback = "Я поискал, но не смог собрать внятный ответ. Попробуй сформулировать иначе."

    if not collected:
        return fallback

    facts = "\n\n".join(collected)[:8000]
    prompt = (
        f"Вопрос пользователя: {user_text}\n\n"
        f"Вот сведения, которые удалось собрать:\n\n{facts}\n\n"
        f"Ответь на вопрос, опираясь только на эти сведения. Если их не хватает "
        f"для точного ответа, скажи, что удалось выяснить, честно предупреди, "
        f"чего не хватает, и предложи, где посмотреть."
    )

    try:
        final = await llm.complete(
            system=build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
    except (ProviderError, ToolCallSchemaError):
        logger.exception("Не удалось получить ответ по собранным данным")
        return fallback

    return final.text or fallback
