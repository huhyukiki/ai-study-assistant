from __future__ import annotations

from typing import Protocol

from src.retrieval import SearchResult


class ResponsesClient(Protocol):
    class Responses(Protocol):
        def create(self, *, model: str, instructions: str, input: str): ...

    responses: Responses


SYSTEM_INSTRUCTIONS = """你是一名严谨、友好的 AI 学习助教。
只根据用户提供的资料片段回答。资料不足时要明确说明，不要猜测。
引用事实时使用 [来源 N · 第 X 页] 格式，其中 N 对应上下文编号。
先直接回答问题，再在必要时补充条目。"""


def answer_question(
    client: ResponsesClient,
    question: str,
    results: list[SearchResult],
    model: str,
) -> str:
    context = _format_context(results)
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=f"资料片段：\n{context}\n\n用户问题：{question}",
    )
    return response.output_text


def generate_quiz(
    client: ResponsesClient,
    results: list[SearchResult],
    model: str,
    question_count: int = 5,
) -> str:
    context = _format_context(results)
    response = client.responses.create(
        model=model,
        instructions=(
            SYSTEM_INSTRUCTIONS
            + "\n请生成适合复习的题目，并在最后单独列出答案与简短解析。"
        ),
        input=(
            f"根据以下资料生成 {question_count} 道题，混合单选题和简答题。"
            f"\n\n资料片段：\n{context}"
        ),
    )
    return response.output_text


def _format_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[来源 {index} · 第 {result.chunk.page} 页]\n{result.chunk.text}"
        for index, result in enumerate(results, start=1)
    )

