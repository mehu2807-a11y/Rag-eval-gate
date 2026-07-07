"""LLM answer generation given retrieved context. Supports Anthropic or
OpenAI as the backing provider, switched via app.config.LLM_PROVIDER."""
from typing import List, Dict

from app import config

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the "
    "provided context. If the context does not contain the answer, say "
    "you don't know based on the available documents. Do not invent facts "
    "that aren't supported by the context. Cite which source(s) you used."
)


def _build_prompt(question: str, contexts: List[Dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']} | chunk {c['chunk_index']}]\n{c['text']}"
        for c in contexts
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def _generate_anthropic(question: str, contexts: List[Dict]) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(question, contexts)}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


def _generate_openai(question: str, contexts: List[Dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(question, contexts)},
        ],
        max_tokens=800,
    )
    return resp.choices[0].message.content


def generate_answer(question: str, contexts: List[Dict]) -> str:
    if not contexts:
        return "I don't have any indexed documents to answer this from yet."
    if config.LLM_PROVIDER == "anthropic":
        return _generate_anthropic(question, contexts)
    elif config.LLM_PROVIDER == "openai":
        return _generate_openai(question, contexts)
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")
