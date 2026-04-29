"""LLM client wrapper — unified interface for OpenAI-compatible APIs."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from ..models.schemas import LLMConfig

load_dotenv()


class LLMClient:
    """Unified client for any OpenAI-compatible LLM provider.

    Supports: OpenAI, DeepSeek, Qwen, Ollama, and any custom endpoint.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = config.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        self._sync_client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Synchronous generation."""
        response = self._sync_client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        return response.choices[0].message.content or ""

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str:
        """Asynchronous generation."""
        response = await self._async_client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        return response.choices[0].message.content or ""

    def generate_with_history(
        self, messages: list[dict[str, str]], system_prompt: str = ""
    ) -> str:
        """Generate with full conversation history."""
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = self._sync_client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,  # type: ignore[arg-type]
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def agenerate_with_history(
        self, messages: list[dict[str, str]], system_prompt: str = ""
    ) -> str:
        """Async generate with full conversation history."""
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = await self._async_client.chat.completions.create(
            model=self.config.model,
            messages=full_messages,  # type: ignore[arg-type]
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate using tiktoken."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 2

    @classmethod
    def from_env(cls, model: str = "gpt-4o", temperature: float = 0.7) -> LLMClient:
        """Create client from environment variables."""
        return cls(
            LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=model,
                temperature=temperature,
            )
        )
