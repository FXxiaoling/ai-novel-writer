"""Outline Agent — generates novel structure, plot tree, and chapter outlines."""

from __future__ import annotations

from ..models.schemas import AgentRole, AgentMessage, AgentStatus
from ..prompts.templates import OUTLINE_SYSTEM, OUTLINE_USER
from .base import BaseAgent


class OutlineAgent(BaseAgent):
    """Generates the complete novel outline including plot tree and chapter plans."""

    role = AgentRole.OUTLINE

    async def _process(self, input_message: AgentMessage) -> str:
        concept = input_message.content
        metadata = input_message.metadata

        total_chapters = metadata.get("total_chapters", 10)
        min_words = metadata.get("min_chapter_words", 3000)
        max_words = metadata.get("max_chapter_words", 6000)
        style = metadata.get("style", "")
        genre = metadata.get("genre", "")

        system, user = self._format_prompt(
            OUTLINE_SYSTEM, OUTLINE_USER,
            style=style,
            genre=genre,
            concept=concept,
            total_chapters=total_chapters,
            min_words=min_words,
            max_words=max_words,
        )

        response = await self.llm.agenerate(system, user)
        return response
