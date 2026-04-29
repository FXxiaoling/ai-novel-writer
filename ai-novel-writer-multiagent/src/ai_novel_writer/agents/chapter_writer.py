"""Chapter Writer Agent — writes novel chapters with context awareness."""

from __future__ import annotations

from ..models.schemas import AgentRole, AgentMessage
from ..prompts.templates import CHAPTER_SYSTEM, CHAPTER_USER
from .base import BaseAgent


class ChapterWriterAgent(BaseAgent):
    """Writes individual novel chapters with full context injection."""

    role = AgentRole.CHAPTER

    async def _process(self, input_message: AgentMessage) -> str:
        metadata = input_message.metadata

        chapter_number = metadata.get("chapter_number", 1)
        chapter_outline = metadata.get("chapter_outline", "")
        previous_context = metadata.get("previous_context", "无（这是第一章）")
        character_states = metadata.get("character_states", "")
        pending_foreshadowing = metadata.get("pending_foreshadowing", "无")
        writing_notes = metadata.get("writing_notes", "")
        min_words = metadata.get("min_chapter_words", 3000)
        max_words = metadata.get("max_chapter_words", 8000)
        style = metadata.get("style", "")
        genre = metadata.get("genre", "")

        system, user = self._format_prompt(
            CHAPTER_SYSTEM, CHAPTER_USER,
            style=style,
            genre=genre,
            chapter_number=chapter_number,
            chapter_outline=chapter_outline,
            previous_context=previous_context,
            character_states=character_states,
            pending_foreshadowing=pending_foreshadowing,
            writing_notes=writing_notes,
            min_words=min_words,
            max_words=max_words,
        )

        response = await self.llm.agenerate(system, user)
        return response
