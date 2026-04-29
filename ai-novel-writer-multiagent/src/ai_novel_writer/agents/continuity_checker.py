"""Continuity Agent — checks plot consistency, timeline, and character continuity."""

from __future__ import annotations

from ..models.schemas import AgentRole, AgentMessage
from ..prompts.templates import CONTINUITY_SYSTEM, CONTINUITY_USER
from .base import BaseAgent


class ContinuityAgent(BaseAgent):
    """Ensures narrative consistency across chapters — the 'anti-plot-hole' agent."""

    role = AgentRole.CONTINUITY

    async def _process(self, input_message: AgentMessage) -> str:
        metadata = input_message.metadata

        current_chapter = input_message.content
        chapter_number = metadata.get("chapter_number", 1)
        previous_summary = metadata.get("previous_summary", "无前文（这是第一章）")
        foreshadowing_list = metadata.get("foreshadowing_list", "无")

        system, user = self._format_prompt(
            CONTINUITY_SYSTEM, CONTINUITY_USER,
            chapter_number=chapter_number,
            current_chapter=current_chapter,
            previous_summary=previous_summary,
            foreshadowing_list=foreshadowing_list,
        )

        response = await self.llm.agenerate(system, user)
        return response
