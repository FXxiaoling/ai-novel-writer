"""Dialogue Agent — polishes dialogue and scene descriptions."""

from __future__ import annotations

from ..models.schemas import AgentRole, AgentMessage
from ..prompts.templates import DIALOGUE_SYSTEM, DIALOGUE_USER
from .base import BaseAgent


class DialogueAgent(BaseAgent):
    """Polishes dialogue and enhances scene descriptions without altering plot."""

    role = AgentRole.DIALOGUE

    async def _process(self, input_message: AgentMessage) -> str:
        metadata = input_message.metadata

        chapter_content = input_message.content
        character_profiles = metadata.get("character_profiles", "")

        system, user = self._format_prompt(
            DIALOGUE_SYSTEM, DIALOGUE_USER,
            character_profiles=character_profiles,
            chapter_content=chapter_content,
        )

        response = await self.llm.agenerate(system, user)
        return response
