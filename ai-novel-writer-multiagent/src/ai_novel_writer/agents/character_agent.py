"""Character Agent — designs and manages novel characters."""

from __future__ import annotations

from ..models.schemas import AgentRole, AgentMessage
from ..prompts.templates import CHARACTER_SYSTEM, CHARACTER_USER
from .base import BaseAgent


class CharacterAgent(BaseAgent):
    """Designs characters based on the novel outline, including arcs and relationships."""

    role = AgentRole.CHARACTER

    async def _process(self, input_message: AgentMessage) -> str:
        outline_text = input_message.content
        metadata = input_message.metadata

        style = metadata.get("style", "")
        genre = metadata.get("genre", "")

        system, user = self._format_prompt(
            CHARACTER_SYSTEM, CHARACTER_USER,
            style=style,
            genre=genre,
            outline_text=outline_text,
        )

        response = await self.llm.agenerate(system, user)
        return response
