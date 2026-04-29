"""Base Agent class — foundation for all specialized agents in the pipeline."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..models.schemas import AgentRole, AgentStatus, AgentMessage, LLMConfig
from ..utils.llm import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all pipeline agents.

    Each agent has:
    - A role (AgentRole enum)
    - An LLM client with configurable model/temperature
    - State tracking (status, retry count)
    - Standardized input/output message protocol
    """

    role: AgentRole

    def __init__(self, llm_config: LLMConfig, name: str = ""):
        self.name = name or self.role.value
        self.llm = LLMClient(llm_config)
        self.status: AgentStatus = AgentStatus.IDLE
        self.retry_count: int = 0
        self.max_retries: int = 3
        self.context: dict = {}

    def set_context(self, **kwargs):
        """Set shared context accessible to the agent."""
        self.context.update(kwargs)

    async def execute(self, input_message: AgentMessage) -> AgentMessage:
        """Execute the agent's task with retry logic.

        Returns an AgentMessage with the result.
        """
        self.status = AgentStatus.RUNNING
        self.retry_count = 0

        while self.retry_count <= self.max_retries:
            try:
                result = await self._process(input_message)
                self.status = AgentStatus.SUCCESS
                return AgentMessage(
                    from_role=self.role,
                    to_role=input_message.from_role,
                    content=result,
                    metadata={"retries": self.retry_count},
                    status=AgentStatus.SUCCESS,
                )
            except Exception as e:
                self.retry_count += 1
                logger.warning(
                    f"[{self.name}] 第{self.retry_count}次重试: {e}"
                )
                if self.retry_count > self.max_retries:
                    self.status = AgentStatus.FAILED
                    return AgentMessage(
                        from_role=self.role,
                        to_role=input_message.from_role,
                        content="",
                        metadata={"error": str(e), "retries": self.retry_count},
                        status=AgentStatus.FAILED,
                    )
                await asyncio.sleep(2 ** self.retry_count)

        self.status = AgentStatus.FAILED
        return AgentMessage(
            from_role=self.role,
            to_role=input_message.from_role,
            content="",
            metadata={"error": "max retries exceeded"},
            status=AgentStatus.FAILED,
        )

    @abstractmethod
    async def _process(self, input_message: AgentMessage) -> str:
        """Core processing logic — implemented by each agent subclass.

        Args:
            input_message: Message from upstream agent containing instructions.

        Returns:
            String response from the agent.
        """
        ...

    def _format_prompt(self, system: str, user: str, **kwargs) -> tuple[str, str]:
        """Format prompt templates with keyword arguments."""
        return system.format(**kwargs), user.format(**kwargs)
