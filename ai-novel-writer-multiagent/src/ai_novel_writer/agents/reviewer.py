"""Review Agent — evaluates chapter quality across multiple dimensions."""

from __future__ import annotations

import re

from ..models.schemas import AgentRole, AgentMessage, ReviewResult, ReviewDimension
from ..prompts.templates import REVIEW_SYSTEM, REVIEW_USER
from .base import BaseAgent


class ReviewAgent(BaseAgent):
    """Evaluates chapter quality and provides actionable feedback."""

    role = AgentRole.REVIEW

    async def _process(self, input_message: AgentMessage) -> str:
        metadata = input_message.metadata

        chapter_content = input_message.content
        chapter_number = metadata.get("chapter_number", 1)
        chapter_title = metadata.get("chapter_title", "")
        threshold = metadata.get("quality_threshold", 7.0)

        system, user = self._format_prompt(
            REVIEW_SYSTEM, REVIEW_USER,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            chapter_content=chapter_content,
            threshold=threshold,
        )

        response = await self.llm.agenerate(system, user)
        return response

    @staticmethod
    def parse_review(text: str) -> ReviewResult:
        """Parse the review output into a structured result."""
        overall = 5.0
        dimensions: dict[ReviewDimension, float] = {}
        issues: list[str] = []
        suggestions: list[str] = []

        score_match = re.search(r"整体[^\d]*(\d+(?:\.\d+)?)", text)
        if score_match:
            overall = min(10.0, max(0.0, float(score_match.group(1))))

        dim_map = {
            "情节": ReviewDimension.PLOT_HOLES,
            "角色": ReviewDimension.CHARACTER_CONSISTENCY,
            "节奏": ReviewDimension.PACING,
            "对话": ReviewDimension.DIALOGUE_QUALITY,
            "文笔": ReviewDimension.GRAMMAR,
        }

        for label, dim in dim_map.items():
            match = re.search(rf"{label}[^\d]*(\d+(?:\.\d+)?)", text)
            if match:
                dimensions[dim] = min(10.0, max(0.0, float(match.group(1))))

        issues_found = False
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r"^[\d\.、\-\*]+.*(?:问题|矛盾|漏洞|不足|穿帮)", line):
                issues.append(line)
                issues_found = True

        return ReviewResult(
            overall_score=overall,
            dimensions=dimensions,
            issues=issues,
            suggestions=suggestions,
            passed=overall >= 7.0,
            summary=text,
        )
