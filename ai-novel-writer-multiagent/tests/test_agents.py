"""Tests for the agent pipeline."""

import pytest
from unittest.mock import patch, AsyncMock

from ai_novel_writer.models.schemas import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    LLMConfig,
    NovelState,
    Character,
    Chapter,
    ReviewResult,
    ReviewDimension,
    ContinuityReport,
)
from ai_novel_writer.agents.outline_agent import OutlineAgent
from ai_novel_writer.agents.character_agent import CharacterAgent
from ai_novel_writer.agents.chapter_writer import ChapterWriterAgent
from ai_novel_writer.agents.dialogue_agent import DialogueAgent
from ai_novel_writer.agents.reviewer import ReviewAgent
from ai_novel_writer.agents.continuity_checker import ContinuityAgent


class TestOutlineAgent:
    async def test_basic_execution(self, llm_config):
        agent = OutlineAgent(llm_config)

        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "# Test Outline\n\n测试大纲内容..."
            result = await agent.execute(
                AgentMessage(
                    from_role=AgentRole.CHAPTER,
                    to_role=AgentRole.OUTLINE,
                    content="测试概念",
                    metadata={"total_chapters": 10},
                )
            )

        assert result.status == AgentStatus.SUCCESS
        assert "测试大纲" in result.content
        mock_gen.assert_called_once()


class TestCharacterAgent:
    async def test_basic_execution(self, llm_config):
        agent = CharacterAgent(llm_config)

        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "## 角色设计\n\n### 主角: 张三\n..."
            result = await agent.execute(
                AgentMessage(
                    from_role=AgentRole.OUTLINE,
                    to_role=AgentRole.CHARACTER,
                    content="大纲内容",
                    metadata={"style": "玄幻", "genre": "修仙"},
                )
            )

        assert result.status == AgentStatus.SUCCESS
        assert "张三" in result.content


class TestChapterWriter:
    async def test_basic_execution(self, llm_config):
        agent = ChapterWriterAgent(llm_config)

        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "正文内容" * 500  # ~2000 chars
            result = await agent.execute(
                AgentMessage(
                    from_role=AgentRole.CHARACTER,
                    to_role=AgentRole.CHAPTER,
                    content="",
                    metadata={
                        "chapter_number": 1,
                        "chapter_outline": "第一章大纲",
                        "previous_context": "无",
                        "min_chapter_words": 3000,
                        "max_chapter_words": 8000,
                    },
                )
            )

        assert result.status == AgentStatus.SUCCESS


class TestDialogueAgent:
    async def test_basic_execution(self, llm_config):
        agent = DialogueAgent(llm_config)

        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "润色后的内容..."
            result = await agent.execute(
                AgentMessage(
                    from_role=AgentRole.CHAPTER,
                    to_role=AgentRole.DIALOGUE,
                    content="原始内容",
                    metadata={"character_profiles": "角色列表"},
                )
            )

        assert result.status == AgentStatus.SUCCESS


class TestReviewAgent:
    async def test_basic_execution(self, llm_config):
        agent = ReviewAgent(llm_config)

        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "整体评分: 8.5\n情节: 8\n角色: 8"
            result = await agent.execute(
                AgentMessage(
                    from_role=AgentRole.CONTINUITY,
                    to_role=AgentRole.REVIEW,
                    content="章节内容",
                    metadata={"chapter_number": 1, "quality_threshold": 7.0},
                )
            )

        assert result.status == AgentStatus.SUCCESS

    def test_parse_review_pass(self):
        text = "整体评分: 8.5\n情节: 8\n角色: 9\n建议: 可以更好"
        result = ReviewAgent.parse_review(text)
        assert result.overall_score == 8.5
        assert result.passed is True

    def test_parse_review_fail(self):
        text = "整体评分: 5.0\n情节: 4\n角色: 5"
        result = ReviewAgent.parse_review(text)
        assert result.overall_score == 5.0
        assert result.passed is False


class TestContinuityAgent:
    async def test_basic_execution(self, llm_config):
        agent = ContinuityAgent(llm_config)

        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "未发现矛盾，连贯性良好"
            result = await agent.execute(
                AgentMessage(
                    from_role=AgentRole.DIALOGUE,
                    to_role=AgentRole.CONTINUITY,
                    content="章节内容",
                    metadata={
                        "chapter_number": 3,
                        "previous_summary": "前文摘要",
                        "foreshadowing_list": "伏笔1",
                    },
                )
            )

        assert result.status == AgentStatus.SUCCESS


class TestAgentRetry:
    async def test_retry_on_failure(self, llm_config):
        """Agent should retry on failure and eventually fail."""
        llm_config.temperature = 0.3
        agent = ReviewAgent(llm_config)
        agent.max_retries = 2

        with patch.object(agent.llm, 'agenerate', new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("API Error")
            result = await agent.execute(
                AgentMessage(
                    from_role=AgentRole.CONTINUITY,
                    to_role=AgentRole.REVIEW,
                    content="内容",
                    metadata={},
                )
            )

        assert result.status == AgentStatus.FAILED
        assert mock_gen.call_count == 3  # initial + 2 retries
