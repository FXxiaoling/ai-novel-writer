"""Tests for the orchestrator pipeline."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from ai_novel_writer.models.schemas import PipelineConfig, AgentRole, AgentMessage, AgentStatus
from ai_novel_writer.orchestrator import NovelWriterOrchestrator


@pytest.fixture
def pipeline_config():
    return PipelineConfig(
        project_name="test-novel",
        output_dir="./test_output",
        genre="玄幻",
        style="网络小说",
        min_chapter_words=1000,
        max_chapter_words=2000,
        quality_threshold=6.0,
    )


class TestOrchestrator:
    def test_setup_agents(self, pipeline_config):
        orchestrator = NovelWriterOrchestrator(pipeline_config)

        assert orchestrator.outline_agent is not None
        assert orchestrator.character_agent is not None
        assert orchestrator.chapter_writer is not None
        assert orchestrator.dialogue_agent is not None
        assert orchestrator.review_agent is not None
        assert orchestrator.continuity_agent is not None

    def test_initial_state(self, pipeline_config):
        orchestrator = NovelWriterOrchestrator(pipeline_config)

        assert orchestrator.state.current_chapter == 0
        assert orchestrator.state.total_chapters == 0
        assert len(orchestrator.state.chapters) == 0

    @pytest.mark.asyncio
    async def test_run_basic_flow(self, pipeline_config):
        """Test the orchestrator can run the full pipeline."""
        orchestrator = NovelWriterOrchestrator(pipeline_config)

        with patch.object(orchestrator.outline_agent.llm, 'agenerate',
                          new_callable=AsyncMock) as mock_outline, \
             patch.object(orchestrator.character_agent.llm, 'agenerate',
                          new_callable=AsyncMock) as mock_char, \
             patch.object(orchestrator.chapter_writer.llm, 'agenerate',
                          new_callable=AsyncMock) as mock_chapter, \
             patch.object(orchestrator.dialogue_agent.llm, 'agenerate',
                          new_callable=AsyncMock) as mock_dialogue, \
             patch.object(orchestrator.review_agent.llm, 'agenerate',
                          new_callable=AsyncMock) as mock_review, \
             patch.object(orchestrator.continuity_agent.llm, 'agenerate',
                          new_callable=AsyncMock) as mock_continuity, \
             patch.object(orchestrator.memory, 'index_chapter',
                          new_callable=MagicMock) as mock_index:

            mock_outline.return_value = "大纲内容..."
            mock_char.return_value = "角色: 张三..."
            mock_chapter.return_value = "正文内容" * 50
            mock_dialogue.return_value = "润色后正文" * 50
            mock_continuity.return_value = "连贯性正常"
            mock_review.return_value = "整体评分: 8.5\n章节通过审核"

            state = await orchestrator.run("测试概念", total_chapters=1)

            assert len(state.chapters) == 1
            assert state.chapters[1].word_count > 0
