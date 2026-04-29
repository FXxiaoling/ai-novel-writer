"""Test fixtures and helpers."""

import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    mock = MagicMock()
    mock.choices = [
        MagicMock(message=MagicMock(content="Test response content"))
    ]
    return mock


@pytest.fixture
def llm_config():
    """Create a test LLM config."""
    from ai_novel_writer.models.schemas import LLMConfig
    return LLMConfig(
        provider="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        temperature=0.7,
    )


@pytest.fixture
def pipeline_config():
    """Create a test pipeline config."""
    from ai_novel_writer.models.schemas import PipelineConfig
    return PipelineConfig(
        project_name="test-novel",
        output_dir="./test_output",
        genre="玄幻",
        style="网络小说",
        min_chapter_words=1000,
        max_chapter_words=3000,
        quality_threshold=6.0,
    )
