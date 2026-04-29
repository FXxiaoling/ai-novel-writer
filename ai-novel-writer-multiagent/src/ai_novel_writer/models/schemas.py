"""Pydantic data models for the novel writing pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    OUTLINE = "outline"
    CHARACTER = "character"
    CHAPTER = "chapter"
    DIALOGUE = "dialogue"
    REVIEW = "review"
    CONTINUITY = "continuity"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    RETRY = "retry"
    FAILED = "failed"


class ChapterStatus(str, Enum):
    DRAFT = "draft"
    REVIEWING = "reviewing"
    REVISING = "revising"
    APPROVED = "approved"
    PUBLISHED = "published"


class ReviewDimension(str, Enum):
    PLOT_HOLES = "plot_holes"
    CHARACTER_CONSISTENCY = "character_consistency"
    PACING = "pacing"
    DIALOGUE_QUALITY = "dialogue_quality"
    GRAMMAR = "grammar"


# ── Novel Structure ──────────────────────────────────────────


class Character(BaseModel):
    """A character in the novel."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    role: str = ""  # 主角/配角/反派
    gender: str = ""
    age: str = ""
    appearance: str = ""
    personality: str = ""
    background: str = ""
    abilities: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)
    motivations: str = ""
    arc_summary: str = ""
    notes: str = ""


class PlotNode(BaseModel):
    """A node in the plot outline."""

    id: str
    title: str
    summary: str
    key_events: list[str] = Field(default_factory=list)
    characters_involved: list[str] = Field(default_factory=list)
    emotional_tone: str = ""
    foreshadowing: list[str] = Field(default_factory=list)
    children: list[PlotNode] = Field(default_factory=list)


class ChapterOutline(BaseModel):
    """Outline for a single chapter."""

    chapter_number: int
    title: str = ""
    summary: str = ""
    scenes: list[str] = Field(default_factory=list)
    characters_featured: list[str] = Field(default_factory=list)
    plot_points: list[str] = Field(default_factory=list)
    word_count_target: int = 3000


class Chapter(BaseModel):
    """A completed chapter."""

    chapter_number: int
    title: str
    content: str
    outline: Optional[ChapterOutline] = None
    word_count: int = 0
    status: ChapterStatus = ChapterStatus.DRAFT
    revision_history: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class NovelOutline(BaseModel):
    """Complete novel outline."""

    title: str
    author: str = ""
    genre: str = ""
    style: str = ""
    one_liner: str = ""
    synopsis: str = ""
    world_building: str = ""
    themes: list[str] = Field(default_factory=list)
    plot_tree: list[PlotNode] = Field(default_factory=list)
    chapter_outlines: list[ChapterOutline] = Field(default_factory=list)
    total_chapters: int = 0


class NovelState(BaseModel):
    """Full state of the novel writing project."""

    outline: Optional[NovelOutline] = None
    characters: dict[str, Character] = Field(default_factory=dict)
    chapters: dict[int, Chapter] = Field(default_factory=dict)
    current_chapter: int = 0
    total_chapters: int = 0


# ── Agent Communication ───────────────────────────────────────


class AgentMessage(BaseModel):
    """Message passed between agents in the pipeline."""

    from_role: AgentRole
    to_role: AgentRole
    content: str
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    status: AgentStatus = AgentStatus.IDLE


class ReviewResult(BaseModel):
    """Result of a review agent's analysis."""

    overall_score: float = Field(ge=0.0, le=10.0)
    dimensions: dict[ReviewDimension, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    passed: bool = False
    summary: str = ""


class ContinuityReport(BaseModel):
    """Report from the continuity checker agent."""

    is_consistent: bool = True
    contradictions: list[str] = Field(default_factory=list)
    character_timeline_issues: list[str] = Field(default_factory=list)
    unresolved_foreshadowing: list[str] = Field(default_factory=list)
    notes: str = ""


# ── LLM Interaction ───────────────────────────────────────────


class LLMConfig(BaseModel):
    """Configuration for an LLM call."""

    provider: str = "openai"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class PipelineConfig(BaseModel):
    """Global pipeline configuration."""

    project_name: str = "my-novel"
    output_dir: str = "./output"
    language: str = "zh"
    min_chapter_words: int = 3000
    max_chapter_words: int = 8000
    style: str = ""
    genre: str = ""
    auto_fix: bool = True
    max_retries: int = 3
    quality_threshold: float = 7.0
    checkpoint_interval: int = 5
    models: dict[AgentRole, str] = Field(default_factory=dict)
