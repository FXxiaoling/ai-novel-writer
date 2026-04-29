"""Orchestrator — the multi-agent pipeline coordinator.

Coordinates all agents through the novel writing pipeline:
  Concept → Outline → Characters → Chapter Writing → Dialogue Polish
  → Continuity Check → Review → (Auto-fix loop) → Publish

Each chapter goes through the full quality gate before proceeding.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .agents.base import BaseAgent
from .agents.outline_agent import OutlineAgent
from .agents.character_agent import CharacterAgent
from .agents.chapter_writer import ChapterWriterAgent
from .agents.dialogue_agent import DialogueAgent
from .agents.reviewer import ReviewAgent
from .agents.continuity_checker import ContinuityAgent
from .memory.memory_store import MemoryStore
from .models.schemas import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    Chapter,
    ChapterOutline,
    ChapterStatus,
    LLMConfig,
    NovelState,
    PipelineConfig,
    ReviewResult,
)
from .utils.io import FileHandler

logger = logging.getLogger(__name__)

console = Console()


class NovelWriterOrchestrator:
    """The master coordinator for the multi-agent novel writing pipeline.

    Architecture:
    ┌─────────────┐
    │ User Input   │── 创作概念
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Outline Agent │── 大纲规划
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Character Agt │── 角色设计
    └──────┬──────┘
           │
    ┌──────▼──────┐   ┌───────────────┐
    │ Chapter Writer│◄──│ Memory Store  │── 上下文注入
    └──────┬──────┘   └───────────────┘
           │
    ┌──────▼──────┐
    │ Dialogue Agt  │── 对话润色
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Continuity Agt│── 连贯性检查
    └──────┬──────┘
           │
    ┌──────▼──────┐    ┌── 不合格 → 返回 Chapter Writer (auto-fix)
    │ Review Agent │───┤
    └──────────────┘    └── 合格 → 下一章 / 完成
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = NovelState()
        self.io = FileHandler(config.output_dir)
        self.memory = MemoryStore(persist_dir="./chroma_db")
        self._context: dict = {}
        self._setup_agents()

    def _setup_agents(self):
        """Create all agents with their configured LLM settings."""
        model_map = {
            AgentRole.OUTLINE: self.config.models.get(AgentRole.OUTLINE, "gpt-4o"),
            AgentRole.CHARACTER: self.config.models.get(AgentRole.CHARACTER, "gpt-4o"),
            AgentRole.CHAPTER: self.config.models.get(AgentRole.CHAPTER, "gpt-4o"),
            AgentRole.DIALOGUE: self.config.models.get(AgentRole.DIALOGUE, "gpt-4o"),
            AgentRole.REVIEW: self.config.models.get(AgentRole.REVIEW, "gpt-4o-mini"),
            AgentRole.CONTINUITY: self.config.models.get(AgentRole.CONTINUITY, "gpt-4o-mini"),
        }

        temp_creative = 0.8
        temp_review = 0.3

        self.outline_agent = OutlineAgent(
            LLMConfig(model=model_map[AgentRole.OUTLINE], temperature=temp_creative),
            name="大纲Agent",
        )
        self.character_agent = CharacterAgent(
            LLMConfig(model=model_map[AgentRole.CHARACTER], temperature=temp_creative),
            name="角色Agent",
        )
        self.chapter_writer = ChapterWriterAgent(
            LLMConfig(model=model_map[AgentRole.CHAPTER], temperature=temp_creative),
            name="章节写作Agent",
        )
        self.dialogue_agent = DialogueAgent(
            LLMConfig(model=model_map[AgentRole.DIALOGUE], temperature=temp_creative),
            name="对话润色Agent",
        )
        self.review_agent = ReviewAgent(
            LLMConfig(model=model_map[AgentRole.REVIEW], temperature=temp_review),
            name="审核Agent",
        )
        self.continuity_agent = ContinuityAgent(
            LLMConfig(model=model_map[AgentRole.CONTINUITY], temperature=temp_review),
            name="连贯性检查Agent",
        )

    async def run(self, concept: str, total_chapters: int = 10) -> NovelState:
        """Execute the complete multi-agent novel writing pipeline.

        Args:
            concept: The creative concept / premise for the novel.
            total_chapters: Number of chapters to write.

        Returns:
            The complete NovelState with all chapters.
        """
        console.print(f"\n[bold cyan]═══ AI Multi-Agent Novel Writer ═══[/bold cyan]")
        console.print(f"[dim]概念: {concept[:80]}...[/dim]")
        console.print(f"[dim]目标章节数: {total_chapters}[/dim]\n")

        self.state.total_chapters = total_chapters

        # ── Stage 1: Outline ─────────────────────────────────
        await self._stage_outline(concept, total_chapters)

        # ── Stage 2: Character Design ─────────────────────────
        await self._stage_characters()

        # ── Stage 3: Chapter-by-Chapter Pipeline ──────────────
        for ch_num in range(1, total_chapters + 1):
            self.state.current_chapter = ch_num
            console.print(f"\n[bold yellow]━━━ 第{ch_num}章 ━━━[/bold yellow]")
            await self._write_chapter(ch_num)

            if ch_num % self.config.checkpoint_interval == 0:
                self.io.save_checkpoint(self.state, f"ch{ch_num}")

        # ── Stage 4: Export ───────────────────────────────────
        self._finalize()

        return self.state

    # ── Pipeline Stages ──────────────────────────────────────

    async def _stage_outline(self, concept: str, total_chapters: int):
        """Stage 1: Generate novel outline."""
        console.print("[green]► 阶段1: 大纲规划...[/green]")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task("大纲Agent 工作中...", total=None)

            result = await self.outline_agent.execute(
                AgentMessage(
                    from_role=AgentRole.CHAPTER,
                    to_role=AgentRole.OUTLINE,
                    content=concept,
                    metadata={
                        "total_chapters": total_chapters,
                        "min_chapter_words": self.config.min_chapter_words,
                        "max_chapter_words": self.config.max_chapter_words,
                        "style": self.config.style,
                        "genre": self.config.genre,
                    },
                )
            )
            progress.update(task, completed=True)

        if result.status != AgentStatus.FAILED:
            console.print("  [green]✓ 大纲生成完成[/green]")
        else:
            console.print(f"  [red]✗ 大纲生成失败: {result.metadata.get('error')}[/red]")

        self._context["outline_raw"] = result.content

    async def _stage_characters(self):
        """Stage 2: Design characters based on the outline."""
        console.print("[green]► 阶段2: 角色设计...[/green]")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task("角色Agent 工作中...", total=None)

            result = await self.character_agent.execute(
                AgentMessage(
                    from_role=AgentRole.OUTLINE,
                    to_role=AgentRole.CHARACTER,
                    content=self._context.get("outline_raw", ""),
                    metadata={
                        "style": self.config.style,
                        "genre": self.config.genre,
                    },
                )
            )
            progress.update(task, completed=True)

        if result.status != AgentStatus.FAILED:
            console.print("  [green]✓ 角色设计完成[/green]")
        else:
            console.print(f"  [red]✗ 角色设计失败: {result.metadata.get('error')}[/red]")

        self._context["characters_raw"] = result.content

    async def _write_chapter(self, chapter_num: int):
        """Write a single chapter through the full quality pipeline."""
        chapter_outline = self._get_chapter_outline(chapter_num)

        # Inject memory context
        previous_context = self.memory.get_context_window(chapter_num)
        character_states = self.memory.get_all_character_states(chapter_num)
        pending_foreshadowing = self.memory.get_pending_foreshadowing()

        # ── 3a: Write draft ──
        console.print("  [blue]├ 章节写作...[/blue]")
        chapter_result = await self.chapter_writer.execute(
            AgentMessage(
                from_role=AgentRole.CHARACTER,
                to_role=AgentRole.CHAPTER,
                content="",
                metadata={
                    "chapter_number": chapter_num,
                    "chapter_outline": chapter_outline,
                    "previous_context": previous_context,
                    "character_states": character_states,
                    "pending_foreshadowing": pending_foreshadowing,
                    "min_chapter_words": self.config.min_chapter_words,
                    "max_chapter_words": self.config.max_chapter_words,
                    "style": self.config.style,
                    "genre": self.config.genre,
                },
            )
        )

        if chapter_result.status == AgentStatus.FAILED:
            console.print(f"  [red]  章节写作失败[/red]")
            return

        raw_content = chapter_result.content
        word_count = len(raw_content)
        console.print(f"  [dim]  草稿字数: {word_count}[/dim]")

        # ── 3b: Dialogue polish ──
        console.print("  [blue]├ 对话润色...[/blue]")
        dialogue_result = await self.dialogue_agent.execute(
            AgentMessage(
                from_role=AgentRole.CHAPTER,
                to_role=AgentRole.DIALOGUE,
                content=raw_content,
                metadata={
                    "character_profiles": self._context.get("characters_raw", ""),
                },
            )
        )
        polished = dialogue_result.content if dialogue_result.status != AgentStatus.FAILED else raw_content

        # ── 3c: Continuity check ──
        console.print("  [blue]├ 连贯性检查...[/blue]")
        continuity_result = await self.continuity_agent.execute(
            AgentMessage(
                from_role=AgentRole.DIALOGUE,
                to_role=AgentRole.CONTINUITY,
                content=polished,
                metadata={
                    "chapter_number": chapter_num,
                    "previous_summary": previous_context,
                    "foreshadowing_list": self.memory.get_all_foreshadowing(),
                },
            )
        )
        console.print(f"  [dim]  连贯性: {continuity_result.content[:100]}...[/dim]")

        # ── 3d: Review ──
        console.print("  [blue]├ 质量审核...[/blue]")
        review_result = await self.review_agent.execute(
            AgentMessage(
                from_role=AgentRole.CONTINUITY,
                to_role=AgentRole.REVIEW,
                content=polished,
                metadata={
                    "chapter_number": chapter_num,
                    "chapter_title": f"第{chapter_num}章",
                    "quality_threshold": self.config.quality_threshold,
                },
            )
        )

        review_parsed = ReviewAgent.parse_review(review_result.content)
        score = review_parsed.overall_score
        passed = review_parsed.passed

        if passed:
            console.print(f"  [green]  ✓ 审核通过 (评分: {score:.1f}/10)[/green]")
        else:
            console.print(f"  [yellow]  ⚠ 审核未通过 (评分: {score:.1f}/10)[/yellow]")
            if review_parsed.issues:
                for issue in review_parsed.issues[:3]:
                    console.print(f"  [red]    - {issue}[/red]")

            # Auto-fix: re-generate with review feedback
            if self.config.auto_fix:
                await self._auto_fix(chapter_num, polished, review_parsed)

        # ── Save chapter ──
        chapter = Chapter(
            chapter_number=chapter_num,
            title=f"第{chapter_num}章",
            content=polished,
            word_count=word_count,
            status=ChapterStatus.APPROVED if passed else ChapterStatus.DRAFT,
        )
        self.state.chapters[chapter_num] = chapter
        self.io.save_chapter(chapter)

        # ── Update memory ──
        self.memory.add_chapter_summary(chapter_num, polished[:500])
        self.memory.index_chapter(chapter)

    async def _auto_fix(self, chapter_num: int, content: str, review: ReviewResult):
        """Attempt to auto-fix issues found by the review agent."""
        console.print("  [yellow]├ 自动修复中...[/yellow]")
        issues_text = "\n".join(review.issues)

        retry_count = 0
        while retry_count < self.config.max_retries:
            retry_count += 1
            fix_result = await self.chapter_writer.execute(
                AgentMessage(
                    from_role=AgentRole.REVIEW,
                    to_role=AgentRole.CHAPTER,
                    content=content,
                    metadata={
                        "chapter_number": chapter_num,
                        "writing_notes": f"请根据以下审核意见修改本章:\n{issues_text}",
                        "min_chapter_words": self.config.min_chapter_words,
                        "max_chapter_words": self.config.max_chapter_words,
                        "style": self.config.style,
                        "genre": self.config.genre,
                        "previous_context": "",
                        "character_states": "",
                        "pending_foreshadowing": "",
                        "chapter_outline": "",
                    },
                )
            )

            if fix_result.status == AgentStatus.FAILED:
                continue

            content = fix_result.content
            re_review = await self.review_agent.execute(
                AgentMessage(
                    from_role=AgentRole.CHAPTER,
                    to_role=AgentRole.REVIEW,
                    content=content,
                    metadata={
                        "chapter_number": chapter_num,
                        "quality_threshold": self.config.quality_threshold,
                    },
                )
            )
            re_parsed = ReviewAgent.parse_review(re_review.content)
            if re_parsed.passed:
                console.print(f"  [green]  ✓ 修复成功 (评分: {re_parsed.overall_score:.1f})[/green]")
                self.state.chapters[chapter_num] = Chapter(
                    chapter_number=chapter_num,
                    title=f"第{chapter_num}章",
                    content=content,
                    word_count=len(content),
                    status=ChapterStatus.APPROVED,
                    revision_history=review.issues,
                )
                return
            else:
                console.print(f"  [yellow]  第{retry_count}次修复仍未通过 (评分: {re_parsed.overall_score:.1f})[/yellow]")
                review = re_parsed

        console.print(f"  [red]  自动修复失败，章节标记为草稿[/red]")

    def _get_chapter_outline(self, chapter_num: int) -> str:
        """Get chapter outline from the stored outline data."""
        outline_raw = self._context.get("outline_raw", "")
        return f"请根据大纲编写第{chapter_num}章。\n\n大纲内容:\n{outline_raw}"

    def _finalize(self):
        """Export final novel and save state."""
        console.print(f"\n[bold green]═══ 小说写作完成 ═══[/bold green]")
        novel_path = self.io.export_full_novel(self.state)
        state_path = self.io.save_state(self.state)
        self.memory.save()

        total_words = sum(ch.word_count for ch in self.state.chapters.values())
        console.print(f"[green]总章节: {len(self.state.chapters)}[/green]")
        console.print(f"[green]总字数: {total_words:,}[/green]")
        console.print(f"[green]导出文件: {novel_path}[/green]")
        console.print(f"[green]状态文件: {state_path}[/green]")
