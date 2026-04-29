"""File I/O utilities — save, load, export novel data."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from ..models.schemas import Chapter, Character, NovelOutline, NovelState


class FileHandler:
    """Handles reading, writing, and exporting novel project files."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: NovelState, filename: str = "novel_state.json") -> Path:
        """Save full novel state to JSON."""
        path = self.output_dir / filename
        data = state.model_dump(mode="json", exclude_none=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_state(self, filename: str = "novel_state.json") -> Optional[NovelState]:
        """Load novel state from JSON."""
        path = self.output_dir / filename
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return NovelState(**data)

    def save_chapter(self, chapter: Chapter) -> Path:
        """Save a single chapter as Markdown."""
        ch_dir = self.output_dir / "chapters"
        ch_dir.mkdir(exist_ok=True)
        filename = f"ch_{chapter.chapter_number:04d}.md"
        path = ch_dir / filename

        lines = [
            f"# 第{chapter.chapter_number}章 {chapter.title}",
            "",
            chapter.content,
            "",
            "---",
            f"字数: {chapter.word_count}",
            f"状态: {chapter.status.value}",
            f"创建: {chapter.created_at.isoformat()}",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def export_full_novel(self, state: NovelState, filename: str = "novel.md") -> Path:
        """Export the full novel as a single Markdown file."""
        path = self.output_dir / filename
        outline = state.outline
        lines = [
            f"# {outline.title if outline else '未命名小说'}",
            "",
            f"> {outline.synopsis if outline else ''}",
            "",
            "---",
            "",
        ]

        for i in sorted(state.chapters.keys()):
            ch = state.chapters[i]
            lines.append(f"# 第{ch.chapter_number}章 {ch.title}")
            lines.append("")
            lines.append(ch.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def save_checkpoint(self, state: NovelState, label: str = "") -> Path:
        """Save a checkpoint snapshot."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{label}" if label else ""
        return self.save_state(state, f"checkpoint_{ts}{tag}.json")

    def load_config(self, path: str) -> dict:
        """Load YAML config file."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
