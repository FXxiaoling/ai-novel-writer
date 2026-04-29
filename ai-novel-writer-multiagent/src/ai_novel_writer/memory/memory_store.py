"""Novel Memory System — tracks context, characters, and plot state across chapters.

Uses both a structured JSON store for precise recall and an optional
ChromaDB vector store for semantic search of past chapters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..models.schemas import Chapter, Character, NovelOutline, NovelState

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class MemoryStore:
    """Central memory manager for the novel writing pipeline.

    Tracks:
    - Character state at each chapter boundary
    - Unresolved foreshadowing
    - Recent chapter summaries for context window
    - Key plot events timeline
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = Path(persist_dir)
        self._character_snapshots: dict[str, list[dict]] = {}
        self._foreshadowing: list[dict] = []
        self._plot_timeline: list[dict] = []
        self._chapter_summaries: dict[int, str] = {}
        self._vector_client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[object] = None

        if HAS_CHROMADB:
            self._init_vector_store()

    def _init_vector_store(self):
        """Initialize ChromaDB for semantic memory retrieval."""
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._vector_client = chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            self._collection = self._vector_client.get_or_create_collection(
                name="novel_context"
            )
        except Exception:
            self._vector_client = None
            self._collection = None

    # ── Context Window Management ───────────────────────────

    def get_context_window(
        self, current_chapter: int, window_size: int = 5
    ) -> str:
        """Build a context window of recent chapter summaries."""
        parts = []
        for ch in range(max(1, current_chapter - window_size), current_chapter):
            if ch in self._chapter_summaries:
                parts.append(f"第{ch}章摘要: {self._chapter_summaries[ch]}")
        return "\n\n".join(parts)

    def add_chapter_summary(self, chapter_num: int, summary: str):
        """Store a chapter summary for context retrieval."""
        self._chapter_summaries[chapter_num] = summary

    # ── Character State Tracking ────────────────────────────

    def snapshot_character(self, chapter_num: int, character: Character):
        """Record a character's state at a given chapter."""
        state = {
            "chapter": chapter_num,
            "name": character.name,
            "status": character.personality,
            "location": "",
            "condition": "",
            "relationships": dict(character.relationships),
        }
        if character.name not in self._character_snapshots:
            self._character_snapshots[character.name] = []
        self._character_snapshots[character.name].append(state)

    def get_character_history(self, name: str) -> list[dict]:
        """Get a character's state history across chapters."""
        return self._character_snapshots.get(name, [])

    def get_all_character_states(self, at_chapter: int) -> str:
        """Get all character states at or before a given chapter."""
        lines = []
        for name, snapshots in self._character_snapshots.items():
            relevant = [s for s in snapshots if s["chapter"] < at_chapter]
            if relevant:
                latest = relevant[-1]
                lines.append(
                    f"{name}: {latest['status']} - {latest['condition']} "
                    f"(截至第{latest['chapter']}章)"
                )
        return "\n".join(lines) if lines else "暂无角色状态记录"

    # ── Foreshadowing Tracking ──────────────────────────────

    def add_foreshadowing(
        self, content: str, planted_in: int, resolved_in: Optional[int] = None
    ):
        """Track a foreshadowing element."""
        self._foreshadowing.append({
            "content": content,
            "planted_in": planted_in,
            "resolved_in": resolved_in,
        })

    def get_pending_foreshadowing(self) -> str:
        """Get all unresolved foreshadowing elements."""
        pending = [
            f["content"] for f in self._foreshadowing if f["resolved_in"] is None
        ]
        return "\n".join(f"- {p}" for p in pending) if pending else "无待回收伏笔"

    def get_all_foreshadowing(self) -> str:
        """Get full foreshadowing list with status."""
        lines = []
        for f in self._foreshadowing:
            status = f"已回收(第{f['resolved_in']}章)" if f["resolved_in"] else "待回收"
            lines.append(f"- [{status}] {f['content']} (埋于第{f['planted_in']}章)")
        return "\n".join(lines) if lines else "无伏笔"

    def resolve_foreshadowing(self, content_pattern: str, chapter_num: int):
        """Mark matching foreshadowing as resolved."""
        for f in self._foreshadowing:
            if content_pattern in f["content"] and f["resolved_in"] is None:
                f["resolved_in"] = chapter_num

    # ── Plot Timeline ───────────────────────────────────────

    def add_plot_event(self, chapter: int, event: str, importance: str = "normal"):
        """Record a plot event."""
        self._plot_timeline.append({
            "chapter": chapter,
            "event": event,
            "importance": importance,
        })

    def get_timeline(self) -> str:
        """Get full plot timeline."""
        lines = []
        for e in sorted(self._plot_timeline, key=lambda x: x["chapter"]):
            lines.append(f"第{e['chapter']}章: {e['event']}")
        return "\n".join(lines) if lines else "暂无时间线"

    # ── Semantic Search ─────────────────────────────────────

    def index_chapter(self, chapter: Chapter):
        """Index chapter content into the vector store."""
        if not self._collection or not HAS_CHROMADB:
            return
        chunks = self._split_text(chapter.content, max_chars=1000)
        for i, chunk in enumerate(chunks):
            self._collection.add(
                documents=[chunk],
                metadatas=[{
                    "chapter": chapter.chapter_number,
                    "chunk_index": i,
                    "title": chapter.title,
                }],
                ids=[f"ch{chapter.chapter_number}_chunk{i}"],
            )

    def search_context(self, query: str, n_results: int = 5) -> str:
        """Semantically search past chapters for relevant context."""
        if not self._collection or not HAS_CHROMADB:
            return ""
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        docs = results.get("documents", [[]])[0]
        return "\n---\n".join(docs) if docs else ""

    @staticmethod
    def _split_text(text: str, max_chars: int = 1000) -> list[str]:
        """Split text into chunks for indexing."""
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i + max_chars])
        return chunks

    # ── Persistence ─────────────────────────────────────────

    def save(self, path: str = "memory_state.json"):
        """Save memory state to disk."""
        data = {
            "character_snapshots": self._character_snapshots,
            "foreshadowing": self._foreshadowing,
            "plot_timeline": self._plot_timeline,
            "chapter_summaries": {
                str(k): v for k, v in self._chapter_summaries.items()
            },
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str = "memory_state.json"):
        """Load memory state from disk."""
        filepath = Path(path)
        if not filepath.exists():
            return
        data = json.loads(filepath.read_text(encoding="utf-8"))
        self._character_snapshots = data.get("character_snapshots", {})
        self._foreshadowing = data.get("foreshadowing", [])
        self._plot_timeline = data.get("plot_timeline", [])
        self._chapter_summaries = {
            int(k): v for k, v in data.get("chapter_summaries", {}).items()
        }
