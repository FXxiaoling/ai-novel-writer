"""CLI entry point for the AI Novel Writer.

Usage:
    novel-writer new "我的创作概念" -c 20 -o ./my_novel
    novel-writer continue ./my_novel
    novel-writer export ./my_novel
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .models.schemas import AgentRole, PipelineConfig
from .orchestrator import NovelWriterOrchestrator

load_dotenv()
console = Console()


def _build_config(
    project_name: str,
    output_dir: str,
    genre: str,
    style: str,
    total_chapters: int,
    min_words: int,
    max_words: int,
    threshold: float,
    language: str,
    auto_fix: bool,
    config_file: str = "",
) -> PipelineConfig:
    """Build PipelineConfig from CLI args and optional YAML file."""
    models: dict[AgentRole, str] = {
        AgentRole.OUTLINE: os.getenv("MODEL_OUTLINE", "gpt-4o"),
        AgentRole.CHARACTER: os.getenv("MODEL_CHARACTER", "gpt-4o"),
        AgentRole.CHAPTER: os.getenv("MODEL_CHAPTER", "gpt-4o"),
        AgentRole.DIALOGUE: os.getenv("MODEL_DIALOGUE", "gpt-4o"),
        AgentRole.REVIEW: os.getenv("MODEL_REVIEW", "gpt-4o-mini"),
        AgentRole.CONTINUITY: os.getenv("MODEL_CONTINUITY", "gpt-4o-mini"),
    }

    if config_file and Path(config_file).exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        project = data.get("project", {})
        pipeline = data.get("pipeline", {})
        gen = data.get("generation", {})
        review_cfg = data.get("review", {})

        project_name = project.get("name", project_name)
        output_dir = project.get("output_dir", output_dir)
        genre = gen.get("genre", genre)
        style = gen.get("style", style)
        min_words = gen.get("min_chapter_words", min_words)
        max_words = gen.get("max_chapter_words", max_words)
        threshold = review_cfg.get("quality_threshold", threshold)
        auto_fix = pipeline.get("auto_fix", auto_fix)
        language = project.get("language", language)

    return PipelineConfig(
        project_name=project_name,
        output_dir=output_dir,
        language=language,
        genre=genre,
        style=style,
        min_chapter_words=min_words,
        max_chapter_words=max_words,
        quality_threshold=threshold,
        auto_fix=auto_fix,
        models=models,
    )


@click.group()
@click.version_option(version="1.0.0", prog_name="ai-novel-writer")
def main():
    """AI Novel Writer — 多Agent AI小说写作流水线

    使用多个专业AI Agent协同工作，从概念到完整小说，
    自动完成大纲、角色、写作、润色、审核的完整流程。
    """
    pass


@main.command()
@click.argument("concept")
@click.option("-c", "--chapters", default=10, type=int, help="总章节数 (默认: 10)")
@click.option("-o", "--output", default="./output", help="输出目录 (默认: ./output)")
@click.option("-g", "--genre", default="玄幻", help="题材 (默认: 玄幻)")
@click.option("-s", "--style", default="网络小说", help="风格 (默认: 网络小说)")
@click.option("--min-words", default=3000, type=int, help="每章最低字数 (默认: 3000)")
@click.option("--max-words", default=8000, type=int, help="每章最高字数 (默认: 8000)")
@click.option("--threshold", default=7.0, type=float, help="质量阈值 0-10 (默认: 7.0)")
@click.option("--lang", default="zh", help="语言 zh/en (默认: zh)")
@click.option("--no-auto-fix", is_flag=True, help="禁用自动修复")
@click.option("--config", default="config.yaml", help="YAML配置文件路径")
def new(
    concept: str,
    chapters: int,
    output: str,
    genre: str,
    style: str,
    min_words: int,
    max_words: int,
    threshold: float,
    lang: str,
    no_auto_fix: bool,
    config: str,
):
    """从创作概念开始，自动完成整部小说。

    \b
    示例:
      novel-writer new "一个废柴少年意外获得上古神兽传承，踏上逆天改命之路"
      novel-writer new "概念" -c 50 -g 科幻 -s 硬核 --min-words 5000
    """
    pipeline_config = _build_config(
        project_name="novel",
        output_dir=output,
        genre=genre,
        style=style,
        total_chapters=chapters,
        min_words=min_words,
        max_words=max_words,
        threshold=threshold,
        language=lang,
        auto_fix=not no_auto_fix,
        config_file=config,
    )

    # Display config summary
    table = Table(title="Pipeline Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("题材", genre)
    table.add_row("风格", style)
    table.add_row("目标章节", str(chapters))
    table.add_row("每章字数范围", f"{min_words} — {max_words}")
    table.add_row("质量阈值", f"{threshold}/10")
    table.add_row("自动修复", "启用" if not no_auto_fix else "禁用")
    table.add_row("输出目录", output)
    console.print(table)

    console.print(f"\n[bold]创作概念:[/bold] {concept}\n")

    orchestrator = NovelWriterOrchestrator(pipeline_config)
    state = asyncio.run(orchestrator.run(concept, total_chapters=chapters))

    if state.chapters:
        console.print(f"\n[bold green]✓ 小说写作完成![/bold green]")
        console.print(f"  章节数: {len(state.chapters)}")
        console.print(f"  输出目录: {output}")
    else:
        console.print(f"\n[bold red]✗ 小说写作未能完成[/bold red]")


@main.command()
@click.argument("project_dir")
def continue_(project_dir: str):
    """从已有的项目继续写作。

    \b
    示例:
      novel-writer continue ./my_novel
    """
    state_file = Path(project_dir) / "novel_state.json"
    if not state_file.exists():
        console.print(f"[red]错误: 在 {project_dir} 中未找到 novel_state.json[/red]")
        sys.exit(1)

    console.print(f"[yellow]继续功能开发中...[/yellow]")
    console.print(f"状态文件: {state_file}")


@main.command()
@click.argument("project_dir")
@click.option("-f", "--format", default="md", help="导出格式: md / txt / json")
def export(project_dir: str, format: str):
    """导出小说项目为可读格式。

    \b
    示例:
      novel-writer export ./my_novel
      novel-writer export ./my_novel -f txt
    """
    state_file = Path(project_dir) / "novel_state.json"
    if not state_file.exists():
        console.print(f"[red]错误: 在 {project_dir} 中未找到 novel_state.json[/red]")
        sys.exit(1)

    import json
    from .models.schemas import NovelState
    from .utils.io import FileHandler

    data = json.loads(state_file.read_text(encoding="utf-8"))
    state = NovelState(**data)
    io = FileHandler(project_dir)

    if format == "md":
        path = io.export_full_novel(state)
        console.print(f"[green]导出完成: {path}[/green]")
    elif format == "txt":
        path = io.export_full_novel(state, "novel.txt")
        console.print(f"[green]导出完成: {path}[/green]")
    elif format == "json":
        console.print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        console.print(f"[red]不支持的导出格式: {format}[/red]")


@main.command()
def list_models():
    """显示当前配置的Agent模型。"""
    table = Table(title="Agent Model Configuration")
    table.add_column("Agent", style="cyan")
    table.add_column("Model", style="green")

    agents = [
        ("大纲Agent (Outline)", "MODEL_OUTLINE"),
        ("角色Agent (Character)", "MODEL_CHARACTER"),
        ("章节写作Agent (Chapter)", "MODEL_CHAPTER"),
        ("对话润色Agent (Dialogue)", "MODEL_DIALOGUE"),
        ("审核Agent (Review)", "MODEL_REVIEW"),
        ("连贯性Agent (Continuity)", "MODEL_CONTINUITY"),
    ]

    for name, env_key in agents:
        model = os.getenv(env_key, "gpt-4o")
        table.add_row(name, model)

    table.add_row("", "")
    table.add_row("[dim]Provider[/dim]", os.getenv("LLM_PROVIDER", "openai"))
    table.add_row("[dim]Base URL[/dim]", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    console.print(table)


if __name__ == "__main__":
    main()
