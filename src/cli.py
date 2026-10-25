"""Typer CLI interface for Multi-Head Self-Attention Lab."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.attention_service import AttentionService
from src.core.preset_data import export_presets_json, get_all_presets, get_preset_by_id

app = typer.Typer(
    name="attention-cli",
    help="Interactive CLI tool for exploring LLM Multi-Head Self-Attention mechanisms.",
)
console = Console()


@app.command()
def compute(
    sentence: str = typer.Argument(
        "The animal didn't cross the street because it was too tired",
        help="Input sentence to analyze through self-attention",
    ),
    head: int = typer.Option(
        1, "--head", "-h", help="Attention head index (0-3) or -1 for composite"
    ),
    causal: bool = typer.Option(
        False, "--causal", "-c", help="Enable causal masking (decoder-only GPT mode)"
    ),
    temperature: float = typer.Option(
        1.0, "--temperature", "-t", help="Softmax attention temperature"
    ),
) -> None:
    """Compute and display self-attention matrix for a sentence."""
    service = AttentionService()
    result = service.compute_attention(
        sentence=sentence,
        is_causal=causal,
        temperature=temperature,
    )

    mode_desc = "Causal (GPT Autoregressive)" if causal else "Bidirectional (BERT)"
    console.print(
        Panel.fit(
            f"[bold cyan]Input Sentence:[/bold cyan] {sentence}\n"
            f"[bold]Tokens ({result.token_count}):[/bold] {', '.join(result.tokens)}\n"
            f"[bold]Mode:[/bold] {mode_desc} | [bold]Temp:[/bold] {temperature:.2f}",
            title="🎯 Multi-Head Self-Attention",
            border_style="cyan",
        )
    )

    matrix: list[list[float]]
    title: str

    if head < 0 or head >= len(result.heads):
        matrix = result.composite_matrix
        title = "Composite Attention Matrix (Averaged across all heads)"
    else:
        target_head = result.heads[head]
        matrix = target_head.attention_matrix
        title = f"{target_head.name} ({target_head.specialization})"
        console.print(f"[dim]{target_head.description}[/dim]\n")

    # Render table
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Query \\ Key", style="cyan", justify="right")

    for tok in result.tokens:
        table.add_column(tok, justify="center")

    for i, q_tok in enumerate(result.tokens):
        row_vals: list[str] = [f"{i}: {q_tok}"]
        for j in range(result.token_count):
            val = matrix[i][j]
            if causal and j > i:
                row_vals.append("[dim]-[/dim]")
            elif val > 0.40:
                row_vals.append(f"[bold green]{val:.3f}[/bold green]")
            elif val > 0.15:
                row_vals.append(f"[yellow]{val:.3f}[/yellow]")
            else:
                row_vals.append(f"[dim]{val:.3f}[/dim]")
        table.add_row(*row_vals)

    console.print(table)


@app.command()
def presets() -> None:
    """List all available educational preset scenarios."""
    all_presets = get_all_presets()
    table = Table(title="📚 Educational Attention Presets", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="bold")
    table.add_column("Category", style="magenta")
    table.add_column("Sentence", style="dim")

    for p in all_presets:
        table.add_row(p.id, p.title, p.category, p.sentence)

    console.print(table)


@app.command()
def inspect(
    preset_id: str = typer.Argument(..., help="Preset ID to inspect (e.g. winograd-tired)"),
) -> None:
    """Inspect a preset scenario with linguistic explanation and attention details."""
    preset = get_preset_by_id(preset_id)
    if not preset:
        console.print(f"[red]Error: Preset '{preset_id}' not found.[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Title:[/bold cyan] {preset.title}\n"
            f"[bold magenta]Category:[/bold magenta] {preset.category}\n"
            f"[bold]Sentence:[/bold] {preset.sentence}\n\n"
            f"[bold green]Explanation:[/bold green]\n{preset.explanation}",
            title=f"Scenario: {preset.id}",
            border_style="green",
        )
    )

    console.print("[bold]Key Attention Pairs to Observe:[/bold]")
    for pair in preset.key_pairs:
        console.print(
            f"  • [cyan]{pair.source_token}[/cyan] ({pair.source_idx}) ➔ "
            f"[yellow]{pair.target_token}[/yellow] ({pair.target_idx}) "
            f"[bold]({pair.expected_strength.upper()}):[/bold] {pair.reason}"
        )


@app.command()
def export(
    output: str = typer.Option("data/presets.json", "--output", "-o", help="Output JSON path"),
) -> None:
    """Export presets to JSON file for frontend application."""
    path = export_presets_json()
    console.print(f"[green]Successfully exported presets to {path}[/green]")


if __name__ == "__main__":
    app()
