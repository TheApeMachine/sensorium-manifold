"""Clean console interface for the Sensorium.

Usage:
    from sensorium.console import console

    with console.spinner("Loading data..."):
        do_work()

    console.success("Done", detail="Loaded 100 items")
    console.warn("Something odd")
    console.error("Failed", detail=str(err))
    console.info("Device: mps")
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover
    RichConsole = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]


class Console:
    """Minimal logging interface with rich output."""

    __slots__ = ('_console',)

    def __init__(self) -> None:
        self._console = RichConsole() if RichConsole is not None else None

    @contextmanager
    def spinner(self, message: str):
        """Show a spinner while work is in progress."""
        if self._console is None:
            print(message)
            yield
            return
        with self._console.status(f"[bold cyan]{message}", spinner="dots"):
            yield

    def success(self, message: str, *, detail: Optional[str] = None, title: Optional[str] = None) -> None:
        """Green success message."""
        if self._console is None or Text is None:
            print(f"✓ {message}" + (f" {detail}" if detail else ""))
            return
        text = Text(message, style="bold green")
        if detail:
            text.append(f"\n{detail}", style="dim")
        if title:
            self._console.print(Panel(text, title=f"[cyan]{title}[/cyan]", border_style="green"))
        else:
            self._console.print(f"[bold green]✓[/bold green] {message}" + (f" [dim]{detail}[/dim]" if detail else ""))

    def warn(self, message: str, *, detail: Optional[str] = None) -> None:
        """Yellow warning message."""
        if self._console is None:
            print(f"⚠ {message}" + (f" {detail}" if detail else ""))
            return
        self._console.print(f"[yellow]⚠[/yellow] {message}" + (f" [dim]{detail}[/dim]" if detail else ""))

    def error(self, message: str, *, detail: Optional[str] = None) -> None:
        """Red error message."""
        if self._console is None:
            print(f"✗ {message}" + (f" {detail}" if detail else ""))
            return
        self._console.print(f"[bold red]✗[/bold red] {message}" + (f" [dim]{detail}[/dim]" if detail else ""))

    def info(self, message: str, *, detail: Optional[str] = None) -> None:
        """Blue info message."""
        if self._console is None:
            print(f"• {message}" + (f" {detail}" if detail else ""))
            return
        self._console.print(f"[blue]•[/blue] {message}" + (f" [dim]{detail}[/dim]" if detail else ""))

    def header(self, title: str, **fields: str) -> None:
        """Show a panel with key-value fields."""
        if self._console is None or Panel is None:
            print(title)
            for k, v in fields.items():
                print(f"- {k}: {v}")
            return
        lines = [f"[bold]{k}:[/bold] {v}" for k, v in fields.items()]
        self._console.print(Panel("\n".join(lines), title=f"[cyan]{title}[/cyan]", border_style="blue"))


console = Console()
