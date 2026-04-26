"""Explorer screen — repo directory tree (left) + file content viewer (right)."""

from __future__ import annotations

import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DirectoryTree, Label, RichLog

from claw_backend.config import load_settings
from claw_backend.path_access import configured_filesystem_roots

_MAX_FILE_BYTES = 128 * 1024  # 128 KB preview cap

_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".toml": "toml",
    ".sh": "bash",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".c": "c",
    ".cpp": "cpp",
    ".cu": "cuda",
    ".f90": "fortran",
    ".f": "fortran",
    ".txt": "text",
    ".csv": "text",
}


class ExplorerScreen(Screen):
    """Two-pane repo explorer: directory tree on the left, file viewer on the right."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("tab", "focus_next", "Switch pane"),
    ]

    def compose(self) -> ComposeResult:
        settings = load_settings()
        roots = configured_filesystem_roots(settings)
        # Use the first configured root; fall back to cwd
        root_path = str(roots[0]) if roots else str(Path.cwd())

        yield Horizontal(
            Vertical(
                Label("Repository", classes="pane-title"),
                DirectoryTree(root_path, id="repo-tree"),
                id="explorer-tree-pane",
            ),
            Vertical(
                Label("File Viewer", classes="pane-title"),
                RichLog(
                    id="file-viewer",
                    highlight=True,
                    markup=False,
                    wrap=False,
                ),
                id="explorer-content-pane",
            ),
            id="explorer-container",
        )

    def on_mount(self) -> None:
        self.query_one("#repo-tree", DirectoryTree).focus()

    def action_refresh(self) -> None:
        settings = load_settings()
        roots = configured_filesystem_roots(settings)
        root_path = str(roots[0]) if roots else str(Path.cwd())
        tree = self.query_one("#repo-tree", DirectoryTree)
        tree.path = root_path  # type: ignore[assignment]
        tree.reload()

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        viewer = self.query_one("#file-viewer", RichLog)
        viewer.clear()
        path = Path(event.path)

        try:
            size = path.stat().st_size
        except OSError as exc:
            viewer.write(f"Cannot stat file: {exc}")
            return

        if size > _MAX_FILE_BYTES:
            viewer.write(
                f"[File too large to preview: {size // 1024} KB — max {_MAX_FILE_BYTES // 1024} KB]"
            )
            return

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            viewer.write(f"Cannot read file: {exc}")
            return

        # Update pane title with filename
        label = self.query_one("#explorer-content-pane Label", Label)
        label.update(f"[bold]{path.name}[/bold]  [dim]{path}[/dim]")

        for line in text.splitlines():
            viewer.write(line)
