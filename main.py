"""
MCP json config generating tool
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from GlobalConfig import GlobalConfig


@dataclass
class ServerOption:
    script_path: Path

    @property
    def identifier(self) -> str:
        return self.script_path.stem

    @property
    def default_display_name(self) -> str:
        return _humanize_name(self.identifier)


@dataclass
class ServerSelection:
    option: ServerOption
    transport: str = "stdio"
    http_url: Optional[str] = None


@dataclass
class ServerChoice:
    display_name: str
    slug: str
    transport: str
    command: Optional[str]
    script_path: Optional[Path]
    http_url: Optional[str] = None


class _Key(Enum):
    UP = "up"
    DOWN = "down"
    ENTER = "enter"
    SPACE = "space"
    TOGGLE_ALL = "toggle_all"
    TOGGLE_HTTP = "toggle_http"
    TOGGLE_HTTP_ALL = "toggle_http_all"
    ESC = "esc"
    OTHER = "other"


_CTRL_HTTP_ALL_CODES = {"\x08", "\x7f"}


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    _enable_vt_mode()

    mcp_dir = base_dir / "MCPs"
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    options = _discover_servers(mcp_dir)
    if not options:
        print("No MCP servers found in the MCPs directory.")
        return

    interactive = sys.stdin.isatty() and sys.stdout.isatty()

    platform_key = _prompt_platform(interactive)
    default_mode = _prompt_default_selection(interactive)
    selections = _prompt_server_selection(options, default_mode, interactive)
    if not selections:
        print("No servers selected. Nothing to do.")
        return

    summary_names = []
    for sel in selections:
        name = sel.option.default_display_name
        if sel.transport == "http":
            name = f"{name} (http)"
        summary_names.append(name)
    print("Selected servers: " + ", ".join(summary_names))

    python_command = _resolve_python_command(base_dir)
    _ensure_http_urls(selections, interactive)

    choices = [_build_server_choice(sel, python_command) for sel in selections]

    global_transport = GlobalConfig().transport
    if platform_key == "vscode":
        content = _build_vscode_config(choices)
        suggested_name = "vscode_mcp.json"
    elif platform_key == "cursor":
        content = _build_cursor_config(choices)
        suggested_name = "cursor_mcp.json"
    else:
        content = _build_lmstudio_config(choices)
        suggested_name = "lmstudio_mcp.json"

    output_path = _prompt_output_path(data_dir, suggested_name)
    _write_config(output_path, content)
    if global_transport == "stdio" and any(choice.transport == "http" for choice in choices):
        warning = _color(
            "Warning: Global transport is set to stdio but HTTP servers were generated.",
            fg="magenta",
            bold=True,
        )
        print(warning)
    print(f"Saved configuration to {output_path}")


def _discover_servers(folder: Path) -> List[ServerOption]:
    if not folder.exists():
        return []
    return [
        ServerOption(script_path=path)
        for path in sorted(folder.glob("*.py"))
        if path.is_file()
    ]


def _prompt_platform(interactive: bool) -> str:
    options = [("vscode", "VS Code"), ("lmstudio", "LM Studio"), ("cursor", "Cursor")]
    if interactive:
        index = _interactive_single_select(
            title="Target Platform",
            options=[label for _, label in options],
        )
        return options[index][0]

    mapping = {"1": "vscode", "2": "lmstudio", "3": "cursor"}
    prompt = (
        "Select target platform:\n"
        "  1) VS Code\n"
        "  2) LM Studio\n"
        "  3) Cursor\n"
        "Enter choice [1/3]: "
    )
    while True:
        choice = input(prompt).strip()
        if choice in mapping:
            return mapping[choice]
        print("Please enter 1, 2, or 3.")


def _prompt_default_selection(interactive: bool) -> str:
    options = [("all", "Select all servers"), ("none", "Select none")]
    if interactive:
        index = _interactive_single_select(
            title="Default Selection Mode",
            options=[label for _, label in options],
        )
        return options[index][0]

    prompt = (
        "Default selection mode:\n"
        "  1) Select all servers\n"
        "  2) Select none\n"
        "Enter choice [1/2]: "
    )
    while True:
        choice = input(prompt).strip()
        if choice == "1":
            return "all"
        if choice == "2":
            return "none"
        print("Please enter 1 or 2.")


def _prompt_server_selection(
    options: Iterable[ServerOption], default_mode: str, interactive: bool
) -> List[ServerSelection]:
    options = list(options)
    if interactive:
        initial = set(range(len(options))) if default_mode == "all" else set()
        return _interactive_multi_select(
            title="Select MCP Servers",
            options=options,
            initial_selected=initial,
        )

    return _prompt_server_selection_fallback(options, default_mode)

def _prompt_server_selection_fallback(
    options: Sequence[ServerOption], default_mode: str
) -> List[ServerSelection]:
    indices = list(range(len(options)))
    selected = set(indices if default_mode == "all" else [])
    while True:
        print("\nCurrent selection:")
        for idx, opt in enumerate(options, start=1):
            mark = "x" if (idx - 1) in selected else " "
            print(f"[{mark}] {idx}. {opt.default_display_name} ({opt.script_path.name})")
        raw = input(
            "Enter numbers to toggle selection (comma separated), or press Enter to continue: "
        ).strip()
        if not raw:
            break
        tokens = _split_numbers(raw)
        invalid = []
        for token in tokens:
            if not token.isdigit():
                invalid.append(token)
                continue
            index = int(token) - 1
            if index not in range(len(options)):
                invalid.append(token)
                continue
            if index in selected:
                selected.remove(index)
            else:
                selected.add(index)
        if invalid:
            print(f"Ignored invalid entries: {', '.join(invalid)}")

    selections: List[ServerSelection] = []
    for idx in sorted(selected):
        opt = options[idx]
        while True:
            answer = (
                input(
                    f"Use HTTP transport for {opt.default_display_name}? [y/N]: "
                )
                .strip()
                .lower()
            )
            if answer in {"", "n", "no"}:
                transport = "stdio"
                break
            if answer in {"y", "yes"}:
                transport = "http"
                break
            print("Please enter y or n.")
        selections.append(ServerSelection(option=opt, transport=transport))
    return selections


def _ensure_http_urls(selections: List[ServerSelection], interactive: bool) -> None:
    for selection in selections:
        if selection.transport != "http":
            continue
        if selection.http_url:
            continue
        suggestion = f"https://example.com/{_slugify(selection.option.identifier) or selection.option.identifier}"
        if not interactive:
            selection.http_url = suggestion
            continue
        while True:
            url = (
                input(
                    f"Enter URL for {selection.option.default_display_name} [{suggestion}]: "
                )
                .strip()
                or suggestion
            )
            if url:
                selection.http_url = url
                break
            print("URL cannot be empty for HTTP transport.")


def _build_server_choice(selection: ServerSelection, command: str) -> ServerChoice:
    option = selection.option
    display = option.default_display_name
    slug = _slugify(option.identifier) or option.identifier.lower()
    if selection.transport == "http":
        return ServerChoice(
            display_name=display,
            slug=slug,
            transport="http",
            command=None,
            script_path=None,
            http_url=selection.http_url or "",
        )

    script_path = option.script_path.resolve()
    return ServerChoice(
        display_name=display,
        slug=slug,
        transport="stdio",
        command=command,
        script_path=script_path,
        http_url=None,
    )


def _resolve_python_command(base_dir: Path) -> str:
    candidates: List[Path] = []
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env:
        candidates.append(Path(venv_env))
    candidates.append(base_dir / ".venv")

    checked: set[Path] = set()
    for root in candidates:
        if not root:
            continue
        root = root.resolve()
        if root in checked or not root.exists():
            continue
        checked.add(root)
        bin_dir = root / ("Scripts" if os.name == "nt" else "bin")
        for name in ("python.exe", "python3", "python"):
            candidate = bin_dir / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)

    return str(Path(sys.executable).resolve())


def _prompt_output_path(data_dir: Path, suggested: str) -> Path:
    print(f"\nConfiguration files will be saved under {data_dir}")
    while True:
        user_input = input(f"Output file name [{suggested}]: ").strip() or suggested
        filename = user_input if user_input.lower().endswith(".json") else f"{user_input}.json"
        output_path = data_dir / filename
        if output_path.exists():
            answer = input(f"{output_path} exists. Overwrite? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                continue
        return output_path


def _build_vscode_config(choices: List[ServerChoice]) -> dict:
    servers = {}
    for choice in choices:
        if choice.transport == "http":
            servers[choice.display_name] = {
                "type": "http",
                "url": choice.http_url or "",
                "headers": {},
            }
        else:
            servers[choice.display_name] = {
                "type": "stdio",
                "command": choice.command,
                "args": [str(choice.script_path)],
                "env": {},
            }
    return {"servers": servers}


def _build_lmstudio_config(choices: List[ServerChoice]) -> dict:
    servers = {}
    for choice in choices:
        if choice.transport == "http":
            servers[choice.slug] = {
                "url": choice.http_url or "",
                "headers": {},
            }
        else:
            servers[choice.slug] = {
                "command": choice.command,
                "args": [str(choice.script_path)],
                "env": {},
            }
    return {"mcpServers": servers}


def _build_cursor_config(choices: List[ServerChoice]) -> dict:
    servers = {}
    for choice in choices:
        key = choice.slug
        if choice.transport == "http":
            servers[key] = {
                "url": choice.http_url or "",
                "headers": {},
            }
        else:
            servers[key] = {
                "command": choice.command,
                "args": [str(choice.script_path)],
                "env": {},
            }
    return {"mcp": {"servers": servers}}


def _write_config(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def _interactive_single_select(title: str, options: Sequence[str]) -> int:
    if not options:
        raise ValueError("options must not be empty")
    index = 0
    while True:
        _render_lines(
            [
                _color(title, fg="cyan", bold=True),
                "",
                _color("Use arrow keys to move, Enter to confirm.", fg="yellow"),
                "",
                *[
                    _format_option(label, idx == index, marked=None)
                    for idx, label in enumerate(options)
                ],
            ]
        )
        key = _read_key()
        if key == _Key.UP:
            index = (index - 1) % len(options)
        elif key == _Key.DOWN:
            index = (index + 1) % len(options)
        elif key == _Key.ENTER:
            _clear_screen()
            return index
        elif key == _Key.ESC:
            _clear_screen()
            raise KeyboardInterrupt


def _interactive_multi_select(
    title: str, options: Sequence[ServerOption], initial_selected: Iterable[int]
) -> List[ServerSelection]:
    if not options:
        return []
    index = 0
    selected = set(initial_selected)
    transports: Dict[int, str] = {idx: "stdio" for idx in range(len(options))}
    while True:
        lines = [
            _color(title, fg="cyan", bold=True),
            "",
            _color(
                "Use arrow keys to move, Space or A to toggle selection, H to toggle transport, Enter to confirm, Ctrl+A to toggle all selections, Ctrl+H to toggle all transports, Esc to cancel.",
                fg="yellow",
            ),
            "",
        ]
        for idx, option in enumerate(options):
            marked = idx in selected
            transport_label = "http" if transports[idx] == "http" else "stdio"
            label = f"{option.default_display_name} ({transport_label})"
            lines.append(_format_option(label, idx == index, marked=marked))
        lines.append("")
        lines.append(
            _color(f"Selected: {len(selected)}/{len(options)}", fg="green", bold=True)
        )
        _render_lines(lines)

        key = _read_key()
        if key == _Key.UP:
            index = (index - 1) % len(options)
        elif key == _Key.DOWN:
            index = (index + 1) % len(options)
        elif key == _Key.SPACE:
            if index in selected:
                selected.remove(index)
            else:
                selected.add(index)
        elif key == _Key.TOGGLE_ALL:
            if len(selected) == len(options):
                selected.clear()
            else:
                selected = set(range(len(options)))
        elif key == _Key.TOGGLE_HTTP:
            transports[index] = "http" if transports[index] == "stdio" else "stdio"
        elif key == _Key.TOGGLE_HTTP_ALL:
            all_http = all(value == "http" for value in transports.values())
            new_value = "stdio" if all_http else "http"
            for key_idx in transports:
                transports[key_idx] = new_value
        elif key == _Key.ENTER:
            _clear_screen()
            return [
                ServerSelection(option=options[i], transport=transports[i])
                for i in sorted(selected)
            ]
        elif key == _Key.ESC:
            _clear_screen()
            return []


def _format_option(label: str, is_active: bool, marked: bool | None) -> str:
    prefix = "*" if is_active else " "
    if marked is None:
        body = label
    else:
        checkbox = "[x]" if marked else "[ ]"
        body = f"{checkbox} {label}"
    line = f"{prefix} {body}"
    return _color(line, fg="magenta" if marked else None, bold=is_active)


def _render_lines(lines: Sequence[str]) -> None:
    _clear_screen()
    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()


def _clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H\033[0m")
    sys.stdout.flush()


def _read_key() -> _Key:
    return _read_key_windows() if os.name == "nt" else _read_key_posix()


def _read_key_windows() -> _Key:
    import msvcrt

    while True:
        ch = msvcrt.getwch()
        if ch == "\r":
            return _Key.ENTER
        if ch in {" ", "a", "A"}:
            return _Key.SPACE
        if ch in {"h", "H"}:
            return _Key.TOGGLE_HTTP
        if ch in _CTRL_HTTP_ALL_CODES:
            return _Key.TOGGLE_HTTP_ALL
        if ch == "\x01":
            return _Key.TOGGLE_ALL
        if ch in {"\x00", "\xe0"}:
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return _Key.UP
            if ch2 == "P":
                return _Key.DOWN
            continue
        if ch == "\x1b":
            return _Key.ESC
        if ch == "\x03":
            raise KeyboardInterrupt


def _read_key_posix() -> _Key:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch in {"\r", "\n"}:
            return _Key.ENTER
        if ch in {" ", "a", "A"}:
            return _Key.SPACE
        if ch in {"h", "H"}:
            return _Key.TOGGLE_HTTP
        if ch in _CTRL_HTTP_ALL_CODES:
            return _Key.TOGGLE_HTTP_ALL
        if ch == "\x01":
            return _Key.TOGGLE_ALL
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return _Key.UP
            if seq == "[B":
                return _Key.DOWN
            return _Key.ESC
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return _Key.OTHER


def _enable_vt_mode() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        import ctypes.wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.wintypes.DWORD()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            mode.value |= 0x0004
            kernel32.SetConsoleMode(handle, mode)
    except Exception:
        pass


def _color(text: str, *, fg: str | None = None, bold: bool = False) -> str:
    if not text:
        return text
    codes: List[str] = []
    if bold:
        codes.append("1")
    if fg:
        color_map = {
            "black": "30",
            "red": "31",
            "green": "32",
            "yellow": "33",
            "blue": "34",
            "magenta": "35",
            "cyan": "36",
            "white": "37",
        }
        code = color_map.get(fg.lower())
        if code:
            codes.append(code)
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _split_numbers(raw: str) -> List[str]:
    separators = [",", " ", ";"]
    for sep in separators[1:]:
        raw = raw.replace(sep, separators[0])
    return [token.strip() for token in raw.split(separators[0]) if token.strip()]


def _humanize_name(value: str) -> str:
    spaced = re.sub(r"[_\-]+", " ", value)
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", spaced)
    normalized = " ".join(part for part in spaced.split() if part)
    return normalized.title() if normalized else value


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-").lower()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
