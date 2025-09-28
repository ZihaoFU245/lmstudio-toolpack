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
from typing import Iterable, List, Sequence

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
class ServerChoice:
    display_name: str
    slug: str
    command: str
    script_path: Path


class _Key(Enum):
    UP = "up"
    DOWN = "down"
    ENTER = "enter"
    SPACE = "space"
    TOGGLE_ALL = "toggle_all"
    ESC = "esc"
    OTHER = "other"


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
    selected_options = _prompt_server_selection(options, default_mode, interactive)
    if not selected_options:
        print("No servers selected. Nothing to do.")
        return

    print("Selected servers: " + ", ".join(opt.default_display_name for opt in selected_options))

    choices = [_build_server_choice(opt) for opt in selected_options]

    transport = GlobalConfig().transport
    if platform_key == "vscode":
        content = _build_vscode_config(choices, transport)
        suggested_name = "vscode_mcp.json"
    else:
        content = _build_lmstudio_config(choices)
        suggested_name = "lmstudio_mcp.json"

    output_path = _prompt_output_path(data_dir, suggested_name)
    _write_config(output_path, content)
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
    options = [("vscode", "VS Code"), ("lmstudio", "LM Studio")]
    if interactive:
        index = _interactive_single_select(
            title="Target Platform",
            options=[label for _, label in options],
        )
        return options[index][0]

    mapping = {"1": "vscode", "2": "lmstudio"}
    prompt = (
        "Select target platform:\n"
        "  1) VS Code\n"
        "  2) LM Studio\n"
        "Enter choice [1/2]: "
    )
    while True:
        choice = input(prompt).strip()
        if choice in mapping:
            return mapping[choice]
        print("Please enter 1 or 2.")


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
) -> List[ServerOption]:
    options = list(options)
    if interactive:
        labels = [opt.default_display_name for opt in options]
        initial = set(range(len(options))) if default_mode == "all" else set()
        selected_indices = _interactive_multi_select(
            title="Select MCP Servers",
            options=labels,
            initial_selected=initial,
        )
        return [options[i] for i in selected_indices]

    return _prompt_server_selection_fallback(options, default_mode)


def _prompt_server_selection_fallback(
    options: Sequence[ServerOption], default_mode: str
) -> List[ServerOption]:
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
            return [opt for idx, opt in enumerate(options) if idx in selected]
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


def _build_server_choice(option: ServerOption) -> ServerChoice:
    display = option.default_display_name
    slug = _slugify(option.identifier) or option.identifier.lower()
    command = str(Path(sys.executable).resolve())
    script_path = option.script_path.resolve()
    return ServerChoice(
        display_name=display,
        slug=slug,
        command=command,
        script_path=script_path,
    )


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


def _build_vscode_config(choices: List[ServerChoice], transport: str) -> dict:
    servers = {}
    for choice in choices:
        servers[choice.display_name] = {
            "type": transport,
            "command": choice.command,
            "args": [str(choice.script_path)],
        }
    return {"servers": servers}


def _build_lmstudio_config(choices: List[ServerChoice]) -> dict:
    servers = {}
    for choice in choices:
        servers[choice.slug] = {
            "command": choice.command,
            "args": [str(choice.script_path)],
        }
    return {"mcpServers": servers}


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
    title: str, options: Sequence[str], initial_selected: Iterable[int]
) -> List[int]:
    if not options:
        return []
    index = 0
    selected = set(initial_selected)
    while True:
        lines = [
            _color(title, fg="cyan", bold=True),
            "",
            _color(
                "Use arrow keys to move, Space or A to toggle, Enter to confirm, Ctrl+A to toggle all, Esc to cancel.",
                fg="yellow",
            ),
            "",
        ]
        for idx, label in enumerate(options):
            marked = idx in selected
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
        elif key == _Key.ENTER:
            _clear_screen()
            return sorted(selected)
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
