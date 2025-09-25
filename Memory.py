"""
Configure `MEMORY_FILE` variable to store the memory file
"""
from fastmcp import FastMCP
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import tempfile
import shutil
import os

@dataclass
class Result:
    success: bool
    reason: Optional[str] = None
    data: Optional[list] = None

mcp = FastMCP("Memories")

MEMORY_FILE = Path(__file__).parent / "memory.md"

def _ensure_memory_file() -> None:
    """Ensure the memory file and its parent directory exist."""
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.touch(exist_ok=True)

@mcp.tool
def remember(title: str, content: str) -> Result:
    """
    Remember anything, will be stored
    
    Parameters:
        - title : str
            Title of what you want to store, will be used for indexing
        - content : str
            Main body, details
    
    Return:
        - Result, indicating success or not
    """
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Prepend lazily by streaming original to a temp file
        tmp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=str(MEMORY_FILE.parent))
        tmp_path = tmp.name
        try:
            tmp.write(f" * {title}\t{content}\n")
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as src:
                    shutil.copyfileobj(src, tmp)
            except FileNotFoundError:
                # No existing memory file; just the new line
                pass
            tmp.close()
            os.replace(tmp_path, MEMORY_FILE)
            return Result(success=True, data=[{"title": title, "content": content}])
        finally:
            # If anything above raised after creation, ensure cleanup
            try:
                if os.path.exists(tmp_path):
                    Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
    except FileNotFoundError:
        return Result(success=False, reason="Internal error, memory file not found")

@mcp.tool
def retrieve(top_k : int = 25) -> Result:
    """Retrieve most recent memory. Default is 25. Input should not be less than 10"""
    try:
        _ensure_memory_file()
        memories = []
        with open(MEMORY_FILE, "r", encoding='utf-8') as f:
            for line in f:
                if line.startswith(" * "):
                    parts = line[3:].strip().split('\t', 1)
                    if len(parts) == 2:
                        title, content = parts
                        memories.append({"title": title, "content": content})
                        if len(memories) >= top_k:
                            break
        
        return Result(success=True, data=memories)
    except FileNotFoundError:
        return Result(success=False, reason="Memory file not found")
    
@mcp.tool
def modify(title: str, incoming_content: str) -> Result:
    """Modify an existing memory"""
    try:
        _ensure_memory_file()
        found = False
        tmp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=str(MEMORY_FILE.parent))
        tmp_path = tmp.name
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as src:
                for line in src:
                    if not found and line.startswith(" * "):
                        parts = line[3:].strip().split('\t', 1)
                        if len(parts) == 2 and parts[0] == title:
                            tmp.write(f" * {title}\t{incoming_content}\n")
                            found = True
                        else:
                            tmp.write(line)
                    else:
                        tmp.write(line)
            tmp.close()
            if found:
                os.replace(tmp_path, MEMORY_FILE)
                return Result(success=True, data=[{"title": title, "content": incoming_content}])
            else:
                # No change; discard temp file
                try:
                    Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                return Result(success=False, reason="Memory not found")
        except FileNotFoundError:
            # Ensure temp is removed if created
            try:
                Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            return Result(success=False, reason="Memory file not found")
    except FileNotFoundError:
        return Result(success=False, reason="Memory file not found")

@mcp.tool
def remove(title: str) -> Result:
    """Remove a memory permenently"""
    try:
        _ensure_memory_file()
        found = False
        removed_record = None
        tmp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=str(MEMORY_FILE.parent))
        tmp_path = tmp.name
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as src:
                for line in src:
                    if not found and line.startswith(" * "):
                        parts = line[3:].strip().split('\t', 1)
                        if len(parts) == 2 and parts[0] == title:
                            removed_record = {"title": parts[0], "content": parts[1]}
                            found = True
                            # skip writing this line
                            continue
                    tmp.write(line)
            tmp.close()
            if found:
                os.replace(tmp_path, MEMORY_FILE)
                return Result(success=True, data=[removed_record])
            else:
                try:
                    Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                return Result(success=False, reason="Memory not found")
        except FileNotFoundError:
            try:
                Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            return Result(success=False, reason="Memory file not found")
    except FileNotFoundError:
        return Result(success=False, reason="Memory file not found")

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())

