"""
A Global Configuration for all tools
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class GlobalConfig:
    transport: str = "stdio"
    port: Optional[int] = None
    data_folder: Path = Path(__file__).parent / "data"

