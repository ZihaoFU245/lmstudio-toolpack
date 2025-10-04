"""
A Global Configuration for all tools
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

@dataclass
class GlobalConfig:
    transport: str = "stdio"
    port: Optional[int] = None
    data_folder: Path = Path(__file__).parent / "data"
    http_proxy_skip: Tuple[str, ...] = ()
