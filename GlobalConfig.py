"""
A Global Configuration for all tools
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class GlobalConfig:
    transport: str = "stdio"
    port: Optional[int] = None
    