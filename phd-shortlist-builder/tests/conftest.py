"""Tests conftest — adds src/ to path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
