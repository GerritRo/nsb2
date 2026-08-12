"""nsb2 package for simulating NSB in IACTs."""

from pathlib import Path

__version__ = "0.1.0"

ASSETS_PATH = Path(__file__).parent / "data"

__all__ = ["ASSETS_PATH", "__version__"]
