"""Ready-made instrument models and the data files behind them."""

from pathlib import Path

_PKG_DIR = Path(__file__).parent

#: Directory holding the bundled bandpass tables.
BANDPASS_PATH = _PKG_DIR / "bandpass"

#: Directory holding the bundled effective aperture maps.
RESPONSE_PATH = _PKG_DIR / "response"

__all__ = ["BANDPASS_PATH", "RESPONSE_PATH"]
