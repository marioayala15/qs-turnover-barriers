"""Where the stored measurements live.

Scripts ask for a location here instead of hard-coding one, so they can be run
from any working directory.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def data_path(name):
    """Full path for a stored measurement file under ``data/``.

    Unlike figures, these are tracked: they hold the output of stochastic runs
    that are too slow to repeat every time a figure is drawn.
    """
    out = _ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    return str(out / Path(name).name)
