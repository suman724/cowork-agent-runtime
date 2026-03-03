"""Preamble injected at the top of every Python script execution.

Hooks matplotlib.show() to save figures as PNG files and sets pandas display
options for readable text output. Uses underscore-prefixed names to avoid
polluting the user's namespace.
"""

PREAMBLE = """
import os as _os, sys as _sys

# Hook matplotlib to save figures instead of displaying
_output_dir = _os.environ.get("COWORK_OUTPUT_DIR", "/tmp")
_fig_count = 0

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _original_show = _plt.show
    def _cowork_show(*args, **kwargs):
        global _fig_count
        for _fig_num in _plt.get_fignums():
            _fig = _plt.figure(_fig_num)
            _path = _os.path.join(_output_dir, f"figure_{_fig_count}.png")
            _fig.savefig(_path, dpi=100, bbox_inches="tight")
            _fig_count += 1
        _plt.close("all")
    _plt.show = _cowork_show
except ImportError:
    pass

# Set pandas display options for readable text output
try:
    import pandas as _pd
    _pd.set_option("display.max_rows", 50)
    _pd.set_option("display.max_columns", 20)
    _pd.set_option("display.width", 120)
except ImportError:
    pass
"""
