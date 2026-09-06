"""PyInstaller-compatible entry point for the SensorLab desktop GUI."""

from multiprocessing import freeze_support
from pathlib import Path
import sys


# When this file is launched directly (for example with VS Code's "Run Python
# File"), prefer the source tree beside it over any older editable installation
# that may exist elsewhere on the computer.  PyInstaller supplies its bundled
# package path itself, so the executable does not need this adjustment.
if not getattr(sys, "frozen", False):
    local_src = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(local_src))

from sensorlab.gui import main


if __name__ == "__main__":
    freeze_support()
    main()
