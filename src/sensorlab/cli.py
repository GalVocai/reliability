"""Command-line entry point: run the full analysis on a folder, or launch the GUI."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sensorlab", description="Analyze a sensor results folder.")
    parser.add_argument("folder", nargs="?", help="results folder (omit to launch the GUI)")
    parser.add_argument("--day", default="", help="Day# / timepoint label for plot titles")
    parser.add_argument("--no-plots", action="store_true", help="masters only (skip per-device plots)")
    parser.add_argument("--gui", action="store_true", help="launch the GUI")
    args = parser.parse_args(argv)

    if args.gui or not args.folder:
        from .gui import main as gui_main
        gui_main()
        return 0

    from .pipeline import analyze_folder
    written = analyze_folder(args.folder, make_plots=not args.no_plots, day=args.day, log=print)
    print("Masters written:")
    for kind, paths in written.items():
        for path in paths:
            print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
