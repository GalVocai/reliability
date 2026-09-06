"""PyInstaller-compatible entry point for the SensorLab desktop GUI."""

from multiprocessing import freeze_support

from sensorlab.gui import main


if __name__ == "__main__":
    freeze_support()
    main()
