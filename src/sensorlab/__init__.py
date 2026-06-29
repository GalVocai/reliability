"""sensorlab -- unified analysis pipeline for transistor/sensor characterization.

Replaces the two legacy GUI apps (Excel Device Splitter + SETimeSeriesProcessor)
with one headless, testable core plus thin frontends.
"""

from .protocol import DEFAULT, Protocol

__version__ = "0.1.0"
__all__ = ["DEFAULT", "Protocol"]
