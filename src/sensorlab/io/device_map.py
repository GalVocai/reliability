"""Per-device condition cache.

Device identity (ESN + TNumber) is read automatically from each file's
Description sheet. The only thing a human supplies is the experimental
*condition / label* per device -- entered once (later via a GUI window) and
remembered here, keyed by ESN so it is reused across every future run.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CACHE_PATH = Path.home() / ".sensorlab" / "conditions.json"


class ConditionCache:
    """Maps ESN (card) -> condition label. Persisted as JSON."""

    def __init__(self, path: Path = DEFAULT_CACHE_PATH, mapping: dict[str, str] | None = None):
        self.path = Path(path)
        self._map: dict[str, str] = dict(mapping or {})

    def get(self, esn: str, default: str = "") -> str:
        return self._map.get(str(esn), default)

    def set(self, esn: str, condition: str) -> None:
        self._map[str(esn)] = condition

    def update(self, mapping: dict[str, str]) -> None:
        self._map.update({str(k): v for k, v in mapping.items()})

    def missing(self, esns: list[str]) -> list[str]:
        """ESNs that still have no condition assigned (drive the GUI prompt)."""
        return [e for e in dict.fromkeys(map(str, esns)) if not self._map.get(str(e))]

    def as_dict(self) -> dict[str, str]:
        return dict(self._map)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._map, indent=2, ensure_ascii=False), encoding="utf-8")


def load_condition_cache(path: Path = DEFAULT_CACHE_PATH) -> ConditionCache:
    path = Path(path)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return ConditionCache(path, {str(k): str(v) for k, v in data.items()})
    return ConditionCache(path)
