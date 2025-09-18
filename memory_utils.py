from __future__ import annotations

"""Simple persistent memory for the trading bot."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Iterable
import json
import os

MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'bot_memory.json')


def normalize_param_keys(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``params`` with all keys uppercased."""
    return {k.upper(): v for k, v in params.items()}


@dataclass
class MemoryEntry:
    timestamp: str
    type: str
    data: Dict[str, Any]


class MemoryManager:
    """Utility to store and query bot memory."""

    def __init__(self, path: str = MEMORY_FILE) -> None:
        self.path = path
        self.data: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []

    def _save(self) -> None:
        try:
            with open(self.path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record a new memory event."""
        entry = MemoryEntry(datetime.now(timezone.utc).isoformat(), event_type, data)
        self.data.append(asdict(entry))
        self._save()

    # Trade helpers -----------------------------------------------------

    def add_trade_open(self, info: Dict[str, Any]) -> None:
        """Persist trade entry conditions."""
        self.add_event("trade_open", info)

    def add_trade_close(self, info: Dict[str, Any]) -> None:
        """Persist trade exit details."""
        self.add_event("trade_close", info)

    def trade_history(self) -> Iterable[Dict[str, Any]]:
        """Yield all closed trade records from memory."""
        for entry in self.data:
            if entry.get("type") == "trade_close":
                yield entry.get("data", {})

    def trade_stats(self) -> Dict[str, float]:
        """Return aggregate statistics from closed trades."""
        trades = list(self.trade_history())
        if not trades:
            return {"count": 0, "profit": 0.0, "winrate": 0.0}
        count = len(trades)
        profit = sum(t.get("profit", 0.0) for t in trades)
        winrate = sum(1 for t in trades if t.get("profit", 0.0) > 0) / count
        return {"count": count, "profit": profit, "winrate": winrate}

    def last_best_params(self) -> Dict[str, Any] | None:
        """Return the most recent parameters from optimization."""
        for entry in reversed(self.data):
            if entry.get('type') == 'optimize':
                params = entry.get('data', {}).get('best_params')
                if params:
                    return normalize_param_keys(params)
        return None


memory_manager = MemoryManager()

