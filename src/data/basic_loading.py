"""Explicit match-table loading; importing this module performs no data IO."""

from pathlib import Path

import pandas as pd

from src.data.data_loader import stream_jsonl


def load_match(match_id=1886347, data_dir=None):
    """Return event/phase DataFrames and a lazy tracking iterator.

    Whole-match tracking is deliberately not materialized into a DataFrame.
    ``data_dir`` is required outside a repository checkout.
    """
    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data"
    directory = root / "matches" / str(match_id)
    events = pd.read_csv(directory / f"{match_id}_dynamic_events.csv")
    phases = pd.read_csv(directory / f"{match_id}_phases_of_play.csv")
    tracking = stream_jsonl(directory / f"{match_id}_tracking_extrapolated.jsonl")
    return events, phases, tracking
