"""How long the dashboard keeps what it reads.

Every loader in app.py keeps its result for CACHE_SECONDS, shared by every
visitor (st.cache_data), except an empty result. The data changes once a day,
after the US close, so an hour still shows each refresh within the hour, while
it bounds what the page reads from the database: at most one read an hour for
each loader, ticker and lookback, whatever the traffic (docs/architecture.md,
"Running on free tiers").

An empty result is read again on the next visit instead: kept, one read
before a ticker's first run or during an outage would show nothing to every
visitor for up to an hour after the data was there. An empty result is a few
bytes, so reading it again costs nothing that matters.
"""

import functools
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

CACHE_SECONDS = 3600


class _EmptyResultError(Exception):
    """Raised inside the cached function, carrying the result: Streamlit keeps
    nothing from a call that raised."""

    def __init__(self, result: Any) -> None:
        super().__init__("an empty result is not cached")
        self.result = result


def is_empty(result: Any) -> bool:
    """True for None and for a frame, series, dict, list or tuple with
    nothing in it."""
    if result is None:
        return True
    if isinstance(result, pd.DataFrame | pd.Series):
        return result.empty
    if isinstance(result, dict | list | tuple):
        return len(result) == 0
    return False


def cached(loader: Callable[..., Any]) -> Callable[..., Any]:
    """loader with its results kept for CACHE_SECONDS, except an empty one.

    Streamlit keys a cache by the function's module, name and source, which
    functools.wraps passes on from loader, so each loader keeps its own.
    """

    @st.cache_data(ttl=CACHE_SECONDS)
    @functools.wraps(loader)
    def kept(*args: Any, **kwargs: Any) -> Any:
        result = loader(*args, **kwargs)
        if is_empty(result):
            raise _EmptyResultError(result)
        return result

    @functools.wraps(loader)
    def load(*args: Any, **kwargs: Any) -> Any:
        try:
            return kept(*args, **kwargs)
        except _EmptyResultError as empty:
            return empty.result

    return load
