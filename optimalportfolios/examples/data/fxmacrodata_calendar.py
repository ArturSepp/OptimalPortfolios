"""FXMacroData macro calendar helper for portfolio examples."""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import requests

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def load_fxmacrodata_calendar(
    currency: str = "usd",
    *,
    limit: int = 100,
    min_tier: Optional[int] = 2,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Load FXMacroData release-calendar rows as a DataFrame."""

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    token = api_key or os.getenv("FXMACRODATA_API_KEY")
    if token:
        params["api_key"] = token

    response = requests.get(
        f"{FXMACRODATA_BASE_URL}/calendar/{currency.lower()}",
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    rows: list[dict[str, Any]] = response.json().get("data", [])
    if min_tier is not None:
        rows = [
            row
            for row in rows
            if int(row.get("market_tier") or 99) <= min_tier
        ]

    data = pd.DataFrame(rows[:limit_count])
    if not data.empty and "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
    if not data.empty and "announcement_datetime" in data.columns:
        data["announcement_datetime"] = pd.to_datetime(
            data["announcement_datetime"],
            unit="s",
            utc=True,
        )
    return data
