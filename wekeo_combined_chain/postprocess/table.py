"""
Plume summary tables.

Splits the postprocess plume DataFrame into *plumes* and *tiny plumes*,
returning clean DataFrames with readable column names.

Usage
-----
    from wekeo_combined_chain.postprocess import table as T

    df_p = T.plume_table(df_plumes)
    df_t = T.tiny_plume_table(df_plumes)
"""

from __future__ import annotations

import pandas as pd

TINY_PLUME_THRESHOLD = 100

_COLUMN_MAP = {
    "label": "Plume label",
    "n_pixels_plume": "Number of pixels",
    "centroid_lat_plume": "Latitude (°)",
    "centroid_lon_plume": "Longitude (°)",
}


def _table(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Select columns and rename."""
    subset = df[list(_COLUMN_MAP.keys())].copy()
    subset.columns = _COLUMN_MAP.values()
    return subset


def plume_table(
    df: pd.DataFrame,
    threshold: int = TINY_PLUME_THRESHOLD,
) -> pd.DataFrame:
    """Plumes with ``label < threshold``."""
    if df.empty or "label" not in df.columns:
        return pd.DataFrame(columns=list(_COLUMN_MAP.values()))
    return _table(df[df["label"] < threshold].reset_index(drop=True), threshold)

 
def tiny_plume_table(
    df: pd.DataFrame,
    threshold: int = TINY_PLUME_THRESHOLD,
) -> pd.DataFrame:
    """Tiny plumes with ``label >= threshold``."""
    if df.empty or "label" not in df.columns:
        return pd.DataFrame(columns=list(_COLUMN_MAP.values()))
    return _table(df[df["label"] >= threshold].reset_index(drop=True), threshold)
