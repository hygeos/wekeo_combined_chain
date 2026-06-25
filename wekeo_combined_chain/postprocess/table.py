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

# MAJ 11/06/2026 SP
_COLS_GENERAL = {
    "label"                  : "Plume label",
    "n_pixels_plume"         : "Size (cells)",
    "centroid_lat_plume"     : "Centroid lat (deg)",
    "centroid_lon_plume"     : "Centroid lon (deg)",
    "n_frp_cells_total_plume": "FRP cells found",
}

_COLS_FIRE = {
    "label"                 : "Plume label",
    "fire_score_MWIR_plume" : "Fire score MWIR (day)",
    "frp_energy_MWIR_plume" : "FRP energy MWIR (MW)",
    "mean_score_CO_plume"   : "Mean CO score (no units)",
}

_COLS_SOURCE = {
    "label"                              : "Plume label",
    "source_is_localized_MWIR_sum_plume" : "Source localized",
    "source_lat_MWIR_sum_plume"          : "Source lat (deg)",
    "source_lon_MWIR_sum_plume"          : "Source lon (deg)",
    "source_dist_km_MWIR_sum_plume"      : "Source dist to centroid (km)",
    "source_confidence_label_MWIR_plume" : "Source confidence",
}

# def _table(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
#     """Select columns and rename."""
#     subset = df[list(_COLUMN_MAP.keys())].copy()
#     subset.columns = _COLUMN_MAP.values()
#     return subset

# MAJ 11/06/2026 SP --------------------------------------------
def _subtable(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    cols = [c for c in col_map.keys() if c in df.columns]
    sub = df[cols].copy()
    sub.columns = [col_map[c] for c in cols]
    return sub

def _display_three_tables(df: pd.DataFrame, label: str) -> None:
    from IPython.display import display
    print(f"--- {label} - General info ---")
    display(_subtable(df, _COLS_GENERAL))
    print(f"--- {label} - Fire scores (MWIR/day) ---")
    display(_subtable(df, _COLS_FIRE))
    print(f"--- {label} - Source localization (MWIR/day) ---")
    display(_subtable(df, _COLS_SOURCE))


def display_plume_tables(
    df: pd.DataFrame,
    threshold: int = TINY_PLUME_THRESHOLD,
) -> None:
    """Display 3 sub-tables for plumes with label < threshold."""
    df_p = df[df["label"] < threshold].reset_index(drop=True)
    _display_three_tables(df_p, "Plumes")


def display_tiny_plume_tables(
    df: pd.DataFrame,
    threshold: int = TINY_PLUME_THRESHOLD,
) -> None:
    """Display 3 sub-tables for tiny plumes with label >= threshold."""
    df_t = df[df["label"] >= threshold].reset_index(drop=True)
    _display_three_tables(df_t, f"Tiny plumes (label >= {threshold})")

#----------------------------------------------------------------

# def plume_table(
#     df: pd.DataFrame,
#     threshold: int = TINY_PLUME_THRESHOLD,
# ) -> pd.DataFrame:
#     """Plumes with ``label < threshold``."""
#     return _table(df[df["label"] < threshold].reset_index(drop=True), threshold)


# def tiny_plume_table(
#     df: pd.DataFrame,
#     threshold: int = TINY_PLUME_THRESHOLD,
# ) -> pd.DataFrame:
#     """Tiny plumes with ``label >= threshold``."""
#     return _table(df[df["label"] >= threshold].reset_index(drop=True), threshold)
