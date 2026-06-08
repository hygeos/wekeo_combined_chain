"""
Interactive plotting routines for the postprocess pipeline.

Differences from spascia_prototype_plots.py
--------------------------------------------
* The input datasets are assumed to be **already cropped** to the area of
  interest; the full extent of the data is always plotted (no hardcoded
  global or regional zoom logic).
* Every function is **public** and returns the ``matplotlib.figure.Figure``
  so results display inline in Jupyter without extra calls.
* An optional *save_to* parameter writes the figure to disk when provided.

Usage
-----
    from wekeo_combined_chain.postprocess import plot as P

    ds_combined = xr.open_dataset(...)
    ds_post, df_plumes = postprocess.compute(ds_combined)

    # Inline in a notebook
    fig = P.plot_plumes_frp(ds_combined, ds_post, date_str="20210817")

    # Save to disk
    fig = P.plot_fire_score_plume(
        ds_combined, ds_post, df_plumes,
        date_str="20210817", band="SWIR",
        save_to="./plots/fire_score_SWIR_20210817.png",
    )

    # Generate and save all plots at once
    figs = P.plot_all(ds_combined, ds_post, df_plumes,
                      date_str="20210817", output_dir="./plots")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.ndimage import binary_dilation
from ._params import FRP_VAR_MAP

# ---------------------------------------------------------------------------
# Module-level style constants
# ---------------------------------------------------------------------------

_PALETTE = [
    ("#1f77b4", "#aec7e8"), ("#2ca02c", "#98df8a"), ("#d62728", "#ff9896"),
    ("#9467bd", "#c5b0d5"), ("#8c564b", "#c49c94"), ("#e377c2", "#f7b6d2"),
    ("#bcbd22", "#dbdb8d"), ("#17becf", "#9edae5"), ("#ff7f0e", "#ffbb78"),
    ("#7f7f7f", "#c7c7c7"),
]
_COLORS60 = (list(plt.cm.get_cmap("tab20").colors)
           + list(plt.cm.get_cmap("tab20b").colors)
           + list(plt.cm.get_cmap("tab20c").colors))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extent_from_ds(ds: xr.Dataset) -> tuple[float, float, float, float]:
    """Return (lon_min, lon_max, lat_min, lat_max) from a dataset's coords."""
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())


def _base_map(extent: tuple[float, float, float, float], figsize=(18, 9)) -> tuple[Figure, plt.Axes]:
    """Build a Cartopy PlateCarree figure constrained to *extent*."""
    lon_min, lon_max, lat_min, lat_max = extent
    xticks = np.arange(np.floor(lon_min / 10) * 10, lon_max + 10, 10)
    yticks = np.arange(np.floor(lat_min / 10) * 10, lat_max + 10, 10)

    fig = plt.figure(figsize=figsize)
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, linestyle=":", color="k", alpha=0.5)
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    fig.canvas.draw()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return fig, ax


def _extract_grids(
    ds_combined: xr.Dataset, ds_post: xr.Dataset
) -> tuple:
    """Pull out all 2-D arrays needed by the plot functions."""
    lat = ds_combined["latitude"].values
    lon = ds_combined["longitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    labels_2d       = ds_combined["s5p_pca__plume_labels"].values.astype(int)
    frp_active_mask = ds_post["frp_active_mask"].values.astype(bool)

    frp_data: dict[str, np.ndarray] = {}
    for short, prefixed in FRP_VAR_MAP.items():
        if prefixed in ds_combined:
            frp_data[short] = ds_combined[prefixed].values
        else:
            frp_data[short] = np.full(labels_2d.shape, np.nan)

    return lat, lon, lat_2d, lon_2d, labels_2d, frp_data, frp_active_mask


def _maybe_save(fig: Figure, save_to: str | Path | None, dpi: int = 150) -> None:
    if save_to is not None:
        path = Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=dpi)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_plumes(
    ds_combined: xr.Dataset,
    ds_post: xr.Dataset,
    date_str: str,
    save_to: str | Path | None = None,
    figsize: tuple[int, int] = (18, 9),
) -> Figure:
    """
    Map — Plume pixels coloured by label (no FRP overlay).

    Parameters
    ----------
    ds_combined :
        Raw combined input dataset.
    ds_post :
        Output of ``postprocess.compute()``.
    date_str :
        Date string used in the figure title, e.g. ``"20210817"``.
    save_to :
        File path to save the figure. ``None`` (default) skips saving.
    figsize :
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    extent = _extent_from_ds(ds_combined)
    _, _, lat_2d, lon_2d, labels_2d, _, _ = _extract_grids(
        ds_combined, ds_post)

    fig, ax = _base_map(extent, figsize=figsize)

    num_plumes = np.unique(labels_2d)
    num_plumes = num_plumes[num_plumes != 0]
    for i, lbl in enumerate(num_plumes):
        mask = labels_2d == lbl
        ax.scatter(lon_2d[mask], lat_2d[mask],
                   s=2, color=_COLORS60[i % len(_COLORS60)], alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=3)

    ax.scatter([], [], s=6, color="steelblue", alpha=0.6,
               transform=ccrs.PlateCarree(), label="Plumes S5P-PCA")
    ax.set_title(
        f"Plumes S5P-PCA ({len(num_plumes)}) — {date_str}",
        fontsize=16,
    )
    ax.legend(loc="lower left", fontsize=14, markerscale=2)

    _maybe_save(fig, save_to)
    plt.close(fig)
    return fig


def plot_plumes_frp(
    ds_combined: xr.Dataset,
    ds_post: xr.Dataset,
    date_str: str,
    frp_channel: Literal["SWIR", "MWIR"] = "SWIR",
    save_to: str | Path | None = None,
    figsize: tuple[int, int] = (18, 9),
) -> Figure:
    """
    Map 1 — Plume pixels coloured by label, overlaid with FRP detections.

    Parameters
    ----------
    ds_combined :
        Raw combined input dataset.
    ds_post :
        Output of ``postprocess.compute()``.
    date_str :
        Date string used in the figure title, e.g. ``"20210817"``.
    frp_channel :
        ``"SWIR"`` or ``"MWIR"``.
    save_to :
        File path to save the figure. ``None`` (default) skips saving.
    figsize :
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    frp_key   = "FRP_SWIR_no_SAA_mean" if frp_channel == "SWIR" else "FRP_MWIR_mean"
    frp_label = "FRP SWIR no SAA"      if frp_channel == "SWIR" else "FRP MWIR"

    extent = _extent_from_ds(ds_combined)
    _, _, lat_2d, lon_2d, labels_2d, frp_data, frp_active_mask = _extract_grids(
        ds_combined, ds_post)

    fig, ax = _base_map(extent, figsize=figsize)
    n_plot_frp = 0

    if frp_key in frp_data:
        frp_val = frp_data[frp_key]
        mask_ns = frp_active_mask & ~np.isnan(frp_val) & (frp_val > 0)
        if mask_ns.sum() > 0:
            n_plot_frp = int(mask_ns.sum())
            ax.scatter(lon_2d[mask_ns], lat_2d[mask_ns],
                       s=12, c="navy", marker="x", alpha=0.2,
                       transform=ccrs.PlateCarree(), zorder=2, label=frp_label)

    num_plumes = np.unique(labels_2d)
    num_plumes = num_plumes[num_plumes != 0]
    for i, lbl in enumerate(num_plumes):
        mask = labels_2d == lbl
        ax.scatter(lon_2d[mask], lat_2d[mask],
                   s=2, color=_COLORS60[i % len(_COLORS60)], alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=3)

    ax.scatter([], [], s=6, color="steelblue", alpha=0.6,
               transform=ccrs.PlateCarree(), label="Plumes S5P-PCA")
    ax.set_title(
        f"Combination Plumes S5P-PCA ({len(num_plumes)}) × {frp_label} ({n_plot_frp}) — {date_str}",
        fontsize=16,
    )
    ax.legend(loc="lower left", fontsize=14, markerscale=2)

    _maybe_save(fig, save_to)
    plt.close(fig)
    return fig


def plot_fire_score_plume(
    ds_combined: xr.Dataset,
    ds_post: xr.Dataset,
    df_plumes: pd.DataFrame,
    date_str: str,
    band: Literal["SWIR", "MWIR"] = "SWIR",
    save_to: str | Path | None = None,
    figsize: tuple[int, int] = (18, 9),
) -> Figure | str:
    """
    Map 2 — Plume pixels coloured by per-plume fire score (log scale),
    with source-location stars.

    Parameters
    ----------
    band :
        ``"SWIR"`` or ``"MWIR"``.
    save_to :
        File path to save the figure, or ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    col_var   = f"fire_score_{band}_plume"
    cmap_name = "jet" if band == "SWIR" else "plasma"

    if df_plumes.empty or col_var not in df_plumes.columns:
        return "No data to plot"  # or raise an exception, or return an empty figure

    extent = _extent_from_ds(ds_combined)
    _, _, lat_2d, lon_2d, labels_2d, _, _ = _extract_grids(ds_combined, ds_post)

    fig, ax = _base_map(extent, figsize=figsize)
    valid_scores = df_plumes.dropna(subset=[col_var])[col_var]

    if len(valid_scores) > 0:
        norm_score = LogNorm(vmin=valid_scores.min() + 1e-3, vmax=valid_scores.max())
        cmap_score = plt.cm.get_cmap(cmap_name)

        df_no_fire   = df_plumes[df_plumes[col_var].isna()]
        df_with_fire = df_plumes.dropna(subset=[col_var]).sort_values(col_var)

        for _, r in df_no_fire.iterrows():
            mask = labels_2d == int(r["label"])
            if not mask.any():
                continue
            ax.scatter(lon_2d[mask], lat_2d[mask],
                       s=3, color="gray", alpha=0.7,
                       transform=ccrs.PlateCarree(), zorder=3)

        star_col_map = {"high": "red", "low": "gold"}
        for _, r in df_with_fire.iterrows():
            mask = labels_2d == int(r["label"])
            if not mask.any():
                continue
            color = cmap_score(norm_score(r[col_var]))
            ax.scatter(lon_2d[mask], lat_2d[mask],
                       s=3, color=color, alpha=0.75,
                       transform=ccrs.PlateCarree(), zorder=4)

            src_loc = f"source_is_localized_{band}_count_plume"
            src_lat = f"source_lat_{band}_count_plume"
            src_lon = f"source_lon_{band}_count_plume"
            conf_col = f"source_confidence_label_{band}_plume"
            if src_loc in r and r[src_loc] == 1:
                star_c = star_col_map.get(r.get(conf_col, ""), "gold")
                ax.plot(r[src_lon], r[src_lat], marker="*",
                        color=star_c, markersize=10,
                        markeredgecolor="k", markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(), zorder=15)

        sm = plt.cm.ScalarMappable(cmap=cmap_score, norm=norm_score)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.5, label=f"{col_var} (log)")
        ax.scatter([], [], s=6, color="gray",
                   label=f"No FRP ({len(df_no_fire)})")
        ax.plot([], [], marker="*", color="gold", markersize=6,
                markeredgecolor="k", linestyle="None", label="Source")
        ax.legend(loc="lower left", fontsize=12)

    ax.set_title(f"S5P-PCA plumes — {col_var} — {date_str}", fontsize=14)

    _maybe_save(fig, save_to)
    plt.close(fig)
    return fig


def plot_fire_score_pixel(
    ds_combined: xr.Dataset,
    ds_post: xr.Dataset,
    date_str: str,
    band: Literal["SWIR", "MWIR"] = "SWIR",
    save_to: str | Path | None = None,
    figsize: tuple[int, int] = (18, 9),
) -> Figure:
    """
    Map 3 — Per-pixel fire score rendered as a pcolormesh (log scale).

    Parameters
    ----------
    band :
        ``"SWIR"`` or ``"MWIR"``.
    save_to :
        File path to save the figure, or ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    extent = _extent_from_ds(ds_combined)
    lat, lon, *_ = _extract_grids(ds_combined, ds_post)

    grid_px = ds_post[f"fire_score_{band}"].values

    fig, ax = _base_map(extent, figsize=figsize)
    valid_px = grid_px[~np.isnan(grid_px)]
    if len(valid_px) > 0:
        norm_px = LogNorm(vmin=valid_px.min() + 1e-6, vmax=valid_px.max())
        cmap_px = plt.cm.get_cmap("inferno")
        sc = ax.pcolormesh(lon, lat, grid_px,
                           norm=norm_px, cmap=cmap_px,
                           transform=ccrs.PlateCarree(), zorder=3)
        plt.colorbar(
            sc, ax=ax, shrink=0.5,
            label=(f"fire_score_{band} = score_CO(pixel) "
                   f"× log(1 + FRP_{band}_energy_plume) (log scale)"),
        )
    ax.set_title(
        f"S5P-PCA plumes — fire_score_{band} per pixel — {date_str}", fontsize=14)

    _maybe_save(fig, save_to)
    plt.close(fig)
    return fig


def plot_plume_envelopes(
    ds_combined: xr.Dataset,
    ds_post: xr.Dataset,
    df_plumes: pd.DataFrame,
    date_str: str,
    frp_channel: Literal["SWIR", "MWIR"] = "SWIR",
    save_to: str | Path | None = None,
    figsize: tuple[int, int] = (18, 9),
) -> Figure:
    """
    Map 4 — Plume pixels with dilation-based search envelopes, FRP detections,
    and source-location confidence stars, over the full dataset extent.

    Parameters
    ----------
    frp_channel :
        ``"SWIR"`` or ``"MWIR"``.
    save_to :
        File path to save the figure, or ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    frp_key   = "FRP_SWIR_no_SAA_mean" if frp_channel == "SWIR" else "FRP_MWIR_mean"
    frp_label = "FRP SWIR no SAA"      if frp_channel == "SWIR" else "FRP MWIR"

    conf_color_map = {"high": "r", "low": "gold"}

    extent = _extent_from_ds(ds_combined)
    lon_min, lon_max, lat_min, lat_max = extent
    _, _, lat_2d, lon_2d, labels_2d, frp_data, frp_active_mask = _extract_grids(
        ds_combined, ds_post)

    fig, ax = _base_map(extent, figsize=figsize)

    src_col       = frp_channel
    src_lat_count = f"source_lat_{src_col}_count_plume"
    src_lon_count = f"source_lon_{src_col}_count_plume"
    src_loc_count = f"source_is_localized_{src_col}_count_plume"
    src_lat_sum   = f"source_lat_{src_col}_sum_plume"
    src_lon_sum   = f"source_lon_{src_col}_sum_plume"
    src_loc_sum   = f"source_is_localized_{src_col}_sum_plume"
    conf_col_key  = f"source_confidence_label_{src_col}_plume"

    has_envelope = has_src_count = has_src_sum = False

    for i, row in df_plumes.iterrows():
        lbl = int(row["label"])
        col_plume, col_env = _PALETTE[i % len(_PALETTE)]

        buf          = int(row["buffer_pixels_plume"])
        s5p_mask     = (labels_2d == lbl)
        s5p_mask_ext = binary_dilation(s5p_mask, iterations=buf)
        s5p_mask_ext &= (labels_2d == lbl) | (labels_2d == 0)
        env_mask     = s5p_mask_ext & ~s5p_mask

        if env_mask.any():
            has_envelope = True
            ax.contour(
                lon_2d, lat_2d, env_mask.astype(float),
                levels=[0.5], colors=[col_env], linestyles="dashed",
                linewidths=1.2, transform=ccrs.PlateCarree(), zorder=4,
            )

        if s5p_mask.any():
            ax.scatter(lon_2d[s5p_mask], lat_2d[s5p_mask],
                       s=2, color=col_plume,
                       transform=ccrs.PlateCarree(), zorder=3)

        conf_label = row.get(conf_col_key, "none")
        star_color = conf_color_map.get(conf_label)

        if star_color and src_loc_count in row and row[src_loc_count] == 1:
            slat, slon = row[src_lat_count], row[src_lon_count]
            if (lon_min <= slon <= lon_max) and (lat_min <= slat <= lat_max):
                has_src_count = True
                ax.plot(slon, slat, marker="*", color=star_color, markersize=12,
                        markeredgecolor="k", markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(), zorder=21)

        if star_color and src_loc_sum in row and row[src_loc_sum] == 1:
            slat, slon = row[src_lat_sum], row[src_lon_sum]
            if (lon_min <= slon <= lon_max) and (lat_min <= slat <= lat_max):
                has_src_sum = True
                ax.plot(slon, slat, marker="*", color=star_color, markersize=8,
                        markeredgecolor="k", markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(), zorder=22)

    mask_ns = np.zeros(lat_2d.shape, dtype=bool)
    has_frp = False
    if frp_key in frp_data:
        frp_val = frp_data[frp_key]
        mask_ns = frp_active_mask & ~np.isnan(frp_val) & (frp_val > 0)
        if mask_ns.sum() > 0:
            has_frp = True
            ax.scatter(lon_2d[mask_ns], lat_2d[mask_ns],
                       s=12, c="navy", marker="x",
                       transform=ccrs.PlateCarree(), zorder=20)

    legend_handles = [plt.scatter([], [], s=6, alpha=0.75, color=_PALETTE[0][0],
                                  label="Plumes S5P-PCA")]
    if has_envelope:
        legend_handles.append(
            plt.scatter([], [], s=6, alpha=0.75, color=_PALETTE[0][1],
                        marker="s", label="Search envelope"))
    if has_frp:
        legend_handles.append(
            plt.scatter([], [], s=12, color="navy", marker="x", label=frp_label))
    for lbl_conf, col_conf in [("high", "red"), ("low", "gold")]:
        legend_handles.append(
            plt.plot([], [], marker="*", color=col_conf, markersize=8,
                     markeredgecolor="k", linestyle="None",
                     label=f"Confidence index : {lbl_conf}")[0])

    n_frp_vis   = int(mask_ns.sum())
    n_plume_vis = int((labels_2d > 0).sum())
    ax.set_title(
        f"Combination Plumes S5P-PCA ({n_plume_vis} cells) × {frp_label} "
        f"({n_frp_vis} cells) — {date_str}",
        fontsize=18,
    )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=16, markerscale=2)

    _maybe_save(fig, save_to)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def plot_all(
    ds_combined: xr.Dataset,
    ds_post: xr.Dataset,
    df_plumes: pd.DataFrame,
    date_str: str,
    output_dir: str | Path | None = None,
) -> dict[str, Figure]:
    """
    Generate all four map types and return them as a dict.

    Parameters
    ----------
    output_dir :
        If provided, each figure is saved to this directory using a
        standardised filename. The directory is created if absent.

    Returns
    -------
    dict mapping plot name → Figure, e.g.::

        {
            "plumes_frp_SWIR": <Figure>,
            "plumes_frp_MWIR": <Figure>,
            "fire_score_plume_SWIR": <Figure>,
            ...
        }
    """
    def _path(name: str) -> Path | None:
        if output_dir is None:
            return None
        return Path(output_dir) / f"{name}_{date_str}.png"

    figs: dict[str, Figure] = {}

    for ch in ("SWIR", "MWIR"):
        key = f"plumes_frp_{ch}"
        figs[key] = plot_plumes_frp(
            ds_combined, ds_post, date_str,
            frp_channel=ch, save_to=_path(key),
        )

    for band in ("SWIR", "MWIR"):
        key = f"fire_score_plume_{band}"
        figs[key] = plot_fire_score_plume(
            ds_combined, ds_post, df_plumes, date_str,
            band=band, save_to=_path(key),
        )

    for band in ("SWIR", "MWIR"):
        key = f"fire_score_pixel_{band}"
        figs[key] = plot_fire_score_pixel(
            ds_combined, ds_post, date_str,
            band=band, save_to=_path(key),
        )

    for ch in ("SWIR", "MWIR"):
        key = f"plume_envelopes_{ch}"
        figs[key] = plot_plume_envelopes(
            ds_combined, ds_post, df_plumes, date_str,
            frp_channel=ch, save_to=_path(key),
        )

    return figs
