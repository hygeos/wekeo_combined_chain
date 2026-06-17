"""
Map-based timeseries aggregation plots for combined datasets.

For each pixel, compute the number of days matching a criterion and display
the result as a spatial map using Cartopy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extent_from_ds(ds: xr.Dataset) -> Tuple[float, float, float, float]:
    """Return (lon_min, lon_max, lat_min, lat_max) from dataset coords."""
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())


def _base_map(
    extent: Tuple[float, float, float, float],
    figsize: Tuple[int, int] = (18, 9),
) -> Tuple[Figure, plt.Axes]:
    """Create a Cartopy PlateCarree figure constrained to *extent*."""
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cf.COASTLINE, linewidth=0.6)
    ax.add_feature(cf.BORDERS, linewidth=0.4, linestyle="--")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    return fig, ax


def _finish_map(
    fig: Figure,
    ax: plt.Axes,
    title: str,
    save_fig_path: Optional[Union[str, Path]] = None,
) -> None:
    """Add title, tighten layout, and either save or show."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_fig_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_plume_occurrence_map(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (18, 9),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None,
) -> Tuple[Figure, plt.Axes]:
    """
    Map of the number of days each pixel falls within a plume or tiny plume.

    A pixel is counted if ``s5p_pca__plume_labels`` is > 0 (both normal plumes
    with label < 100 and tiny plumes with label >= 100).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``time`` dimension and ``s5p_pca__plume_labels`` variable.
    figsize : tuple, optional
        Figure size, by default (18, 9).
    title : str, optional
        Plot title.  If *None* an automatic title is generated.
    ax : Axes, optional
        Pre-existing Cartopy axes.  If *None* a new figure is created.
    save_fig_path : str or Path, optional
        When provided the figure is saved to this path instead of being shown.

    Returns
    -------
    tuple[Figure, Axes]
    """
    if "s5p_pca__plume_labels" not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__plume_labels' variable")
    if "time" not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    plume_labels = ds["s5p_pca__plume_labels"]

    # For each pixel, count the number of days with label > 0
    occurrence = (plume_labels > 0).sum(dim="time")

    extent = _extent_from_ds(ds)

    if ax is None:
        fig, ax = _base_map(extent, figsize=figsize)
    else:
        fig = ax.get_figure()

    lon, lat = ds["longitude"].values, ds["latitude"].values
    im = ax.pcolormesh(
        lon, lat, occurrence.values,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.7)
    cbar.set_label("Number of days with plume", fontsize=11)

    if title is None:
        times = ds.time.values
        title = (
            f"Plume Occurrence "
            f"({np.datetime_as_string(times[0], unit='D')} to "
            f"{np.datetime_as_string(times[-1], unit='D')})"
        )

    _finish_map(fig, ax, title, save_fig_path)
    return fig, ax


def plot_detection_occurrence_map(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (18, 9),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None,
) -> Tuple[Figure, plt.Axes]:
    """
    Map of the number of days each pixel has a valid CO detection.

    A pixel is considered detected when ``s5p_pca__mean_score_CO`` is finite
    (not NaN, not Inf, i.e. not a fill value).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``time`` dimension and ``s5p_pca__mean_score_CO`` variable.
    figsize : tuple, optional
        Figure size, by default (18, 9).
    title : str, optional
        Plot title.  If *None* an automatic title is generated.
    ax : Axes, optional
        Pre-existing Cartopy axes.  If *None* a new figure is created.
    save_fig_path : str or Path, optional
        When provided the figure is saved to this path instead of being shown.

    Returns
    -------
    tuple[Figure, Axes]
    """
    if "s5p_pca__mean_score_CO" not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__mean_score_CO' variable")
    if "time" not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    score = ds["s5p_pca__mean_score_CO"]

    # For each pixel, count the number of days with a finite value
    occurrence = np.isfinite(score).sum(dim="time")

    extent = _extent_from_ds(ds)

    if ax is None:
        fig, ax = _base_map(extent, figsize=figsize)
    else:
        fig = ax.get_figure()

    lon, lat = ds["longitude"].values, ds["latitude"].values
    im = ax.pcolormesh(
        lon, lat, occurrence.values,
        transform=ccrs.PlateCarree(),
        cmap="YlGnBu",
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.7)
    cbar.set_label("Number of days with detection", fontsize=11)

    if title is None:
        times = ds.time.values
        title = (
            f"Detection Occurrence (s5p_pca__mean_score_CO) "
            f"({np.datetime_as_string(times[0], unit='D')} to "
            f"{np.datetime_as_string(times[-1], unit='D')})"
        )

    _finish_map(fig, ax, title, save_fig_path)
    return fig, ax


def plot_fire_occurrence_map(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (18, 9),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None,
) -> Tuple[Figure, plt.Axes]:
    """
    Map of the number of days each pixel has an SLSTR active fire detection.

    A pixel is counted when at least one of ``frp_slstr__day_FRP_MWIR_mean``
    or ``frp_slstr__night_FRP_MWIR_mean`` is finite (not NaN, not fill value).
    If neither variable is present, a ``ValueError`` is raised.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``time`` dimension and at least one FRP variable.
    figsize : tuple, optional
        Figure size, by default (18, 9).
    title : str, optional
        Plot title.  If *None* an automatic title is generated.
    ax : Axes, optional
        Pre-existing Cartopy axes.  If *None* a new figure is created.
    save_fig_path : str or Path, optional
        When provided the figure is saved to this path instead of being shown.

    Returns
    -------
    tuple[Figure, Axes]
    """
    day_var = "frp_slstr__day_FRP_MWIR_mean"
    night_var = "frp_slstr__night_FRP_MWIR_mean"

    has_day = day_var in ds.data_vars
    has_night = night_var in ds.data_vars

    if not has_day and not has_night:
        raise ValueError(
            f"Dataset must contain at least one of '{day_var}' or '{night_var}'"
        )
    if "time" not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    # Build a boolean mask: True on days where at least one FRP variable is finite
    fire_detected = None
    if has_day:
        fire_detected = np.isfinite(ds[day_var])
    if has_night:
        night_mask = np.isfinite(ds[night_var])
        fire_detected = night_mask if fire_detected is None else (fire_detected | night_mask)

    occurrence = fire_detected.sum(dim="time")

    extent = _extent_from_ds(ds)

    if ax is None:
        fig, ax = _base_map(extent, figsize=figsize)
    else:
        fig = ax.get_figure()

    lon, lat = ds["longitude"].values, ds["latitude"].values
    im = ax.pcolormesh(
        lon, lat, occurrence.values,
        transform=ccrs.PlateCarree(),
        cmap="OrRd",
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.7)
    cbar.set_label("Number of days with active fire", fontsize=11)

    if title is None:
        times = ds.time.values
        title = (
            f"SLSTR Active Fire Occurrence "
            f"({np.datetime_as_string(times[0], unit='D')} to "
            f"{np.datetime_as_string(times[-1], unit='D')})"
        )

    _finish_map(fig, ax, title, save_fig_path)
    return fig, ax
