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
    lon_min, lon_max, lat_min, lat_max = extent
    is_global = (lon_max - lon_min) >= 350 and (lat_max - lat_min) >= 170
    if is_global:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=proj)
    ax.add_feature(cf.COASTLINE, linewidth=0.6)
    ax.add_feature(cf.BORDERS, linewidth=0.4, linestyle="--")
    gl = ax.gridlines(draw_labels=not is_global, linewidth=0.3, alpha=0.5, linestyle="--")
    if not is_global:
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

    A pixel is counted when ``frp_slstr__day_FRP_MWIR_mean`` is finite
    (not NaN, not fill value).

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

    if day_var not in ds.data_vars:
        raise ValueError(f"Dataset must contain '{day_var}'")
    if "time" not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    occurrence = np.isfinite(ds[day_var]).sum(dim="time")

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


# ---------------------------------------------------------------------------
# Daily map animations (interactive slider + auto-play + GIF export)
# ---------------------------------------------------------------------------

def _run_animation(
    ds: xr.Dataset,
    da: xr.DataArray,
    label: str,
    cbar_label: str,
    cmap,
    norm,
    figsize: Tuple[int, int],
    save_gif_path: Optional[Union[str, Path]],
    fps: int,
    dpi: int,
):
    """
    Shared rendering engine.

    Interactive mode pre-renders every frame as a PNG while reusing a single
    Cartopy figure. This avoids rebuilding the map for every frame.

    GIF mode renders frames sequentially using FuncAnimation.
    """
    import io
    import matplotlib.animation as mpl_anim

    times = ds.time.values
    n_days = len(times)

    extent = _extent_from_ds(ds)
    lon = ds["longitude"].values
    lat = ds["latitude"].values

    start_str = np.datetime_as_string(times[0], unit="D")
    end_str = np.datetime_as_string(times[-1], unit="D")

    def _frame_title(day_str: str) -> str:
        return f"{label}  ·  {day_str}  [{start_str} → {end_str}]"

    # ------------------------------------------------------------------ GIF --
    if save_gif_path is not None:
        save_gif_path = Path(save_gif_path)

        fig, ax = _base_map(extent, figsize=figsize)

        im = ax.pcolormesh(
            lon,
            lat,
            da.isel(time=0).values,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="auto",
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            shrink=0.7,
        )
        cbar.set_label(cbar_label, fontsize=11)

        title = ax.set_title("", fontsize=13, fontweight="bold", pad=15)

        plt.tight_layout()

        def _update(frame):
            im.set_array(da.isel(time=frame).values.ravel())
            title.set_text(
                _frame_title(
                    np.datetime_as_string(times[frame], unit="D")
                )
            )
            return (im, title)

        anim = mpl_anim.FuncAnimation(
            fig,
            _update,
            frames=n_days,
            interval=1000 // fps,
            blit=False,
        )

        save_gif_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(
            save_gif_path,
            writer="pillow",
            fps=fps,
            dpi=dpi,
        )

        plt.close(fig)
        print(f"GIF saved to: {save_gif_path}")
        return None

    # ------------------------------------------------ Pre-render PNG frames --
    from ipywidgets import (
        Play,
        IntSlider,
        HBox,
        VBox,
        Image as IpyImage,
        jslink,
    )
    from IPython.display import display

    print(f"Pre-rendering {n_days} frames…", end=" ", flush=True)

    # Build the map only once
    fig, ax = _base_map(extent, figsize=figsize)

    im = ax.pcolormesh(
        lon,
        lat,
        da.isel(time=0).values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        shrink=0.7,
    )
    cbar.set_label(cbar_label, fontsize=11)

    title = ax.set_title("", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()

    frames: list[bytes] = []

    for i in range(n_days):
        im.set_array(da.isel(time=i).values.ravel())

        title.set_text(
            _frame_title(
                np.datetime_as_string(times[i], unit="D")
            )
        )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
        frames.append(buf.getvalue())

    plt.close(fig)

    print("done.")

    img_widget = IpyImage(
        value=frames[0],
        format="png",
        layout={"max_width": "100%"},
    )

    play = Play(
        value=0,
        min=0,
        max=n_days - 1,
        step=1,
        interval=max(200, 1000 // fps),
        description="▶",
    )

    day_slider = IntSlider(
        value=0,
        min=0,
        max=n_days - 1,
        description="Day",
        layout={"width": "500px"},
    )

    fps_slider = IntSlider(
        value=fps,
        min=1,
        max=10,
        step=1,
        description="FPS",
        layout={"width": "200px"},
    )

    jslink((play, "value"), (day_slider, "value"))

    day_slider.observe(
        lambda change: setattr(
            img_widget,
            "value",
            frames[change["new"]],
        ),
        names="value",
    )

    fps_slider.observe(
        lambda change: setattr(
            play,
            "interval",
            max(100, 1000 // change["new"]),
        ),
        names="value",
    )

    return display(
        VBox(
            [
                HBox([play, day_slider, fps_slider]),
                img_widget,
            ]
        )
    )


def animate_plume_map(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (18, 9),
    save_gif_path: Optional[Union[str, Path]] = None,
    fps: int = 2,
    dpi: int = 100,
):
    """
    Animate daily plume presence maps (any ``s5p_pca__plume_labels > 0``).

    Pixels with a plume are shown in orange; all others are transparent.
    Regular plumes (label < 100) and tiny plumes (label ≥ 100) are treated
    identically.

    Interactive mode shows a Play/loop widget + day slider + FPS control with
    pre-rendered frames (no flicker).  Pass *save_gif_path* to export a GIF.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    if "s5p_pca__plume_labels" not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__plume_labels'")
    if "time" not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    labels = ds["s5p_pca__plume_labels"]
    # 1.0 where any plume present, NaN elsewhere
    da = xr.where(labels > 0, 1.0, np.nan)

    cmap = ListedColormap(["#E8671B"])  # single orange colour
    norm = BoundaryNorm([0.5, 1.5], ncolors=1)

    return _run_animation(
        ds, da,
        label="S5P-PCA Plume Presence",
        cbar_label="Plume present",
        cmap=cmap, norm=norm,
        figsize=figsize,
        save_gif_path=save_gif_path,
        fps=fps, dpi=dpi,
    )


def animate_detection_map(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (18, 9),
    save_gif_path: Optional[Union[str, Path]] = None,
    fps: int = 2,
    dpi: int = 100,
):
    """
    Animate daily S5P-PCA CO score maps (``s5p_pca__mean_score_CO``).

    Pixels with no detection are transparent (NaN).  The colorscale is fixed
    across all frames using the 2nd–98th percentile of the full period so
    relative intensities are comparable day-to-day.

    Interactive mode shows a Play/loop widget + day slider with pre-rendered
    frames (no flicker).  Pass *save_gif_path* to export a GIF instead.
    """
    from matplotlib.colors import Normalize

    if "s5p_pca__mean_score_CO" not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__mean_score_CO'")
    if "time" not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    score = ds["s5p_pca__mean_score_CO"]
    da = score.where(np.isfinite(score))  # NaN where not detected

    # Consistent colorscale across the whole period
    all_vals = da.values[np.isfinite(da.values)]
    vmin = float(np.percentile(all_vals, 2))  if len(all_vals) else 0.0
    vmax = float(np.percentile(all_vals, 98)) if len(all_vals) else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    return _run_animation(
        ds, da,
        label="S5P-PCA Mean CO Score",
        cbar_label="Mean CO score (–)",
        cmap="YlOrRd", norm=norm,
        figsize=figsize,
        save_gif_path=save_gif_path,
        fps=fps, dpi=dpi,
    )


def animate_fire_map(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (18, 9),
    log_scale: bool = True,
    save_gif_path: Optional[Union[str, Path]] = None,
    fps: int = 2,
    dpi: int = 100,
):
    """
    Animate daily SLSTR active fire FRP maps (``frp_slstr__day_FRP_MWIR_mean``).

    Pixels with no fire (NaN or ≤ 0) are transparent.  The colorscale is fixed
    across all frames using the 2nd–98th percentile of positive values.
    By default a log scale is used (``log_scale=True``).

    Interactive mode shows a Play/loop widget + day slider + FPS control with
    pre-rendered frames (no flicker).  Pass *save_gif_path* to export a GIF.
    """
    from matplotlib.colors import LogNorm, Normalize

    day_var = "frp_slstr__day_FRP_MWIR_mean"

    if day_var not in ds.data_vars:
        raise ValueError(f"Dataset must contain '{day_var}'")
    if "time" not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    da = ds[day_var].where(np.isfinite(ds[day_var]) & (ds[day_var] > 0))

    # Fixed colorscale across the full period
    all_vals = da.values[np.isfinite(da.values) & (da.values > 0)]
    vmin = float(np.percentile(all_vals, 2))  if len(all_vals) else 1.0
    vmax = float(np.percentile(all_vals, 98)) if len(all_vals) else 1e4

    norm = LogNorm(vmin=max(vmin, 1e-3), vmax=vmax) if log_scale \
           else Normalize(vmin=vmin, vmax=vmax)
    scale_str = "log scale" if log_scale else "linear scale"

    return _run_animation(
        ds, da,
        label=f"SLSTR Active Fire — FRP MWIR day ({scale_str})",
        cbar_label="FRP MWIR day (W m⁻²)",
        cmap="inferno", norm=norm,
        figsize=figsize,
        save_gif_path=save_gif_path,
        fps=fps, dpi=dpi,
    )


def animate_score_CO_map(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (18, 9),
    save_gif_path: Optional[Union[str, Path]] = None,
    fps: int = 2,
    dpi: int = 100,
):
    """
    Animate daily S5P-PCA CO score maps (``s5p_pca__mean_score_CO``).

    Pixels with no value (NaN/Inf) are transparent.  The colorscale is fixed
    across all frames using the 2nd–98th percentile of the full period so
    relative intensities are comparable day-to-day.

    Interactive mode shows a Play/loop widget + day slider + FPS control with
    pre-rendered frames (no flicker).  Pass *save_gif_path* to export a GIF.
    """
    from matplotlib.colors import Normalize

    var = "s5p_pca__mean_score_CO"

    if var not in ds.data_vars:
        raise ValueError(f"Dataset must contain '{var}'")
    if "time" not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    da = ds[var].where(np.isfinite(ds[var]))  # NaN where not detected

    # Consistent colorscale across the whole period
    all_vals = da.values[np.isfinite(da.values)]
    vmin = float(np.percentile(all_vals, 2))  if len(all_vals) else 0.0
    vmax = float(np.percentile(all_vals, 98)) if len(all_vals) else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    return _run_animation(
        ds, da,
        label="S5P-PCA Mean CO Score",
        cbar_label="Mean CO score (–)",
        cmap="YlOrRd", norm=norm,
        figsize=figsize,
        save_gif_path=save_gif_path,
        fps=fps, dpi=dpi,
    )
