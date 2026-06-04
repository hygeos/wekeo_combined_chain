"""
Plotting routines for the postprocess pipeline.

Adapted from combine_S5P-PCA_FRP_v6.py lines ~240–310 (base_map, map_check_plumes)
and lines ~940–1210 (the four main map types + regional zoom maps).

Usage
-----
    from wekeo_combined_chain import postprocess

    ds_combined = xr.open_dataset(...)          # raw combined input
    ds_post, df_plumes = postprocess.compute(ds_combined)
    postprocess.plot(ds_combined, ds_post, df_plumes, output_dir="./plots", date_str="20210817")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.colors import LogNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.ndimage import binary_dilation

from ._params import FRP_VAR_MAP

# ---------------------------------------------------------------------------
# Default regional zoom definitions — v6.py lines ~1190–1230
# ---------------------------------------------------------------------------
DEFAULT_ZONES: list[tuple[str, float, float, float, float]] = [
    ("USA",        -125, -110,  35,  45),
    ("Siberie",      60,  135,  45,  75),
    ("Siberie_bis", 120,  150,  55,  70),
    ("Afrique",      10,   30, -20,   4),
    ("Canada_US",  -125,  -95,  40,  65),
]

_XTICKS = np.arange(-180, 180.1, 30)
_YTICKS = np.arange(-90,   90.1, 20)

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

def _base_map(figsize=(18, 9), extent=(-180, 180, -90, 90)):
    """Build a Cartopy PlateCarree figure — v6.py line ~240."""
    fig = plt.figure(figsize=figsize)
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, linestyle=":", color="k", alpha=0.5)
    gl.xlocator = mticker.FixedLocator(_XTICKS)
    gl.ylocator = mticker.FixedLocator(_YTICKS)
    ax.set_xticks(_XTICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(_YTICKS, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def _extract_grids(ds_combined: xr.Dataset, ds_post: xr.Dataset):
    """Pull out all the 2D arrays needed by the plot functions."""
    lat = ds_combined["latitude"].values
    lon = ds_combined["longitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    labels_2d      = ds_combined["s5p_pca__plume_labels"].values.astype(int)
    frp_active_mask = ds_post["frp_active_mask"].values.astype(bool)

    # FRP data dict (short names) — mirrors v6.py frp_data dict
    frp_data: dict[str, np.ndarray] = {}
    for short, prefixed in FRP_VAR_MAP.items():
        if prefixed in ds_combined:
            frp_data[short] = ds_combined[prefixed].values
        else:
            frp_data[short] = np.full(labels_2d.shape, np.nan)

    return lat, lon, lat_2d, lon_2d, labels_2d, frp_data, frp_active_mask


# ---------------------------------------------------------------------------
# Map 1 — Plumes + FRP overlay (SWIR or MWIR)
# v6.py lines ~944–984
# ---------------------------------------------------------------------------

def _map_plumes_frp(
    lat, lon, lat_2d, lon_2d, labels_2d,
    frp_data, frp_active_mask, date_str, output_dir, frp_channel="SWIR",
):
    frp_key   = "FRP_SWIR_no_SAA_mean" if frp_channel == "SWIR" else "FRP_MWIR_mean"
    frp_label = "FRP SWIR no SAA"      if frp_channel == "SWIR" else "FRP MWIR"

    fig, ax = _base_map()
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
        fontsize=16)
    ax.legend(loc="lower left", fontsize=14, markerscale=2)
    fig.savefig(Path(output_dir) / f"map_S5P-PCA_FRP_{frp_channel}_{date_str}.png",
                bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Map 2 — Fire score per plume, coloured + source stars
# v6.py lines ~986–1050
# ---------------------------------------------------------------------------

def _map_fire_score_plume(
    lat_2d, lon_2d, labels_2d, df_plumes, date_str, output_dir,
):
    for band, col_var, cmap_name in [
        ("SWIR", "fire_score_SWIR_plume", "jet"),
        ("MWIR", "fire_score_MWIR_plume", "plasma"),
    ]:
        fig, ax = _base_map()
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

            star_col_map = {"high": "red", "low": "gold", "none": None}
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
                    star_c = star_col_map.get(r.get(conf_col, "none"), "gold")
                    if star_c:
                        ax.plot(r[src_lon], r[src_lat], marker="*",
                                color=star_c, markersize=10,
                                markeredgecolor="k", markeredgewidth=0.5,
                                transform=ccrs.PlateCarree(), zorder=15)

            sm = plt.cm.ScalarMappable(cmap=cmap_score, norm=norm_score)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.5,
                         label=f"fire_score_{band}_plume (log)")
            ax.scatter([], [], s=6, color="gray",
                       label=f"No FRP ({len(df_no_fire)})")
            ax.plot([], [], marker="*", color="gold", markersize=6,
                    markeredgecolor="k", linestyle="None", label="Source ?")
            ax.legend(loc="lower left", fontsize=12)

        ax.set_title(f"S5P-PCA plumes — fire_score_{band}_plume — {date_str}", fontsize=14)
        fig.savefig(Path(output_dir) / f"map_fire_score_{band}_plume_{date_str}.png",
                    bbox_inches="tight", dpi=600)
        plt.close()


# ---------------------------------------------------------------------------
# Map 3 — Per-pixel fire score
# v6.py lines ~1053–1075
# ---------------------------------------------------------------------------

def _map_fire_score_pixel(lat, lon, ds_post: xr.Dataset, date_str, output_dir):
    for band in ("SWIR", "MWIR"):
        grid_px = ds_post[f"fire_score_{band}"].values
        fig, ax = _base_map()
        valid_px = grid_px[~np.isnan(grid_px)]
        if len(valid_px) > 0:
            norm_px = LogNorm(vmin=valid_px.min() + 1e-6, vmax=valid_px.max())
            cmap_px = plt.cm.get_cmap("inferno")
            sc = ax.pcolormesh(lon, lat, grid_px,
                               norm=norm_px, cmap=cmap_px,
                               transform=ccrs.PlateCarree(), zorder=3)
            plt.colorbar(sc, ax=ax, shrink=0.5,
                         label=f"fire_score_{band} = score_CO(pixel) × log(1 + FRP_{band}_energy_plume) (log scale)")
        ax.set_title(f"S5P-PCA plumes — fire_score_{band} per pixel — {date_str}", fontsize=14)
        fig.savefig(Path(output_dir) / f"map_fire_score_{band}_pixel_{date_str}.png",
                    bbox_inches="tight", dpi=600)
        plt.close()


# ---------------------------------------------------------------------------
# Map 4 — Zoom with envelope + source confidence
# v6.py lines ~240–310 (map_check_plumes function)
# ---------------------------------------------------------------------------

def _map_check_plumes_zoom(
    lat_2d, lon_2d, labels_2d,
    frp_data, frp_active_mask,
    df_plumes, date_str, zone,
    ll_lon, ll_lat, ur_lon, ur_lat,
    buf_coeff, output_dir,
    frp_channel="SWIR",
):
    """Zoom map with plume envelopes, FRP overlay, and source stars."""
    frp_key   = "FRP_SWIR_no_SAA_mean" if frp_channel == "SWIR" else "FRP_MWIR_mean"
    frp_label = "FRP SWIR no SAA"      if frp_channel == "SWIR" else "FRP MWIR"

    conf_color_map = {"high": "r", "low": "gold", "none": None}

    fig = plt.figure(figsize=(18, 9))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, linestyle=":", color="k", alpha=0.5)
    gl.xlocator = mticker.FixedLocator(_XTICKS)
    gl.ylocator = mticker.FixedLocator(_YTICKS)
    ax.set_xticks(_XTICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(_YTICKS, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent((ll_lon, ur_lon, ll_lat, ur_lat), crs=ccrs.PlateCarree())

    src_col       = "SWIR" if frp_channel == "SWIR" else "MWIR"
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

        in_ext = ((lon_2d >= ll_lon) & (lon_2d <= ur_lon)
                & (lat_2d >= ll_lat) & (lat_2d <= ur_lat))

        if (env_mask & in_ext).any():
            has_envelope = True
            ax.contour(
                lon_2d, lat_2d, (env_mask & in_ext).astype(float),
                levels=[0.5], colors=[col_env], linestyles="dashed",
                linewidths=1.2, transform=ccrs.PlateCarree(), zorder=4,
            )

        if (s5p_mask & in_ext).any():
            ax.scatter(lon_2d[s5p_mask & in_ext],
                       lat_2d[s5p_mask & in_ext],
                       s=2, color=col_plume,
                       transform=ccrs.PlateCarree(), zorder=3)

        conf_label = row.get(conf_col_key, "none")
        star_color = conf_color_map.get(conf_label, "gray")

        if src_loc_count in row and row[src_loc_count] == 1 and star_color:
            slat, slon = row[src_lat_count], row[src_lon_count]
            if (ll_lon <= slon <= ur_lon) and (ll_lat <= slat <= ur_lat):
                has_src_count = True
                ax.plot(slon, slat, marker="*", color=star_color, markersize=12,
                        markeredgecolor="k", markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(), zorder=21)

        if src_loc_sum in row and row[src_loc_sum] == 1 and star_color:
            slat, slon = row[src_lat_sum], row[src_lon_sum]
            if (ll_lon <= slon <= ur_lon) and (ll_lat <= slat <= ur_lat):
                has_src_sum = True
                ax.plot(slon, slat, marker="*", color=star_color, markersize=8,
                        markeredgecolor="k", markeredgewidth=0.5,
                        transform=ccrs.PlateCarree(), zorder=22)

    mask_ns  = np.zeros(lat_2d.shape, dtype=bool)
    has_frp  = False
    in_ext_frp = ((lon_2d >= ll_lon) & (lon_2d <= ur_lon)
                & (lat_2d >= ll_lat) & (lat_2d <= ur_lat))
    if frp_key in frp_data:
        frp_val = frp_data[frp_key]
        mask_ns = frp_active_mask & ~np.isnan(frp_val) & (frp_val > 0)
        if (mask_ns & in_ext_frp).sum() > 0:
            has_frp = True
            ax.scatter(lon_2d[mask_ns & in_ext_frp],
                       lat_2d[mask_ns & in_ext_frp],
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

    in_ext_s5p = ((lon_2d >= ll_lon) & (lon_2d <= ur_lon)
                & (lat_2d >= ll_lat) & (lat_2d <= ur_lat))
    n_frp_vis   = int((mask_ns & in_ext_frp).sum())
    n_plume_vis = int(((labels_2d > 0) & in_ext_s5p).sum())

    ax.set_title(
        f"Combination Plumes S5P-PCA ({n_plume_vis} cells) × {frp_label} "
        f"({n_frp_vis} cells) — {date_str}",
        fontsize=18)
    ax.legend(handles=legend_handles, loc="lower right", fontsize=16, markerscale=2)
    fig.savefig(
        Path(output_dir) / f"map_S5P-PCA_FRP_superposition_{date_str}_{frp_channel}"
                           f"_zoom_{zone}_envelope_{buf_coeff}.png",
        bbox_inches="tight", dpi=600,
    )
    plt.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot(
    ds_combined: xr.Dataset,
    ds_post: xr.Dataset,
    df_plumes: pd.DataFrame,
    output_dir: str | Path,
    date_str: str,
    zones: list[tuple[str, float, float, float, float]] | None = None,
    buf_coeff: float = 0.2,
) -> None:
    """
    Generate all map types from v6.py and save them to output_dir.

    Parameters
    ----------
    ds_combined :
        Raw combined input dataset (used for FRP values and plume labels).
    ds_post :
        Output of ``postprocess.compute()`` (fire score grids, confidence grids …).
    df_plumes :
        Per-plume summary DataFrame, second return value of ``compute()``.
    output_dir :
        Directory where PNG files are written.  Created if absent.
    date_str :
        Date string used in file/title names, e.g. ``"20210817"``.
    zones :
        List of (name, ll_lon, ur_lon, ll_lat, ur_lat) for zoom maps.
        Defaults to ``DEFAULT_ZONES``.
    buf_coeff :
        Buffer coefficient passed to the zoom envelope re-draw.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if zones is None:
        zones = DEFAULT_ZONES

    lat, lon, lat_2d, lon_2d, labels_2d, frp_data, frp_active_mask = _extract_grids(
        ds_combined, ds_post)

    # Map 1 — plumes + FRP overlay — v6.py lines ~944–984 (SWIR only, as in v6)
    _map_plumes_frp(lat, lon, lat_2d, lon_2d, labels_2d,
                    frp_data, frp_active_mask, date_str, output_dir,
                    frp_channel="SWIR")

    # Map 2 — fire score per plume — v6.py lines ~986–1050
    _map_fire_score_plume(lat_2d, lon_2d, labels_2d, df_plumes, date_str, output_dir)

    # Map 3 — per-pixel fire score — v6.py lines ~1053–1075
    _map_fire_score_pixel(lat, lon, ds_post, date_str, output_dir)

    # Map 4 — regional zoom with envelopes — v6.py lines ~1080–1230
    for zone_name, ll_lon, ur_lon, ll_lat, ur_lat in zones:
        for channel in ("SWIR", "MWIR"):
            _map_check_plumes_zoom(
                lat_2d, lon_2d, labels_2d,
                frp_data, frp_active_mask,
                df_plumes, date_str, zone_name,
                ll_lon, ll_lat, ur_lon, ur_lat,
                buf_coeff, output_dir,
                frp_channel=channel,
            )
