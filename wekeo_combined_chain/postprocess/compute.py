"""
Core postprocess computation — two-stage API.

Adapted from combine_S5P-PCA_FRP_v6.py, reading from the combined NetCDF
(all variables prefixed, single input file) instead of three separate files.

Usage
-----
    import xarray as xr
    from wekeo_combined_chain import postprocess

    ds = xr.open_dataset("/mnt/ceph/.../s5p_pca_plumes_frp_iasi_product.nc")

    # Two-stage (inspect / iterate between stages)
    df_plumes = postprocess.compute_plume_stats(ds)
    df_plumes.to_csv("plumes_summary.csv", index=False)
    ds_post = postprocess.build_grids(ds, df_plumes)
    ds_post.to_netcdf("postprocess_output.nc")

    # One-shot convenience
    ds_post, df_plumes = postprocess.compute(ds)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ._params import (
    BUF_COEFF, BUF_MIN, BUF_MAX, MAX_EXTRA_ITER,
    SOURCE_N_MIN, MAX_CLUSTERS_SAVED, CONF_MAX_DIST_AGREE_KM,
    FRP_VAR_MAP, S5P_SCORE_VAR, S5P_LABELS_VAR, S5P_SAMPLES_VAR,
)
from ._utils import (
    fire_score, haversine_km, dynamic_buffer, adaptive_buffer,
    estimate_source_with_clusters, compute_source_confidence,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Output variable metadata — adapted from v6.py lines ~820–870
# ---------------------------------------------------------------------------
_PLUME_VAR_ATTRS: dict[str, tuple[str, str]] = {
    "mean_score_CO_plume"            : ("Mean S5P-PCA CO score over plume",                     "1"),
    "std_score_CO_plume"             : ("Std S5P-PCA CO score over plume",                      "1"),
    "frp_energy_SWIR_plume"          : ("Total SWIR FRP energy in search area (sum of means)",  "MW"),
    "frp_energy_MWIR_plume"          : ("Total MWIR FRP energy in search area (sum of means)",  "MW"),
    "n_frp_cells_total_plume"        : ("Number of active FRP cells in plume + envelope",       "1"),
    "fire_score_SWIR_plume"          : ("Plume fire score SWIR = score_CO_avg * log(1+E_SWIR)", "1"),
    "fire_score_MWIR_plume"          : ("Plume fire score MWIR = score_CO_avg * log(1+E_MWIR)", "1"),
    "fire_type_ratio_SWIR_MWIR_plume": ("FRP SWIR/MWIR ratio (high = intense fire)",           "1"),
    # SWIR count
    "source_lat_SWIR_count_plume"          : ("Source lat SWIR (dominant cluster by n_cells)", "degrees"),
    "source_lon_SWIR_count_plume"          : ("Source lon SWIR (dominant cluster by n_cells)", "degrees"),
    "source_is_localized_SWIR_count_plume" : ("1 if SWIR count source is well-localised",      "1"),
    "source_dist_km_SWIR_count_plume"      : ("Distance source SWIR count to plume centroid",  "km"),
    # SWIR sum
    "source_lat_SWIR_sum_plume"            : ("Source lat SWIR (dominant cluster by sum_FRP)", "degrees"),
    "source_lon_SWIR_sum_plume"            : ("Source lon SWIR (dominant cluster by sum_FRP)", "degrees"),
    "source_is_localized_SWIR_sum_plume"   : ("1 if SWIR sum source is well-localised",        "1"),
    "source_dist_km_SWIR_sum_plume"        : ("Distance source SWIR sum to plume centroid",    "km"),
    # MWIR count
    "source_lat_MWIR_count_plume"          : ("Source lat MWIR (dominant cluster by n_cells)", "degrees"),
    "source_lon_MWIR_count_plume"          : ("Source lon MWIR (dominant cluster by n_cells)", "degrees"),
    "source_is_localized_MWIR_count_plume" : ("1 if MWIR count source is well-localised",      "1"),
    "source_dist_km_MWIR_count_plume"      : ("Distance source MWIR count to plume centroid",  "km"),
    # MWIR sum
    "source_lat_MWIR_sum_plume"            : ("Source lat MWIR (dominant cluster by sum_FRP)", "degrees"),
    "source_lon_MWIR_sum_plume"            : ("Source lon MWIR (dominant cluster by sum_FRP)", "degrees"),
    "source_is_localized_MWIR_sum_plume"   : ("1 if MWIR sum source is well-localised",        "1"),
    "source_dist_km_MWIR_sum_plume"        : ("Distance source MWIR sum to plume centroid",    "km"),
    # Confidence
    "source_confidence_score_SWIR_plume"   : ("Source confidence score SWIR (0=none,1=low,2=high)", "1"),
    "source_confidence_score_MWIR_plume"   : ("Source confidence score MWIR (0=none,1=low,2=high)", "1"),
}

PLUME_VARS = list(_PLUME_VAR_ATTRS.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_arrays(ds: xr.Dataset) -> dict:
    """Extract all numpy arrays needed by both stages from the combined dataset."""
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    labels_2d = ds[S5P_LABELS_VAR].values.astype(int)
    score_co_2d = ds[S5P_SCORE_VAR].values
    nb_samples = ds[S5P_SAMPLES_VAR].values
    s5p_valid_mask = nb_samples > 0

    frp_data: dict[str, np.ndarray] = {}
    for short, prefixed in FRP_VAR_MAP.items():
        if prefixed in ds:
            frp_data[short] = ds[prefixed].values
        else:
            frp_data[short] = np.full(labels_2d.shape, np.nan)

    swir_ok = (
        (frp_data["FRP_SWIR_no_SAA_count"] > 0)
        & (frp_data["FRP_SWIR_no_SAA_mean"] > 0)
        & (~np.isnan(frp_data["FRP_SWIR_no_SAA_mean"]))
    )
    mwir_ok = (
        (frp_data["FRP_MWIR_count"] > 0)
        & (frp_data["FRP_MWIR_mean"] > 0)
        & (~np.isnan(frp_data["FRP_MWIR_mean"]))
    )
    frp_active_mask = swir_ok | mwir_ok

    num_plumes = np.unique(labels_2d)
    num_plumes = num_plumes[num_plumes != 0]

    return {
        "lat": lat, "lon": lon,
        "lat_2d": lat_2d, "lon_2d": lon_2d,
        "labels_2d": labels_2d,
        "num_plumes": num_plumes,
        "score_co_2d": score_co_2d,
        "s5p_valid_mask": s5p_valid_mask,
        "frp_data": frp_data,
        "frp_active_mask": frp_active_mask,
    }


# ---------------------------------------------------------------------------
# Public API — Stage 1
# ---------------------------------------------------------------------------

def compute_plume_stats(
    ds: xr.Dataset,
    buf_coeff: float = BUF_COEFF,
    buf_min: int = BUF_MIN,
    buf_max: int = BUF_MAX,
    max_extra_iter: int = MAX_EXTRA_ITER,
    source_n_min: int = SOURCE_N_MIN,
    max_clusters_saved: int = MAX_CLUSTERS_SAVED,
    conf_max_dist_km: float = CONF_MAX_DIST_AGREE_KM,
) -> pd.DataFrame:
    """
    Stage 1 — compute per-plume statistics.

    For each plume: buffers, FRP aggregation, fire scores, source localisation,
    confidence scoring, and top-N cluster info.

    Parameters
    ----------
    ds : xr.Dataset
        Combined dataset produced by ``wekeo_combined_chain.combined.get_combined_product``.
        Must contain the variables listed in ``_params.FRP_VAR_MAP`` and the S5P vars.
    buf_coeff, buf_min, buf_max, max_extra_iter :
        Dynamic / adaptive buffer parameters.
    source_n_min : int
        Minimum FRP pixels required to estimate a source position.
    max_clusters_saved : int
        Top-N FRP clusters stored in the output DataFrame.
    conf_max_dist_km : float
        Threshold (km) for source location agreement (confidence scoring).

    Returns
    -------
    pd.DataFrame
        One row per plume.  All numeric columns are float32-compatible; string
        columns hold confidence labels ("none" / "low" / "high").
    """

    arrs = _load_arrays(ds)
    lat_2d          = arrs["lat_2d"]
    lon_2d          = arrs["lon_2d"]
    labels_2d       = arrs["labels_2d"]
    num_plumes      = arrs["num_plumes"]
    score_co_2d     = arrs["score_co_2d"]
    s5p_valid_mask  = arrs["s5p_valid_mask"]
    frp_data        = arrs["frp_data"]
    frp_active_mask = arrs["frp_active_mask"]

    swir_grid = frp_data.get("FRP_SWIR_no_SAA_mean", np.full(labels_2d.shape, np.nan))
    mwir_grid = frp_data.get("FRP_MWIR_mean",         np.full(labels_2d.shape, np.nan))

# MAJ 11/06/2026 SP
    print("==========================================")
    print(" ** S5P-PCA/FRP combined analysis **")
    print(f" - Active FRP cells (filtered): {frp_active_mask.sum()}")
    print(f" - Total number of plumes to process: {len(num_plumes)}")
    print("==========================================")

    # print(f"--> Active FRP cells (filtered): {frp_active_mask.sum()}")
    # print(f"--> {len(num_plumes)} plumes")

    rows = []
    n_confirmed = 0

    for lbl in num_plumes:
        mask_plume = (labels_2d == lbl)
        n_pix = int(mask_plume.sum())

        # CO score over the plume — v6.py line ~445
        sc_CO = score_co_2d[mask_plume]
        sc_CO = sc_CO[~np.isnan(sc_CO)]
        if len(sc_CO) > 0:
            mean_sc_co = float(np.mean(sc_CO))
            std_sc_co  = float(np.std(sc_CO))
        else:
            mean_sc_co = std_sc_co = np.nan

        # Plume centroid — v6.py line ~455
        centroid_lat = float(np.mean(lat_2d[mask_plume]))
        centroid_lon = float(np.mean(lon_2d[mask_plume]))

        # Dynamic + adaptive buffer — v6.py lines ~460–475
        buf_init = dynamic_buffer(n_pix, buf_coeff, buf_min, buf_max)
        buf, s5p_mask_ext, env_mask, _ = adaptive_buffer(
            mask_plume, labels_2d, frp_active_mask,
            buf_init, buf_max, max_extra_iter,
        )

        # MAJ SP 11/06/2026
        # print(f"--------------------------")
        # print(f"--> Plume {lbl}: buf_init={buf_init}  buf_final={buf}")

        frp_mask_strict = mask_plume & frp_active_mask
        frp_mask_env    = env_mask   & frp_active_mask
        frp_mask_search = frp_mask_strict | frp_mask_env

        n_frp_strict = int(frp_mask_strict.sum())
        n_frp_env    = int(frp_mask_env.sum())
        n_frp_cells  = int(frp_mask_search.sum())

        # MAJ SP 11/06/2026

        # if n_frp_cells > 0:
        #     print(f"** Plume {lbl} confirmed ** (strict={n_frp_strict}, env={n_frp_env})")
        # else:
        #     print(f"** Plume {lbl} alone **")


        print("------------------------------------------")
        print(f"--> Plume #{lbl}")
        print("------------------------------------------")
        print(f" Size         : {n_pix} cells")
        print(f" Search buffer: initial={buf_init}px -> final={buf}px")
        print( "                (iteratively expanded to search for FRP)")
        print()
        if n_frp_cells > 0:
            print(f" FRP data     : [FOUND] {n_frp_strict}px within plume, {n_frp_env}px in envelope")
        else:
            print( " FRP data     : [NONE ] no FRP found within plume or envelope")

        # Aggregate FRP stats over search zone — v6.py lines ~485–500
 
        frp_stats: dict[str, float] = {}
        for var in frp_data:
            vals = frp_data[var][frp_mask_search]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                frp_stats[f"{var}_avg_plume"] = np.nan
                frp_stats[f"{var}_max_plume"] = np.nan
                frp_stats[f"{var}_sum_plume"] = np.nan
            else:
                frp_stats[f"{var}_avg_plume"] = float(np.nanmean(vals))
                frp_stats[f"{var}_max_plume"] = float(np.nanmax(vals))
                frp_stats[f"{var}_sum_plume"] = float(np.nansum(vals))

        frp_energy_SWIR = frp_stats.get("FRP_SWIR_no_SAA_mean_sum_plume", np.nan)
        frp_energy_MWIR = frp_stats.get("FRP_MWIR_mean_sum_plume", np.nan)

        # SWIR/MWIR ratio — v6.py line ~507
        mwir_avg = frp_stats.get("FRP_MWIR_mean_avg_plume", np.nan)
        swir_avg = frp_stats.get("FRP_SWIR_no_SAA_mean_avg_plume", np.nan)
        if (not np.isnan(mwir_avg)) and mwir_avg > 0:
            fire_type_ratio = float(swir_avg / mwir_avg)
        else:
            fire_type_ratio = np.nan

        fire_score_SWIR = fire_score(frp_energy_SWIR, mean_sc_co)
        fire_score_MWIR = fire_score(frp_energy_MWIR, mean_sc_co)

        # Source estimation — v6.py lines ~515–540
        (src_SWIR_count, src_SWIR_sum,
         n_clust_SWIR, clusters_SWIR, _) = estimate_source_with_clusters(
            mask_plume, s5p_valid_mask, frp_active_mask, swir_grid,
            lat_2d, lon_2d, source_n_min,
        )
        (src_MWIR_count, src_MWIR_sum,
         n_clust_MWIR, clusters_MWIR, _) = estimate_source_with_clusters(
            mask_plume, s5p_valid_mask, frp_active_mask, mwir_grid,
            lat_2d, lon_2d, source_n_min,
        )

        # Source distances — v6.py line ~560
        def _src_dist(src: dict) -> float:
            if np.isnan(src["source_lat"]):
                return np.nan
            return haversine_km(src["source_lat"], src["source_lon"],
                                centroid_lat, centroid_lon)

        # Confidence scores — v6.py lines ~563–573
        conf_score_SWIR, conf_label_SWIR, _ = compute_source_confidence(
            src_SWIR_count, src_SWIR_sum, conf_max_dist_km)
        conf_score_MWIR, conf_label_MWIR, _ = compute_source_confidence(
            src_MWIR_count, src_MWIR_sum, conf_max_dist_km)

        # MAJ SP 11/06/2026
        # print(f"   SWIR conf: {conf_label_SWIR} ({conf_score_SWIR}/2) | "
        #       f"MWIR conf: {conf_label_MWIR} ({conf_score_MWIR}/2)")
        
        swir_localized = not np.isnan(src_SWIR_sum["source_lat"])
        mwir_localized = not np.isnan(src_MWIR_sum["source_lat"])
        print()
        swir_tag = "[OK]" if swir_localized else "[--]"
        mwir_tag = "[OK]" if mwir_localized else "[--]"
        print(f" SWIR source  : {swir_tag} {'localized' if swir_localized else 'not localized':<15}  (confidence: {conf_score_SWIR}/2 - {conf_label_SWIR})")
        print(f" MWIR source  : {mwir_tag} {'localized' if mwir_localized else 'not localized':<15}  (confidence: {conf_score_MWIR}/2 - {conf_label_MWIR})")
        print("------------------------------------------")

        # Top-N cluster columns — v6.py lines ~576–590
        cluster_cols: dict[str, object] = {}
        for band, cl_list in (("SWIR", clusters_SWIR), ("MWIR", clusters_MWIR)):
            for k in range(max_clusters_saved):
                pfx = f"cluster_{k+1}_{band}"
                if k < len(cl_list):
                    c = cl_list[k]
                    cluster_cols[f"{pfx}_n_cells"]      = c["n_cells"]
                    cluster_cols[f"{pfx}_sum_frp"]      = round(c["sum_frp"], 4)
                    cluster_cols[f"{pfx}_centroid_lat"] = round(c["centroid_lat"], 4)
                    cluster_cols[f"{pfx}_centroid_lon"] = round(c["centroid_lon"], 4)
                else:
                    cluster_cols[f"{pfx}_n_cells"]      = np.nan
                    cluster_cols[f"{pfx}_sum_frp"]      = np.nan
                    cluster_cols[f"{pfx}_centroid_lat"] = np.nan
                    cluster_cols[f"{pfx}_centroid_lon"] = np.nan

        # Assemble row — v6.py lines ~593–640
        row: dict[str, object] = {
            "label"                          : lbl,
            "n_pixels_plume"                 : n_pix,
            "buffer_pixels_init_plume"       : buf_init,
            "buffer_pixels_plume"            : buf,
            "centroid_lat_plume"             : centroid_lat,
            "centroid_lon_plume"             : centroid_lon,
            "mean_score_CO_plume"            : mean_sc_co,
            "std_score_CO_plume"             : std_sc_co,
            "n_frp_cells_strict_plume"       : n_frp_strict,
            "n_frp_cells_contour_plume"      : n_frp_env,
            "n_frp_cells_total_plume"        : n_frp_cells,
            "frp_energy_SWIR_plume"          : frp_energy_SWIR,
            "frp_energy_MWIR_plume"          : frp_energy_MWIR,
            "fire_score_SWIR_plume"          : fire_score_SWIR,
            "fire_score_MWIR_plume"          : fire_score_MWIR,
            "fire_type_ratio_SWIR_MWIR_plume": fire_type_ratio,
            "n_frp_clusters_SWIR_strict"     : n_clust_SWIR,
            "n_frp_clusters_MWIR_strict"     : n_clust_MWIR,
            # Source SWIR count
            "source_lat_SWIR_count_plume"          : src_SWIR_count["source_lat"],
            "source_lon_SWIR_count_plume"          : src_SWIR_count["source_lon"],
            "source_is_localized_SWIR_count_plume" : src_SWIR_count["source_is_localized"],
            "source_dist_km_SWIR_count_plume"      : _src_dist(src_SWIR_count),
            "source_method_SWIR_count_plume"       : src_SWIR_count["dominant_cluster_method"],
            # Source SWIR sum
            "source_lat_SWIR_sum_plume"            : src_SWIR_sum["source_lat"],
            "source_lon_SWIR_sum_plume"            : src_SWIR_sum["source_lon"],
            "source_is_localized_SWIR_sum_plume"   : src_SWIR_sum["source_is_localized"],
            "source_dist_km_SWIR_sum_plume"        : _src_dist(src_SWIR_sum),
            "source_method_SWIR_sum_plume"         : src_SWIR_sum["dominant_cluster_method"],
            # Source MWIR count
            "source_lat_MWIR_count_plume"          : src_MWIR_count["source_lat"],
            "source_lon_MWIR_count_plume"          : src_MWIR_count["source_lon"],
            "source_is_localized_MWIR_count_plume" : src_MWIR_count["source_is_localized"],
            "source_dist_km_MWIR_count_plume"      : _src_dist(src_MWIR_count),
            "source_method_MWIR_count_plume"       : src_MWIR_count["dominant_cluster_method"],
            # Source MWIR sum
            "source_lat_MWIR_sum_plume"            : src_MWIR_sum["source_lat"],
            "source_lon_MWIR_sum_plume"            : src_MWIR_sum["source_lon"],
            "source_is_localized_MWIR_sum_plume"   : src_MWIR_sum["source_is_localized"],
            "source_dist_km_MWIR_sum_plume"        : _src_dist(src_MWIR_sum),
            "source_method_MWIR_sum_plume"         : src_MWIR_sum["dominant_cluster_method"],
            # Confidence
            "source_confidence_score_SWIR_plume"   : conf_score_SWIR,
            "source_confidence_label_SWIR_plume"   : conf_label_SWIR,
            "source_confidence_score_MWIR_plume"   : conf_score_MWIR,
            "source_confidence_label_MWIR_plume"   : conf_label_MWIR,
        }
        row.update(cluster_cols)
        rows.append(row)
        if n_frp_cells > 0:
            n_confirmed += 1
        # print("--------------------------")

    df_plumes = pd.DataFrame(rows)
    # MAJ SP 11/06/2026
    # print(f"Plumes with FRP: {n_confirmed}")
    percent = round(n_confirmed / len(num_plumes) * 100) if len(num_plumes) > 0 else 0
    print("==========================================")
    print(" Complete Analysis ")
    print(f" Total plumes       : {len(num_plumes)}")
    print(f" Plumes with FRP    : {n_confirmed}  ({percent}%)")
    print(f" Plumes without FRP : {len(num_plumes) - n_confirmed}")
    print("==========================================")
    
    return df_plumes


# ---------------------------------------------------------------------------
# Public API — Stage 2
# ---------------------------------------------------------------------------

def build_grids(
    ds: xr.Dataset,
    df_plumes: pd.DataFrame,
    source_n_min: int = SOURCE_N_MIN,
) -> xr.Dataset:
    """
    Stage 2 — spread per-plume stats onto 2-D grids and build the output dataset.

    Runs ``estimate_source_with_clusters`` again per plume to reconstruct the
    cluster-ID grids (deterministic given the same inputs as Stage 1).

    Parameters
    ----------
    ds : xr.Dataset
        Same combined dataset passed to ``compute_plume_stats``.
    df_plumes : pd.DataFrame
        Output of ``compute_plume_stats``.
    source_n_min : int
        Same value used in Stage 1 (for deterministic cluster-ID grids).

    Returns
    -------
    xr.Dataset
        Per-pixel grids: fire scores, plume-level stats, cluster IDs,
        confidence labels, active FRP mask, and plume label grid.
    """
    arrs = _load_arrays(ds)
    lat             = arrs["lat"]
    lon             = arrs["lon"]
    lat_2d          = arrs["lat_2d"]
    lon_2d          = arrs["lon_2d"]
    labels_2d       = arrs["labels_2d"]
    score_co_2d     = arrs["score_co_2d"]
    s5p_valid_mask  = arrs["s5p_valid_mask"]
    frp_data        = arrs["frp_data"]
    frp_active_mask = arrs["frp_active_mask"]

    swir_grid = frp_data.get("FRP_SWIR_no_SAA_mean", np.full(labels_2d.shape, np.nan))
    mwir_grid = frp_data.get("FRP_MWIR_mean",         np.full(labels_2d.shape, np.nan))

    height, width = labels_2d.shape

    grids_plume = {
        var: np.full((height, width), np.nan, dtype=np.float32)
        for var in PLUME_VARS
    }
    grid_fire_score_SWIR = np.full((height, width), np.nan, dtype=np.float32)
    grid_fire_score_MWIR = np.full((height, width), np.nan, dtype=np.float32)
    grid_cluster_id_SWIR = np.zeros((height, width), dtype=np.int16)
    grid_cluster_id_MWIR = np.zeros((height, width), dtype=np.int16)

    CONF_LABEL_INT = {"none": 0, "low": 1, "high": 2}
    grid_conf_label_SWIR = np.zeros((height, width), dtype=np.int8)
    grid_conf_label_MWIR = np.zeros((height, width), dtype=np.int8)

    label_to_row = df_plumes.set_index("label")

    for lbl in df_plumes["label"].values:
        mask = (labels_2d == lbl)
        if not mask.any():
            continue
        row = label_to_row.loc[lbl]

        # Fill constant-per-plume grid variables from df
        for var in PLUME_VARS:
            if var in row.index and not pd.isna(row[var]):
                grids_plume[var][mask] = np.float32(row[var])

        # Per-pixel fire scores (score_CO varies per pixel)
        score_co_pixels = score_co_2d[mask]

        e_swir = row.get("frp_energy_SWIR_plume", np.nan)
        if not pd.isna(e_swir) and e_swir > 0:
            grid_fire_score_SWIR[mask] = (score_co_pixels * np.log1p(e_swir)).astype(np.float32)

        e_mwir = row.get("frp_energy_MWIR_plume", np.nan)
        if not pd.isna(e_mwir) and e_mwir > 0:
            grid_fire_score_MWIR[mask] = (score_co_pixels * np.log1p(e_mwir)).astype(np.float32)

        grid_conf_label_SWIR[mask] = CONF_LABEL_INT.get(
            row.get("source_confidence_label_SWIR_plume", "none"), 0)
        grid_conf_label_MWIR[mask] = CONF_LABEL_INT.get(
            row.get("source_confidence_label_MWIR_plume", "none"), 0)

        # Re-derive cluster-ID grids (deterministic — same inputs as Stage 1)
        mask_plume = (labels_2d == lbl)
        (_, _, _, _, cid_SWIR) = estimate_source_with_clusters(
            mask_plume, s5p_valid_mask, frp_active_mask, swir_grid,
            lat_2d, lon_2d, source_n_min,
        )
        (_, _, _, _, cid_MWIR) = estimate_source_with_clusters(
            mask_plume, s5p_valid_mask, frp_active_mask, mwir_grid,
            lat_2d, lon_2d, source_n_min,
        )
        grid_cluster_id_SWIR[cid_SWIR > 0] = cid_SWIR[cid_SWIR > 0]
        grid_cluster_id_MWIR[cid_MWIR > 0] = cid_MWIR[cid_MWIR > 0]

    grid_label = np.where(labels_2d > 0, labels_2d.astype(np.float32), np.nan)

    # Build xr.Dataset
    dims = ("latitude", "longitude")
    coords = {
        "latitude" : xr.DataArray(lat, dims=("latitude",),
                                  attrs={"units": "degrees_north", "long_name": "Latitude"}),
        "longitude": xr.DataArray(lon, dims=("longitude",),
                                  attrs={"units": "degrees_east", "long_name": "Longitude"}),
    }

    data_vars: dict[str, xr.DataArray] = {}

    for var in PLUME_VARS:
        long_name, units = _PLUME_VAR_ATTRS[var]
        data_vars[var] = xr.DataArray(
            grids_plume[var], dims=dims,
            attrs={"long_name": long_name, "units": units,
                   "note": "Constant value over all cells of the plume"},
        )

    data_vars["fire_score_SWIR"] = xr.DataArray(
        grid_fire_score_SWIR, dims=dims,
        attrs={"long_name": "Per-pixel fire score SWIR: score_CO(pixel) * log(1+frp_energy_SWIR_plume)",
               "units": "1",
               "note": "score_CO varies per pixel; frp_energy_SWIR is constant over the plume"},
    )
    data_vars["fire_score_MWIR"] = xr.DataArray(
        grid_fire_score_MWIR, dims=dims,
        attrs={"long_name": "Per-pixel fire score MWIR: score_CO(pixel) * log(1+frp_energy_MWIR_plume)",
               "units": "1",
               "note": "score_CO varies per pixel; frp_energy_MWIR is constant over the plume"},
    )

    data_vars["cluster_id_SWIR"] = xr.DataArray(
        grid_cluster_id_SWIR, dims=dims,
        attrs={"long_name": "FRP cluster id within plume (SWIR)", "units": "1",
               "note": "0 = no cluster; clusters numbered per plume"},
    )
    data_vars["cluster_id_MWIR"] = xr.DataArray(
        grid_cluster_id_MWIR, dims=dims,
        attrs={"long_name": "FRP cluster id within plume (MWIR)", "units": "1",
               "note": "0 = no cluster; clusters numbered per plume"},
    )

    data_vars["source_confidence_label_SWIR"] = xr.DataArray(
        grid_conf_label_SWIR, dims=dims,
        attrs={"long_name": "Source confidence label SWIR (0=none,1=low,2=high)", "units": "1"},
    )
    data_vars["source_confidence_label_MWIR"] = xr.DataArray(
        grid_conf_label_MWIR, dims=dims,
        attrs={"long_name": "Source confidence label MWIR (0=none,1=low,2=high)", "units": "1"},
    )

    data_vars["frp_active_mask"] = xr.DataArray(
        frp_active_mask.astype(np.int8), dims=dims,
        attrs={"long_name": "Active FRP mask (SWIR or MWIR valid)", "units": "1"},
    )

    data_vars["plume_label"] = xr.DataArray(
        grid_label, dims=dims,
        attrs={"long_name": "Plume label (NaN = background)"},
    )

    return xr.Dataset(data_vars, coords=coords)


# ---------------------------------------------------------------------------
# Public API — convenience wrapper
# ---------------------------------------------------------------------------

def compute(
    ds: xr.Dataset,
    buf_coeff: float = BUF_COEFF,
    buf_min: int = BUF_MIN,
    buf_max: int = BUF_MAX,
    max_extra_iter: int = MAX_EXTRA_ITER,
    source_n_min: int = SOURCE_N_MIN,
    max_clusters_saved: int = MAX_CLUSTERS_SAVED,
    conf_max_dist_km: float = CONF_MAX_DIST_AGREE_KM,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """
    One-shot convenience wrapper: run both stages and return ``(ds_post, df_plumes)``.

    Equivalent to::

        df_plumes = compute_plume_stats(ds, ...)
        ds_post   = build_grids(ds, df_plumes, source_n_min=source_n_min)
        return ds_post, df_plumes

    Returns
    -------
    ds_post : xr.Dataset
    df_plumes : pd.DataFrame
    """
    df_plumes = compute_plume_stats(
        ds,
        buf_coeff=buf_coeff,
        buf_min=buf_min,
        buf_max=buf_max,
        max_extra_iter=max_extra_iter,
        source_n_min=source_n_min,
        max_clusters_saved=max_clusters_saved,
        conf_max_dist_km=conf_max_dist_km,
    )
    ds_post = build_grids(ds, df_plumes, source_n_min=source_n_min)
    return ds_post, df_plumes

