"""
Helper / math functions for the postprocess pipeline.
Ported verbatim from combine_S5P-PCA_FRP_v6.py lines 36–228,
with no algorithmic changes.
"""

import numpy as np
from scipy.ndimage import binary_dilation, label as ndlabel


# ---------------------------------------------------------------------------
# v6.py line 36
# ---------------------------------------------------------------------------
def fire_score(energy: float, mean_score: float) -> float:
    if (not np.isnan(energy)) and (energy > 0) and (not np.isnan(mean_score)):
        return float(mean_score * np.log1p(energy))
    return np.nan


# ---------------------------------------------------------------------------
# v6.py line 44
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return float(2 * R * np.arcsin(np.sqrt(a)))


# ---------------------------------------------------------------------------
# v6.py line 50
# ---------------------------------------------------------------------------
def dynamic_buffer(n_pixels: int, coeff: float, min_buf: int, max_buf: int) -> int:
    """Return integer dilation iterations proportional to sqrt(n_pixels)."""
    raw = coeff * np.sqrt(n_pixels)
    return int(np.clip(raw, min_buf, max_buf))


# ---------------------------------------------------------------------------
# v6.py line 62
# ---------------------------------------------------------------------------
def adaptive_buffer(s5p_mask, labels_2d, frp_active_mask,
                    buf_init: int, buf_max: int, max_extra_iter: int):
    """
    Extend the plume search envelope if doing so captures more FRP cells.

    Returns
    -------
    buf_final, s5p_mask_ext, env_mask, n_frp_final
    """
    s5p_mask_ext = binary_dilation(s5p_mask, iterations=buf_init)
    s5p_mask_ext &= (labels_2d == 0) | s5p_mask

    frp_search  = s5p_mask_ext & frp_active_mask
    n_frp_init  = int(frp_search.sum())

    best_n_frp = n_frp_init
    best_mask  = s5p_mask_ext.copy()
    best_buf   = buf_init

    cur_mask = s5p_mask_ext.copy()
    cur_buf  = buf_init

    for _ in range(max_extra_iter):
        if cur_buf >= buf_max:
            break
        new_mask = binary_dilation(cur_mask, iterations=1)
        new_mask &= (labels_2d == 0) | s5p_mask

        n_frp_new = int((new_mask & frp_active_mask).sum())
        if n_frp_new > best_n_frp:
            best_n_frp = n_frp_new
            best_mask  = new_mask.copy()
            best_buf   = cur_buf + 1

        cur_mask = new_mask
        cur_buf += 1

    if best_n_frp == 0:
        s5p_mask_ext = s5p_mask
        buf_final    = buf_init
        n_frp_final  = n_frp_init
    else:
        s5p_mask_ext = best_mask
        buf_final    = best_buf
        n_frp_final  = best_n_frp

    env_mask = s5p_mask_ext & (~s5p_mask)
    return buf_final, s5p_mask_ext, env_mask, n_frp_final


# ---------------------------------------------------------------------------
# v6.py line 115
# ---------------------------------------------------------------------------
def _centroid(lats_v, lons_v):
    return float(np.mean(lats_v)), float(np.mean(lons_v))


# ---------------------------------------------------------------------------
# v6.py line 124
# ---------------------------------------------------------------------------
def estimate_source_with_clusters(s5p_strict_mask, s5p_valid_mask,
                                   frp_active_mask, frp_val_grid,
                                   lat_2d, lon_2d,
                                   n_min: int):
    """
    Estimate the probable fire source of a plume from FRP cells strictly
    inside the plume and on valid S5P pixels.

    Returns
    -------
    src_by_count, src_by_sum, n_clusters, cluster_info_export, cluster_id_grid
    """
    empty = {
        "source_lat"             : np.nan,
        "source_lon"             : np.nan,
        "source_is_localized"    : 0,
        "dominant_cluster_method": "none",
        "n_frp_strict_used"      : 0,
    }
    cluster_id_grid = np.zeros(frp_val_grid.shape, dtype=np.int16)

    frp_mask_strict = s5p_strict_mask & s5p_valid_mask & frp_active_mask
    if frp_mask_strict.sum() < n_min:
        return dict(empty), dict(empty), 0, [], cluster_id_grid

    struct = np.ones((3, 3), dtype=bool)
    labeled, n_clusters = ndlabel(frp_mask_strict, structure=struct)
    if n_clusters == 0:
        return dict(empty), dict(empty), 0, [], cluster_id_grid

    cluster_info = []
    for cid in range(1, n_clusters + 1):
        cmask  = (labeled == cid)
        vals   = frp_val_grid[cmask]
        vals_v = vals[np.isfinite(vals) & (vals > 0)]

        lats_c = lat_2d[cmask]
        lons_c = lon_2d[cmask]
        w_c    = np.array(frp_val_grid[cmask], dtype=float)
        valid_w = np.isfinite(w_c) & (w_c > 0)

        if valid_w.sum() >= 1:
            clat_c = float(np.sum(w_c[valid_w] * lats_c[valid_w]) / w_c[valid_w].sum())
            clon_c = float(np.sum(w_c[valid_w] * lons_c[valid_w]) / w_c[valid_w].sum())
        else:
            clat_c, clon_c = float(np.mean(lats_c)), float(np.mean(lons_c))

        cluster_info.append({
            "id"          : cid,
            "mask"        : cmask,
            "n_cells"     : int(cmask.sum()),
            "sum_frp"     : float(vals_v.sum()) if len(vals_v) > 0 else 0.0,
            "centroid_lat": clat_c,
            "centroid_lon": clon_c,
        })
        cluster_id_grid[cmask] = np.int16(cid)

    best_by_count = max(cluster_info, key=lambda x: x["n_cells"])
    best_by_sum   = max(cluster_info, key=lambda x: x["sum_frp"])
    method        = "single" if n_clusters == 1 else "dominant"

    def _make_result(best):
        cmask  = best["mask"]
        lats   = lat_2d[cmask]
        lons   = lon_2d[cmask]
        w      = np.array(frp_val_grid[cmask], dtype=float)
        valid  = np.isfinite(w) & (w > 0)
        if int(valid.sum()) < n_min:
            return dict(empty)
        clat, clon = _centroid(lats[valid], lons[valid])
        return {
            "source_lat"             : clat,
            "source_lon"             : clon,
            "source_is_localized"    : 1,
            "dominant_cluster_method": method,
            "n_frp_strict_used"      : int(valid.sum()),
        }

    cluster_info_export = sorted(
        [{k: v for k, v in c.items() if k != "mask"} for c in cluster_info],
        key=lambda x: x["sum_frp"], reverse=True,
    )

    return _make_result(best_by_count), _make_result(best_by_sum), n_clusters, cluster_info_export, cluster_id_grid


# ---------------------------------------------------------------------------
# v6.py line 213
# ---------------------------------------------------------------------------
def compute_source_confidence(src_count: dict, src_sum: dict,
                               max_dist_agreement_km: float = 5.0):
    """
    Return (flag: int 0-2, label: str, details: dict).
    flag 0 = no source, 1 = low agreement, 2 = high agreement.
    """
    details = {"agreement": 0}

    if src_count["source_is_localized"] == 0 or src_sum["source_is_localized"] == 0:
        return 0, "none", details

    d = haversine_km(
        src_count["source_lat"], src_count["source_lon"],
        src_sum["source_lat"],   src_sum["source_lon"],
    )
    details["agreement"] = 2 if d < max_dist_agreement_km else 1
    flag  = details["agreement"]
    label = {0: "none", 1: "low", 2: "high"}
    return flag, label[flag], details
