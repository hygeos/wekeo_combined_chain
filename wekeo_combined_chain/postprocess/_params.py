# Tuning parameters for the postprocess pipeline.
# Adapted from combine_S5P-PCA_FRP_v6.py lines 318–330.
# Edit these to tune buffer sizing, source detection, etc.

# --- Dynamic plume envelope (buffer) ---
BUF_COEFF     = 0.2   # multiplier of sqrt(n_pixels)
BUF_MIN       = 5     # minimum buffer in pixels (~60 km on 0.11° grid)
BUF_MAX       = 40    # maximum buffer in pixels (~480 km)
MAX_EXTRA_ITER = 5    # extra adaptive iterations beyond the initial buffer

# --- Source estimation ---
SOURCE_N_MIN        = 2    # minimum FRP pixels to declare a source
MAX_CLUSTERS_SAVED  = 10   # top-N clusters saved in the per-plume CSV

# --- Source confidence ---
CONF_MAX_DIST_AGREE_KM = 5.0   # max distance (km) between count/sum methods to agree

# ---------------------------------------------------------------------------
# Variable name mapping: short internal name → prefixed var in combined ds
# FRP: day_ variants are used; adjust if night detections should be included
# (see combine_S5P-PCA_FRP_v6.py line ~358 for the original FRP_VARS list)
# ---------------------------------------------------------------------------

FRP_VAR_MAP = {
    "FRP_SWIR_no_SAA_mean"  : "frp_slstr__day_FRP_SWIR_no_SAA_mean",
    "FRP_SWIR_no_SAA_count" : "frp_slstr__day_FRP_SWIR_no_SAA_count",
    "FRP_SWIR_no_SAA_max"   : "frp_slstr__day_FRP_SWIR_no_SAA_max",
    "FRP_SWIR_no_SAA_min"   : "frp_slstr__day_FRP_SWIR_no_SAA_min",
    "FRP_SWIR_no_SAA_std"   : "frp_slstr__day_FRP_SWIR_no_SAA_std",
    "FRP_MWIR_mean"         : "frp_slstr__day_FRP_MWIR_mean",
    "FRP_MWIR_count"        : "frp_slstr__day_FRP_MWIR_count",
    "FRP_MWIR_max"          : "frp_slstr__day_FRP_MWIR_max",
    "FRP_MWIR_min"          : "frp_slstr__day_FRP_MWIR_min",
    "FRP_MWIR_std"          : "frp_slstr__day_FRP_MWIR_std",
}

# S5P-PCA input variables in combined dataset
# (see v6.py lines ~377-382 for the original unprefixed names)
S5P_SCORE_VAR   = "s5p_pca__mean_score_CO"   # per-pixel mean CO score
S5P_SAMPLES_VAR = "s5p_pca__nb_samples"      # sample count (used to build valid mask)
S5P_LABELS_VAR  = "s5p_pca__plume_labels"    # plume label grid (0 = background)
