"""
wekeo_combined_chain.postprocess
================================

Post-processing of the combined L3 dataset: plume × FRP association,
fire scores, source localisation, and map generation.

Adapted from combine_S5P-PCA_FRP_v6.py (Sarah Pipien, 2026).

Typical usage
-------------
    import xarray as xr
    from wekeo_combined_chain import postprocess

    ds_combined = xr.open_dataset(
        "/mnt/ceph/proj/WEKEO/CLEAN/OUT/gridded_combined/s5p_pca_plumes_frp_iasi_product.nc"
    )

    # Run both stages
    df_plumes = postprocess.compute_plume_stats(ds_combined)
    df_plumes.to_csv("plumes_summary.csv", index=False, float_format="%.4f")
    ds_post = postprocess.build_grids(ds_combined, df_plumes)

    # Or one-shot
    ds_post, df_plumes = postprocess.compute(ds_combined)

    # Save the output dataset
    ds_post.to_netcdf("postprocess_output.nc")

    # Generate all maps
    postprocess.plot(ds_combined, ds_post, df_plumes, output_dir="./plots", date_str="20210817")
"""

from .compute import compute, compute_plume_stats, build_grids
from .plot import plot

__all__ = ["compute", "compute_plume_stats", "build_grids", "plot"]

