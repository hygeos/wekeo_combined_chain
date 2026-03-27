from datetime import date, datetime
from pathlib import Path
import tempfile

import xarray as xr

from wekeo_frp_l3 import frp_slstr
from wekeo_s5p_pca_l3 import s5p_pca
from wekeo_iasi_l3 import iasi

from wekeo_combined_chain.hygeos_core import env


def get_combined_product(*, 
    s5p_pca_product_path: Path|None=None, 
    day: date|None=None,
    frp_slstr_l3: bool = True,
    iasi_l3: bool = True,
    width: int = 3272 
) -> xr.Dataset:
    
    """
    Get the combined product of S5P_PCA, IASI, and FRP SLSTR Level 3 datasets.
    Parameters
    ----------
    s5p_pca_product_path : Path, optional
        Path to the S5P_PCA Level 3 product. If not provided, the product will be retrieved based on the provided day.
    day : date, optional
        Day for which to retrieve the products. If not provided, the S5P_PCA product will be bypassed and only the IASI and FRP SLSTR products will be retrieved if their respective flags are set to True.
    frp_slstr_l3 : bool, optional
        Whether to include the FRP SLSTR Level 3
    iasi_l3 : bool, optional
        Whether to include the IASI Level 3
    width : int, optional
        Width of the grid for the gridded products. Default is 3272, which corresponds to a grid resolution of approximately 0.1 degrees at the equator.
    Returns
    -------
    xr.Dataset
        Combined dataset of Level 3 datasets of S5P_PCA, IASI, and FRP SLSTR.
    """


    WIDTH = width # resolution of output grid
    
    # storage objects for merged dataset
    attrs = {"content": []} 
    products = []

    if s5p_pca_product_path is None and day is None:
        raise ValueError("Either s5p_pca_product_path or day must be provided.")
    
    if s5p_pca_product_path is not None and day is not None:
        raise ValueError("Only one of s5p_pca_product_path or day can be provided.")

    if not s5p_pca_product_path and not (iasi_l3 or frp_slstr_l3):
        raise ValueError("At least one of the products must be included. Provide a path to the S5P_PCA product or set iasi_l3 or frp_slstr_l3 to True.")

    if s5p_pca_product_path:
        
        # --------------------------------------------
        # Get the S5P_PCA Level 3 product
        # --------------------------------------------
        
        print("Getting S5P_PCA Level 3 product...")
        
        input_file = s5p_pca_product_path
        ds_s5p_pca = s5p_pca.get_gridded_s5p_pca_l3(
            dataset=input_file,
            width=WIDTH,
            min_count=1,
            save_result=True,
            use_cache=True,
        )

        date = datetime.strptime(ds_s5p_pca.attrs["date"], "%Y-%m-%d")
        
        # prefix all data_vars with "s5p_pca_"
        ds_s5p_pca = ds_s5p_pca.rename({var: f"s5p_pca__{var}" for var in ds_s5p_pca.data_vars if var not in ds_s5p_pca.coords}) 
        
        products.append(ds_s5p_pca)
        attrs["s5p_pca"] = str(ds_s5p_pca.attrs) # str to serialize
        attrs["content"].append("s5p_pca")

    # If day is provided, use it as the date for the products
    if day:
        date = day
    
    if iasi_l3:
    
        # --------------------------------------------
        # Get the IASI Level 3 product
        # --------------------------------------------
        
        print("Getting IASI Level 3 product...")
        
        ds_iasi = iasi.get_gridded_iasi_l3(
            day=date,
            width=WIDTH,
            variables=["INTEGRATED_CO"],
            remove_night=True,
            save_result=True,
            use_cache=True,
        )
        
        # prefix all data_vars with "iasi_"
        ds_iasi = ds_iasi.rename({var: f"iasi__{var}" for var in ds_iasi.data_vars if var not in ds_iasi.coords})
        
        products.append(ds_iasi)
        attrs["iasi"] = str(ds_iasi.attrs) # str to serialize
        attrs["content"].append("iasi")
    
    if frp_slstr_l3:
            
        # --------------------------------------------
        # Get the FRP SLSTR Level 3 product
        # --------------------------------------------
        
        print("Getting FRP SLSTR Level 3 product...")
        
        ds_frp_slstr = frp_slstr.get_gridded_frp_slstr_l3(
            day=date,
            width=WIDTH,
            min_count=1,
            save_result=True,
            use_cache=True,
        )
        
        # prefix all data_vars with "frp_slstr_"
        ds_frp_slstr = ds_frp_slstr.rename({var: f"frp_slstr__{var}" for var in ds_frp_slstr.data_vars if var not in ds_frp_slstr.coords})
        
        products.append(ds_frp_slstr)
        attrs["frp_slstr"] = str(ds_frp_slstr.attrs) # str to serialize
        attrs["content"].append("frp_slstr")
    
    # merge the three datasets
    ds_combined = xr.merge(products, compat="no_conflicts")
    ds_combined.attrs = attrs | {
        "description": "Combined dataset of Level3 datasets of S5P_PCA, IASI, and FRP SLSTR.",
        "date": date.strftime("%Y-%m-%d"),
    }
    
    return ds_combined


def save_combined_product(ds: xr.Dataset, output_dir: Path=None) -> None:
    """
    Save the combined dataset to a NetCDF file.
    Parameters
    ----------
    ds : xr.Dataset
        Combined dataset to save.
    output_path : Path
        Path to save the NetCDF file.
    """

    version = "v1"

    outdir = output_dir
    if outdir is None:
        outdir = env.getdir("OUTPUT_DIR")
    
    if not outdir.exists():
        raise ValueError(f"Output directory {outdir} does not exist.")
        
    content = "_".join(ds.attrs["content"])
    outfile = outdir / f"wekeo_l3_combined__{ds.attrs['date']}__{content}__{version}.nc"
    
    print(f"Saving combined dataset to {outfile}...")
    
    with tempfile.NamedTemporaryFile(suffix='.nc', dir=outfile.parent, delete=False) as f:
        tmp_path = Path(f.name)

    try:
        ds.to_netcdf(tmp_path)
        tmp_path.replace(outfile)  # Atomic rename
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise