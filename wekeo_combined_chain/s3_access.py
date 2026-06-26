"""
S3 Zarr access variant
"""
from datetime import datetime, date
import s3fs
import xarray as xr

from wekeo_combined_chain.utils import select_area


def _make_fs(endpoint_url: str) -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        anon=False,
        client_kwargs={"endpoint_url": endpoint_url},
        config_kwargs={"max_pool_connections": 20},
        s3_additional_kwargs={"retries": {"max_attempts": 5, "mode": "adaptive"}},
    )


def _open_combined_zarr() -> xr.Dataset:
    """
    Open the yearly concatenated global COMBINED Zarr store on S3.
    Time is unordered, so callers must sort/selection by time explicitly.
    """
    BUCKET = "S5P_PCA_V0.1"
    ZARR_PATH = f"s3://{BUCKET}/COMBINED/v1.0/COMBINED_dataset_v0.1.zarr"
    fs = _make_fs("https://s3.waw4-1.cloudferro.com")
    store = fs.get_mapper(ZARR_PATH)
    # consolidated=True for fast metadata read; chunks stay lazy on S3
    return xr.open_zarr(store, consolidated=True)


def remove_vars(ds: xr.Dataset, vars_to_not_remove: list[str]) -> xr.Dataset:
    # general rules of vars to remove:
    remove = []
    vars = ds.data_vars.keys()

    patterns = ["__night", "_std", "_max", "_min", "_count"]
    for var in vars:
        if any([pattern in var for pattern in patterns]) and var not in vars_to_not_remove:
            remove.append(var)

    ds = ds.drop_vars(remove)
    return ds


def get_combined_ds_range(
    start_day: date = date(2025, 1, 1),
    end_day: date = date(2025, 12, 31),
) -> xr.Dataset:
    """
    Load a date range from the S3 Zarr store and select the requested area.

    The Zarr store is a yearly concatenated global file with unordered time,
    so we sort by time before selecting the [start_day, end_day] slice.
    Area selection is applied lazily on the resulting slice.
    """
    ds = _open_combined_zarr()

    # The store time is unordered: sort once so .sel(time=slice(...)) is reliable.
    ds = ds.sortby("time")

    # Build a [start, end] inclusive slice at day granularity.
    start = datetime(start_day.year, start_day.month, start_day.day)
    end = datetime(end_day.year, end_day.month, end_day.day)
    ds = ds.sel(time=slice(start, end))

    ds = remove_vars(ds, vars_to_not_remove=[])

    return ds

def get_combined_ds(date: date) -> xr.Dataset:
    """
    Load data from the S3 Zarr store.
    """
    BUCKET = 'S5P_PCA_V0.1'
    ZARR_PATH = f"s3://{BUCKET}/COMBINED/v1.0/COMBINED_dataset_v0.1.zarr"
    def make_fs(endpoint_url: str) -> s3fs.S3FileSystem:
            return s3fs.S3FileSystem(
                    anon=False,
                    client_kwargs={"endpoint_url": endpoint_url},
                    config_kwargs={"max_pool_connections": 20},
                    s3_additional_kwargs={"retries": {"max_attempts": 5, "mode": "adaptive"}},
            )
    fs = make_fs(" https://s3.waw4-1.cloudferro.com")
    store = fs.get_mapper(ZARR_PATH)
    
    ds = xr.open_zarr(store, consolidated=True)
    
    # return sel of == date (as datetime64[ns])
    return ds.sel(time=date.strftime("%Y-%m-%d"))