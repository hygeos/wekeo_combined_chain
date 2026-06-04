import xarray as xr
from typing import Literal


def center_longitude(ds: xr.Dataset, center: Literal[0, 180]=0, lon_name: str="longitude") -> xr.Dataset:
    """
    Center longitudes from [0, 360] to [-180, 180] or from [-180, 180] to [0, 360]
    """
    
    assert (center == 0.0) or (center == 180.0)
    
    lon = None
    if center == 0.0:
        lon = (ds[lon_name].values + 180) % 360 - 180
    elif center == 180.0:
        lon = (ds[lon_name].values) % 360
    
    ds = ds.assign_coords({lon_name:lon})
    ds = ds.sortby(lon_name)
    
    return ds



def select_area(ds, area: list):
    """
    Select a specific area from the dataset.
    
    Parameters:
    - ds: xarray.Dataset
    - area: list with [north, south, east, west] bounds /!\\ antemeridian crossing is possible
    
    Returns:
    - xarray.Dataset with the selected area
    """
    north, south, east, west = area

    global_longitudes = west == -180.0 and east == 180.0

    lat = ds["latitude"].values
    lat_slice = slice(north, south) if lat[0] > lat[-1] else slice(south, north)
    lon_slice = slice(west, east)
    
    if global_longitudes:
        return ds.sel(latitude=lat_slice)

    elif west < east: # Normal case
        return ds.sel(
            latitude=lat_slice,
            longitude=lon_slice
        )    

    else: # Antimeridian crossing case — shift longitudes to [0, 360]
        bs = center_longitude(ds, center=180, lon_name="longitude")

        selection = bs.sel(
            latitude=lat_slice,
            longitude=slice(west % 360, east % 360)
        )

        return center_longitude(selection, center=0, lon_name="longitude")