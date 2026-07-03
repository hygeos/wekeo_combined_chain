#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:06:38 2026

@author: spipien
"""

from datetime import datetime, timedelta
from IPython.display import display, Image as IPImage
import io
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image

def animate_cartes_journalieres(ds, varname, date_debut, date_fin, duration_ms=500):
    """
    Animation GIF des cartes journalières d'une variable sur une période.

    Paramètres
    ----------
    ds       : xarray.Dataset avec dims (time, latitude, longitude)
    varname  : nom de la variable à afficher
    date_debut, date_fin : str 'YYYY-MM-DD' (bornes incluses)
    duration_ms : durée par frame en ms
    """
    #--------------------------------------------------------------

    def masque_valide(arr):
        """Remplace les -1 par NaN et retourne un tableau float."""
        arr = arr.astype(float).copy()
        arr[arr == -1] = np.nan
        return arr
    
    def colormap_pour_variable(varname):
        """Retourne une colormap adaptée au nom de variable."""
        if "score" in varname or "diag" in varname:
            return "jet"
        elif "FRP" in varname:
            return "hot_r"
        elif "cloud" in varname:
            return "Blues"
        elif "detection" in varname or "detect" in varname:
            return "YlOrRd"
        else:
            return "viridis"
        
    #--------------------------------------------------------------
    
    if varname not in ds:
        print(f"Variable '{varname}' absente du dataset.")
        return

    # Sélection de la période
    ds_periode = ds.sel(time=slice(date_debut, date_fin))

    if ds_periode.sizes["time"] == 0:
        print(f"Aucune date trouvée entre {date_debut} et {date_fin}.")
        return

    ref_lats = ds_periode["latitude"].values
    ref_lons = ds_periode["longitude"].values

    # vmin/vmax global sur la période
    data_periode = ds_periode[varname].values  # (time, lat, lon)
    data_periode = masque_valide(data_periode)  # à adapter si masque_valide attend 2D
    vals = data_periode[~np.isnan(data_periode)]
    if len(vals) == 0:
        print(f"Aucune valeur valide pour '{varname}' sur la période.")
        return
    vmin = np.nanpercentile(vals, 2)
    vmax = np.nanpercentile(vals, 98)

    cmap = colormap_pour_variable(varname)
    lat_min, lat_max = ref_lats.min(), ref_lats.max()
    lon_min, lon_max = ref_lons.min(), ref_lons.max()

    frames = []

    for t in ds_periode["time"].values:
        date_str = str(t)[:10]  # 'YYYY-MM-DD'
        arr = ds_periode[varname].sel(time=t).values  # 2D (lat, lon)
        arr = masque_valide(arr)

        fig, ax = plt.subplots(figsize=(10, 6),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        im = ax.pcolormesh(ref_lons, ref_lats, arr,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree())
        plt.colorbar(im, ax=ax, shrink=0.7, label=varname)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)/home/joackim/Downloads/thunderbird.tmp/pid-24433/maj_notebook2/wekeo_combined_chain/animation
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=":")
        ax.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        gl.top_labels   = False
        gl.right_labels = False
        ax.set_title(f"{varname}  |  {date_str}", fontsize=20)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)

    # Assemblage GIF en mémoire
    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf, format='GIF',
        save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0
    )
    gif_buf.seek(0)

    print(f"{len(frames)} frames | {date_debut} → {date_fin}")
    display(IPImage(data=gif_buf.read()))
