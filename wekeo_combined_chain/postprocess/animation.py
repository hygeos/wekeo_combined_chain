#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:59:07 2026

@author: spipien
"""

from datetime import date, timedelta
import io

from IPython.display import display, Image as IPImage
from PIL import Image


def _fig_to_pil(fig):
    """Sauvegarde une Figure matplotlib dans un buffer mémoire et la retourne en PIL.Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy()


def animate_fire_score(
    day, area, n_days, kind="plume", band="SWIR", duration_ms=500
):
    """
    Animation GIF des cartes fire_score sur les n_days jours précédant `day` (inclus).

    Paramètres
    ----------
    day      : datetime.date, dernier jour de la fenêtre
    area     : [north, south, east, west]
    n_days   : nombre de jours (fenêtre se terminant à `day`)
    kind     : "plume" ou "pixel"
    band     : "SWIR" ou "MWIR"
    duration_ms : durée par frame en ms
    """
    frames = []

    for k in range(n_days - 1, -1, -1):
        d = day - timedelta(days=k)
        date_str = d.strftime("%Y%m%d")

        try:
            ds = get_combined_ds(d)
        except Exception as exc:
            print(f"{date_str} : jour ignoré ({exc!r})")
            continue

        ds_area = select_area(ds, area)
        ds_post, df_plumes = postprocess.compute(ds_area)

        if kind == "plume":
            fig = P.plot_fire_score_plume(ds_area, ds_post, df_plumes, date_str, band=band)
        elif kind == "pixel":
            fig = P.plot_fire_score_pixel(ds_area, ds_post, date_str, band=band)
        else:
            raise ValueError("kind doit être 'plume' ou 'pixel'")

        frames.append(_fig_to_pil(fig))

    if not frames:
        print("Aucune frame disponible.")
        return

    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf, format="GIF",
        save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0,
    )
    gif_buf.seek(0)

    print(f"{len(frames)} frames | fire_score_{kind} {band} | se terminant le {day}")
    display(IPImage(data=gif_buf.read()))