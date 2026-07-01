"""
Timeseries plotting functions for combined datasets
Used in demo_time_bis.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import xarray as xr
from typing import Optional, Tuple, Union
from pathlib import Path
import pandas as pd

def plot_plume_combined_timeseries(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_fig_path: Optional[Union[str, Path]] = None
) -> Tuple[Figure, np.ndarray]:
    """
    Combine plot_plume_timeseries et plot_plume_pixels_timeseries (normal + tiny)
    en une figure à 2 sous-graphiques empilés, sous forme de barres groupées.

    Subplot 1 (haut) : nombre de panaches uniques par jour (plumes vs tiny plumes)
    Subplot 2 (bas)  : nombre de pixels par jour (plumes vs tiny plumes)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset avec dimension 'time' et variable 's5p_pca__plume_labels'
    figsize : Tuple[int, int], optional
        Taille de la figure, par défaut (12, 10)
    title : str, optional
        Titre global de la figure, par défaut None (auto-généré)
    save_fig_path : str or Path, optional
        Chemin de sauvegarde. Si None, affiche la figure, par défaut None

    Returns
    -------
    Tuple[Figure, np.ndarray]
        La figure et un tableau des 2 axes [ax_plumes, ax_pixels]
    """
    if 's5p_pca__plume_labels' not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__plume_labels' variable")

    if 'time' not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    plume_labels = ds['s5p_pca__plume_labels']
    times = ds.time.values

    plumes_count = []
    tiny_plumes_count = []
    pixels_normal_count = []
    pixels_tiny_count = []

    for t in times:
        labels_at_t = plume_labels.sel(time=t).values

        # --- Comptage des labels uniques (logique de plot_plume_timeseries) ---
        valid_labels_unique = labels_at_t[(~np.isnan(labels_at_t)) & (labels_at_t != 0)]
        if len(valid_labels_unique) > 0:
            unique_labels = np.unique(valid_labels_unique)
            plumes_count.append(int(np.sum(unique_labels < 100)))
            tiny_plumes_count.append(int(np.sum(unique_labels >= 100)))
        else:
            plumes_count.append(0)
            tiny_plumes_count.append(0)

        # --- Comptage des pixels (logique de plot_plume_pixels_timeseries) ---
        valid_labels_px = labels_at_t[~np.isnan(labels_at_t)]
        mask_normal = (valid_labels_px > 0) & (valid_labels_px < 100)
        mask_tiny = valid_labels_px >= 100
        pixels_normal_count.append(int(np.sum(mask_normal)))
        pixels_tiny_count.append(int(np.sum(mask_tiny)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    x = np.arange(len(times))
    width = 0.4

    # --- Subplot 1 : nombre de panaches ---
    ax1.bar(x - width / 2, plumes_count, width=width,
            color='#2E86AB', label='Plumes (label < 100)')
    ax1.bar(x + width / 2, tiny_plumes_count, width=width,
            color='#A23B72', label='Tiny Plumes (label ≥ 100)')

    ax1.set_ylabel('Number of Unique Plumes', fontsize=12, fontweight='bold')
    ax1.set_title('Plume Detection Count', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # --- Subplot 2 : nombre de pixels ---
    ax2.bar(x - width / 2, pixels_normal_count, width=width,
            color='#2E86AB', label='Plume Cells (label < 100)')
    ax2.bar(x + width / 2, pixels_tiny_count, width=width,
            color='#A23B72', label='Tiny Plume Cells (label ≥ 100)')

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [np.datetime_as_string(t, unit="D") for t in times],
        rotation=45, ha='right'
    )
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
    ax2.set_title('Plume Cell Count', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    if title is None:
        start_date = np.datetime_as_string(times[0], unit="D")
        end_date = np.datetime_as_string(times[-1], unit="D")
        title = f'Plume Timeseries ({start_date} to {end_date})'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_fig_path}")
    else:
        plt.show()

    return fig, np.array([ax1, ax2])

def plot_sources_timeseries(
    df_summary: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None
) -> Tuple[Figure, Axes]:
    """
    Plot timeseries of daily number of detected fire sources.

    Parameters
    ----------
    df_summary : pd.DataFrame
        DataFrame indexed by date with column 'n_sources_detected'
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 6)
    title : str, optional
        Plot title, by default None (auto-generated)
    ax : Axes, optional
        Matplotlib axes to plot on, by default None (creates new figure)
    save_fig_path : str or Path, optional
        Path to save figure. If None, displays the plot, by default None

    Returns
    -------
    Tuple[Figure, Axes]
    """
    if 'n_sources_detected' not in df_summary.columns:
        raise ValueError("df_summary must contain column 'n_sources_detected'")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(df_summary.index, df_summary["n_sources_detected"],
           color='crimson', alpha=0.8, width=0.6, label='Sources detected')
    # ax.plot(df_summary.index, df_summary["n_sources_detected"],
    #         marker='o', linestyle='-', linewidth=1.5, markersize=5,
    #         color='#3D405B', alpha=0.6)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sources Detected', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if title is None:
        title = (f'Daily Number of Fire Sources Detected '
                 f'({df_summary.index[0]} to {df_summary.index[-1]})')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_fig_path}")
    else:
        plt.show()

    return fig, ax
    
def plot_fire_score_timeseries(
    df_summary: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot timeseries of daily mean and max fire score as bar charts.

    Parameters
    ----------
    df_summary : pd.DataFrame
        DataFrame indexed by date with columns 'mean_fire_score' and 'max_fire_score'.
        NaN values are treated as zero (no fire detected that day).
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 6)
    title : str, optional
        Plot title, by default None (auto-generated)
    ax : Axes, optional
        Matplotlib axes to plot on, by default None (creates new figure)
    save_fig_path : str or Path, optional
        Path to save figure. If None, displays the plot, by default None

    Returns
    -------
    Tuple[Figure, Axes]
    """
    for col in ("mean_fire_score", "max_fire_score"):
        if col not in df_summary.columns:
            raise ValueError(f"df_summary must contain column '{col}'")

    # Fill NaN → 0 (no fire detected that day)
    mean_scores = df_summary["mean_fire_score"].fillna(0)
    max_scores = df_summary["max_fire_score"].fillna(0)

    # Warn if everything is NaN
    if mean_scores.eq(0).all() and max_scores.eq(0).all():
        print("WARNING: All fire scores are NaN or zero. "
              "This likely means no FRP data was associated with any plume.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = np.arange(len(df_summary.index))
    width = 0.4

    ax.bar(x - width / 2, mean_scores, width=width, color="gold", label="Daily mean fire score")
    ax.bar(x + width / 2, max_scores, width=width, color="b", alpha=0.7, label="Daily max fire score")

    ax.set_xticks(x)
    ax.set_xticklabels(df_summary.index, rotation=45, ha="right")

    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fire Score", fontsize=12, fontweight="bold")

    if title is None:
        title = f"Daily Fire Score ({df_summary.index[0]} to {df_summary.index[-1]})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()

    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_fig_path}")
    else:
        plt.show()

    return fig, ax
    

def plot_plume_pixels_timeseries(
    ds: xr.Dataset,
    plume_size: str = "normal",
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None
) -> Tuple[Figure, Axes]:
    """
    Plot timeseries of number of pixels labeled as plumes or tiny plumes.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'time' dimension and 's5p_pca__plume_labels' variable
    plume_size : str, optional
        Type of plumes to plot: "normal" (label < 100) or "tiny" (label >= 100), by default "normal"
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 6)
    title : str, optional
        Plot title, by default None (auto-generated)
    ax : Axes, optional
        Matplotlib axes to plot on, by default None (creates new figure)
    save_fig_path : str or Path, optional
        Path to save figure. If None, displays the plot with plt.show(), by default None
        
    Returns
    -------
    Tuple[Figure, Axes]
        The figure and axes objects
    """
    
    if 's5p_pca__plume_labels' not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__plume_labels' variable")
    
    if 'time' not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")
    
    if plume_size not in ["normal", "tiny"]:
        raise ValueError("plume_size must be either 'normal' or 'tiny'")
    
    # Get plume labels
    plume_labels = ds['s5p_pca__plume_labels']
    
    # Initialize array for pixel counts
    times = ds.time.values
    pixel_counts = []
    
    # Count pixels at each time step
    for t in times:
        labels_at_t = plume_labels.sel(time=t).values
        
        # Remove NaN values
        valid_labels = labels_at_t[~np.isnan(labels_at_t)]
        
        # Count pixels based on plume_size parameter
        if plume_size == "normal":
            # Plumes: label > 0 and < 100
            mask = (valid_labels > 0) & (valid_labels < 100)
        else:  # tiny
            # Tiny plumes: label >= 100
            mask = valid_labels >= 100
        
        pixel_counts.append(np.sum(mask))
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Select color and marker based on plume type
    if plume_size == "normal":
        color = '#2E86AB'
        marker = 'o'
        # label_text = 'Plume Pixels (label < 100)'
        # SP
        label_text = 'Plume Cells (label < 100)'

    else:  # tiny
        color = '#A23B72'
        marker = 's'
        # label_text = 'Tiny Plume Pixels (label ≥ 100)'
        label_text = 'Tiny Plume Cells (label ≥ 100)'

    
    # Plot timeseries
    ax.plot(times, pixel_counts, marker=marker, linestyle='-', linewidth=2, 
            color=color, markersize=6)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Number of Pixels', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')

    
    if title is None:
        start_date = times[0]
        end_date = times[-1]
        title = f'{label_text} - Timeseries ({np.datetime_as_string(start_date, unit="D")} to {np.datetime_as_string(end_date, unit="D")})'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels for better readability
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Show or save the figure
    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_fig_path}")
    else:
        plt.show()
    
    return fig, ax
    

def plot_plume_total_combined_timeseries(
    ds: xr.Dataset,
    df_summary: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_fig_path: Optional[Union[str, Path]] = None
) -> Tuple[Figure, np.ndarray]:
    """
    Combine le nombre total de plumes (normal + tiny) et le nombre total de pixels
    (normal + tiny) par jour, sous forme de barres, en 2 sous-graphiques empilés.
    Ajoute le nombre de plumes confirmées (avec FRP) en axe secondaire
    sur le subplot du haut.

    Subplot 1 (haut) : nombre total de plumes par jour (barres)
                        + nombre de plumes confirmées avec FRP (axe secondaire, points)
    Subplot 2 (bas)  : nombre total de pixels par jour (barres)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset avec dimension 'time' et variable 's5p_pca__plume_labels'
    df_summary : pd.DataFrame
        DataFrame indexé par date, doit contenir les colonnes
        '% of confirmed plumes (with FRP)' et 'n_plumes'. Peut avoir un
        sous-ensemble de dates différent de ds.time (les jours absents de
        df_summary seront affichés sans valeur de plumes confirmées).
    figsize : Tuple[int, int], optional
        Taille de la figure, par défaut (12, 10)
    title : str, optional
        Titre global de la figure, par défaut None (auto-généré)
    save_fig_path : str or Path, optional
        Chemin de sauvegarde. Si None, affiche la figure, par défaut None

    Returns
    -------
    Tuple[Figure, np.ndarray]
        La figure et un tableau des 2 axes [ax_plumes, ax_pixels]
    """
    if 's5p_pca__plume_labels' not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__plume_labels' variable")

    if 'time' not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    pct_col = '% of confirmed plumes (with FRP)'
    if pct_col not in df_summary.columns:
        raise ValueError(f"df_summary must contain column '{pct_col}'")

    n_plumes_col = 'n_plumes'
    if n_plumes_col not in df_summary.columns:
        raise ValueError(f"df_summary must contain column '{n_plumes_col}'")

    plume_labels = ds['s5p_pca__plume_labels']
    times = ds.time.values

    total_plumes_count = []
    total_pixels_count = []
    n_confirmed_plumes = []

    # Index de df_summary normalisé en date() pour le lookup
    # HYPOTHÈSE à vérifier : df_summary.index est de type datetime/Timestamp,
    # convertible via pd.Timestamp(...).date()
    df_summary_by_date = {
        pd.Timestamp(idx).date(): row
        for idx, row in df_summary.iterrows()
    }

    for t in times:
        labels_at_t = plume_labels.sel(time=t).values
        day = pd.Timestamp(t).date()

        # --- Comptage total des labels uniques (plumes + tiny plumes) ---
        valid_labels_unique = labels_at_t[(~np.isnan(labels_at_t)) & (labels_at_t != 0)]
        if len(valid_labels_unique) > 0:
            unique_labels = np.unique(valid_labels_unique)
            total_plumes_count.append(int(len(unique_labels)))
        else:
            total_plumes_count.append(0)

        # --- Comptage total des pixels (normal + tiny) ---
        valid_labels_px = labels_at_t[~np.isnan(labels_at_t)]
        mask_total = valid_labels_px > 0
        total_pixels_count.append(int(np.sum(mask_total)))

        # --- Nombre de plumes confirmées (avec FRP), si la date existe dans df_summary ---
        if day in df_summary_by_date:
            pct_val = df_summary_by_date[day][pct_col]
            n_plumes_val = df_summary_by_date[day][n_plumes_col]
            n_confirmed_plumes.append(pct_val * n_plumes_val / 100.0)
        else:
            n_confirmed_plumes.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    x = np.arange(len(times))
    width = 0.6

    # --- Subplot 1 : nombre total de plumes + nombre de plumes confirmées (axe secondaire) ---
    ax1.bar(x, total_plumes_count, width=width,
            color='#2E86AB', label='Total plumes (normal + tiny)')
    ax1.set_ylabel('Total Number of Plumes', fontsize=12, fontweight='bold')
    ax1.set_title('Total Plume Detection Count', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    ax1_pct = ax1.twinx()
    ax1_pct.plot(x, n_confirmed_plumes, marker='o', linestyle='None',
                 color='crimson', label='Confirmed plumes (with FRP)')
    ax1_pct.set_ylabel('Number of Confirmed Plumes (with FRP)', fontsize=11,
                        fontweight='bold', color='crimson')
    ax1_pct.tick_params(axis='y', labelcolor='crimson')

    # Échelle commune entre les deux axes pour une comparaison directe
    max_val = max(
        max(total_plumes_count) if len(total_plumes_count) > 0 else 0,
        np.nanmax(n_confirmed_plumes) if not np.all(np.isnan(n_confirmed_plumes)) else 0,
    )
    y_top = int(np.ceil(max_val * 1.1)) if max_val > 0 else 1
    ax1.set_ylim(0, y_top)
    ax1_pct.set_ylim(0, y_top)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1_pct.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Légende combinée des deux axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_pct.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               fontsize=10, loc='best', framealpha=0.9)

    # --- Subplot 2 : nombre total de pixels ---
    ax2.bar(x, total_pixels_count, width=width,
            color='#2E86AB', label='Total cells (normal + tiny)')

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [np.datetime_as_string(t, unit="D") for t in times],
        rotation=45, ha='right'
    )
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Number of Cells', fontsize=12, fontweight='bold')
    ax2.set_title('Total Plume Cell Count', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    if title is None:
        start_date = np.datetime_as_string(times[0], unit="D")
        end_date = np.datetime_as_string(times[-1], unit="D")
        title = f'Total Plume Timeseries ({start_date} to {end_date})'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_fig_path}")
    else:
        plt.show()

    return fig, np.array([ax1, ax2])
    
    
def plot_plume_cell_coverage_timeseries(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_fig_path: Optional[Union[str, Path]] = None
) -> Tuple[Figure, Axes]:
    """
    Plot timeseries du nombre total de pixels labellisés comme panache
    (normal + tiny) par jour, avec en axe secondaire le % de pixels
    couverts par un panache par rapport à la taille totale de la grille
    (zone traitée fixe, lat x lon).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset avec dimension 'time' et variable 's5p_pca__plume_labels'
    figsize : Tuple[int, int], optional
        Taille de la figure, par défaut (12, 6)
    title : str, optional
        Titre du graphique, par défaut None (auto-généré)
    save_fig_path : str or Path, optional
        Chemin de sauvegarde. Si None, affiche la figure, par défaut None

    Returns
    -------
    Tuple[Figure, Axes]
    """
    if 's5p_pca__plume_labels' not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__plume_labels' variable")
    if 'time' not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")

    plume_labels = ds['s5p_pca__plume_labels']
    times = ds.time.values
    

    # Taille totale fixe de la grille (zone traitée), hors dimension time.
    # HYPOTHÈSE à vérifier : les seules dimensions de plume_labels sont
    # ('time', 'latitude', 'longitude') ou équivalent — si une 3e dimension
    # spatiale existe, ce calcul serait faux.
    grid_size = int(np.prod([plume_labels.sizes[d] for d in plume_labels.dims if d != 'time']))

    print(grid_size)


    total_pixels_count = []
    pct_coverage = []

    for t in times:
        labels_at_t = plume_labels.sel(time=t).values

        valid_labels_px = labels_at_t[~np.isnan(labels_at_t)]
        mask_total = valid_labels_px > 0
        n_plume_pixels = int(np.sum(mask_total))
        total_pixels_count.append(n_plume_pixels)

        pct_coverage.append(100.0 * n_plume_pixels / grid_size)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(times))
    width = 0.6

    ax.bar(x, total_pixels_count, width=width,
           color='#2E86AB', label='Total plume cells (normal + tiny)')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [np.datetime_as_string(t, unit="D") for t in times],
        rotation=45, ha='right'
    )
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Number of Plume Cells', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    ax_pct = ax.twinx()
    ax_pct.plot(x, pct_coverage, marker='o', linestyle='None',
                color='crimson', label='% of grid covered by plume')
    ax_pct.set_ylabel('% of Treated Zone Covered', fontsize=11,
                       fontweight='bold', color='crimson')
    ax_pct.set_ylim(0, max(pct_coverage) * 1.2 if max(pct_coverage) > 0 else 1)
    ax_pct.tick_params(axis='y', labelcolor='crimson')

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_pct.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              fontsize=10, loc='best', framealpha=0.9)

    if title is None:
        start_date = np.datetime_as_string(times[0], unit="D")
        end_date = np.datetime_as_string(times[-1], unit="D")
        title = f'Plume Cell Coverage ({start_date} to {end_date})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_fig_path}")
    else:
        plt.show()

    return fig, ax