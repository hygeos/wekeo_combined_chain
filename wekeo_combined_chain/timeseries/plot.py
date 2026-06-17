"""
Timeseries plotting functions for combined datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import xarray as xr
from typing import Optional, Tuple, Union
from pathlib import Path


def plot_plume_timeseries(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None
) -> Tuple[Figure, Axes]:
    """
    Plot timeseries of unique plume labels, separated into plumes (<100) and tiny plumes (>=100).
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'time' dimension and 's5p_pca__plume_labels' variable
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
    
    # Get plume labels
    plume_labels = ds['s5p_pca__plume_labels']
    
    # Initialize arrays for counts
    times = ds.time.values
    plumes_count = []
    tiny_plumes_count = []
    
    # Count unique labels at each time step
    for t in times:
        labels_at_t = plume_labels.sel(time=t).values
        
        # Remove NaN and 0 values (no plume)
        valid_labels = labels_at_t[(~np.isnan(labels_at_t)) & (labels_at_t != 0)]
        
        if len(valid_labels) > 0:
            unique_labels = np.unique(valid_labels)
            
            # Separate plumes (<100) and tiny plumes (>=100)
            plumes = unique_labels[unique_labels < 100]
            tiny_plumes = unique_labels[unique_labels >= 100]
            
            plumes_count.append(len(plumes))
            tiny_plumes_count.append(len(tiny_plumes))
        else:
            plumes_count.append(0)
            tiny_plumes_count.append(0)
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot both timeseries
    ax.plot(times, plumes_count, marker='o', linestyle='-', linewidth=2, 
            label='Plumes (label < 100)', color='#2E86AB')
    ax.plot(times, tiny_plumes_count, marker='s', linestyle='-', linewidth=2,
            label='Tiny Plumes (label ≥ 100)', color='#A23B72')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Unique Plumes', fontsize=12, fontweight='bold')
    
    if title is None:
        start_date = times[0]
        end_date = times[-1]
        title = f'Plume Detection Timeseries ({np.datetime_as_string(start_date, unit="D")} to {np.datetime_as_string(end_date, unit="D")})'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
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
        label_text = 'Plume Pixels (label < 100)'
    else:  # tiny
        color = '#A23B72'
        marker = 's'
        label_text = 'Tiny Plume Pixels (label ≥ 100)'
    
    # Plot timeseries
    ax.plot(times, pixel_counts, marker=marker, linestyle='-', linewidth=2, 
            color=color, markersize=6)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Pixels', fontsize=12, fontweight='bold')
    
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


def plot_detected_pixels_timeseries(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None
) -> Tuple[Figure, Axes]:
    """
    Plot timeseries of daily number of detected pixels.
    A pixel is considered detected if s5p_pca__mean_score_CO is not NaN and not a fill value.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'time' dimension and 's5p_pca__mean_score_CO' variable
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
    
    if 's5p_pca__mean_score_CO' not in ds.data_vars:
        raise ValueError("Dataset must contain 's5p_pca__mean_score_CO' variable")
    
    if 'time' not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")
    
    # Get the variable
    mean_score_co = ds['s5p_pca__mean_score_CO']
    
    # Initialize array for pixel counts
    times = ds.time.values
    detected_pixels = []
    
    # Count detected pixels at each time step
    for t in times:
        values_at_t = mean_score_co.sel(time=t).values
        
        # Count valid pixels (not NaN and not fill values)
        # Assuming fill values are typically very large negative or positive numbers
        # We consider a pixel detected if it's finite (not NaN, not inf)
        valid_mask = np.isfinite(values_at_t)
        detected_pixels.append(np.sum(valid_mask))
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot timeseries
    ax.plot(times, detected_pixels, marker='o', linestyle='-', linewidth=2, 
            color='#F18F01', markersize=6)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Detected Pixels', fontsize=12, fontweight='bold')
    
    if title is None:
        start_date = times[0]
        end_date = times[-1]
        title = f'Daily Detected Pixels (s5p_pca__mean_score_CO) - ({np.datetime_as_string(start_date, unit="D")} to {np.datetime_as_string(end_date, unit="D")})'
    
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


def plot_frp_pixels_timeseries(
    ds: xr.Dataset,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_fig_path: Optional[Union[str, Path]] = None
) -> Tuple[Figure, Axes]:
    """
    Plot timeseries of daily number of FRP pixels.
    A pixel is considered as FRP if either frp_slstr__day_FRP_MWIR_mean or 
    frp_slstr__night_FRP_MWIR_mean is not NaN and not a fill value.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'time' dimension and FRP variables
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
    
    # Check for required variables
    day_var = 'frp_slstr__day_FRP_MWIR_mean'
    night_var = 'frp_slstr__night_FRP_MWIR_mean'
    
    has_day = day_var in ds.data_vars
    has_night = night_var in ds.data_vars
    
    if not has_day and not has_night:
        raise ValueError(f"Dataset must contain at least one of '{day_var}' or '{night_var}'")
    
    if 'time' not in ds.dims:
        raise ValueError("Dataset must have 'time' dimension")
    
    # Initialize array for pixel counts
    times = ds.time.values
    frp_pixels = []
    
    # Count FRP pixels at each time step
    for t in times:
        combined_mask = np.zeros_like(ds[list(ds.data_vars)[0]].sel(time=t).values, dtype=bool)
        
        # Check day FRP if available
        if has_day:
            day_values = ds[day_var].sel(time=t).values
            day_mask = np.isfinite(day_values)
            combined_mask |= day_mask
        
        # Check night FRP if available
        if has_night:
            night_values = ds[night_var].sel(time=t).values
            night_mask = np.isfinite(night_values)
            combined_mask |= night_mask
        
        frp_pixels.append(np.sum(combined_mask))
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot timeseries
    ax.plot(times, frp_pixels, marker='o', linestyle='-', linewidth=2, 
            color='#C73E1D', markersize=6)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of FRP Pixels', fontsize=12, fontweight='bold')
    
    if title is None:
        start_date = times[0]
        end_date = times[-1]
        title = f'Daily FRP Pixels (day/night MWIR) - ({np.datetime_as_string(start_date, unit="D")} to {np.datetime_as_string(end_date, unit="D")})'
    
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
