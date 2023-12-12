import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from tueplots import axes, bundles

from neural_lam import utils 
from neural_lam import constants_tcwv as constants



@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map_wo_border(errors, title=None, step_length=3):
    """
    Plot a heatmap of errors of different variables at different predictions horizons
    errors: (pred_steps, d_f)
    """
    if errors.ndim == 1:
        errors = errors[None,:]
    errors_np = errors.mT.cpu().numpy() # (d_f, pred_steps)

    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1) # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(15,10))

    ax.imshow(errors_norm, cmap="coolwarm", vmin=0, vmax=1., interpolation="none",
            aspect="auto", alpha=0.8)

    # ax and labels
    for (j,i),error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i,j, formatted_error,ha='center',va='center', usetex=False)

    # Ticks and labels
    label_size=15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps)+1 # Prediction horiz. in index
    pred_hor_h = step_length*pred_hor_i # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (Day)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [f"{name} ({unit})" for name, unit in
            zip(constants.param_names_short, constants.param_units)]
    ax.set_yticklabels(y_ticklabels , rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig

@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction_wo_border(pred, target, title=None, vrange=None):
    # currently only plot this
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    lat = np.linspace(-20, 20, 81)
    lon = np.linspace(50, 240, 381)
    lon, lat = np.meshgrid(lon, lat)
    # Set up masking of border region
    # mask_reshaped = obs_mask.reshape(*constants.grid_shape)
    # pixel_alpha = mask_reshaped.clamp(0.7, 1).cpu().numpy() # Faded border region

    fig, axes = plt.subplots(2,1, figsize=(12,5))

    axes[0].set_title("Ground Truth", size=15)
    m = Basemap(projection='cyl', llcrnrlat=-20, urcrnrlat=20, llcrnrlon=50, urcrnrlon=240, resolution='l',ax=axes[0])
    m.drawcoastlines()
    target_grid = target.reshape(*constants.grid_shape).cpu().numpy()
    im = m.contourf(lon, lat, target_grid, 21, vmin=vmin, vmax=vmax, cmap="coolwarm")

    axes[1].set_title("Prediction", size=15)
    m = Basemap(projection='cyl', llcrnrlat=-20, urcrnrlat=20, llcrnrlon=50, urcrnrlon=240, resolution='l',ax=axes[1])
    m.drawcoastlines()
    pred_grid = pred.reshape(*constants.grid_shape).cpu().numpy()
    im = m.contourf(lon, lat, pred_grid, 21, vmin=vmin, vmax=vmax, cmap="coolwarm")

    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig

@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error_wo_border(error, title=None, vrange=None):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    # mask_reshaped = obs_mask.reshape(*constants.grid_shape)
    # pixel_alpha = mask_reshaped.clamp(0.7, 1).cpu().numpy() # Faded border region
    print(f'error shape: {error.shape}')
    fig, ax = plt.subplots(figsize=(5,4.8),
            subplot_kw={"projection": constants.lambert_proj})

    ax.coastlines() # Add coastline outlines
    error_grid = error.reshape(*constants.grid_shape).cpu().numpy()

    im = ax.imshow(error_grid, origin="lower", extent=constants.grid_limits,
                    vmin=vmin, vmax=vmax, cmap="coolwarm")

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig

