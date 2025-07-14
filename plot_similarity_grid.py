# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def plot_time_series_grid_advanced(real_series_dict, generated_series_dicts, n_real=5, figsize=(26, 12)):
    """
    Plot a grid where each row is a time series type and each column is either a real or GAN-generated sample.
    The last real sample is used to find the closest GAN sample from each method using Wasserstein distance.

    Parameters:
    - real_series_dict: dict[str, list[np.ndarray]]
        e.g., {"sine": list of 1D arrays of shape (32,)}
    - generated_series_dicts: dict[str, dict[str, list[np.ndarray]]]
        e.g., {"sine": {"WGAN-XIRP": [...], "TimeGAN": [...], ...}}
    - n_real: number of real samples to show before GAN examples
    - figsize: overall figure size
    """

    ts_types = list(real_series_dict.keys())
    model_names = list(next(iter(generated_series_dicts.values())).keys())
    n_rows = len(ts_types)
    n_cols = n_real + len(model_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is 2D
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, ts_type in enumerate(ts_types):
        real_series = real_series_dict[ts_type]
        generated_by_model = generated_series_dicts[ts_type]

        selected_real = real_series[:n_real]
        reference = selected_real[-1]

        # Plot real samples
        for j, series in enumerate(selected_real):
            ax = axes[i, j]
            ax.plot(series, color='blue')
            ax.set_box_aspect(1)  # make the subplot square
            #ax.set_xticks([])
            #ax.set_yticks([])
            if i == 0:
                ax.set_title(f"Original {j+1}", fontsize=9)

         # Plot generated samples
        for k, model_name in enumerate(model_names):
            samples = generated_by_model[model_name]
            distances = [wasserstein_distance(reference, g) for g in samples]
            best_sample = samples[np.argmin(distances)]

            col_idx = n_real + k
            ax = axes[i, col_idx]
            ax.plot(best_sample, color='red')
            ax.set_box_aspect(1)  # make the subplot square
            if i == 0:
                formatted_name = ' '.join(part.capitalize() for part in model_name.split('_'))
                ax.set_title(formatted_name, fontsize=9)

        # Row label
        axes[i, 0].set_ylabel(ts_type, rotation=0, labelpad=60, fontsize=10, va='center')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # less spacing between subplots
    plt.show()

# Load real data
real_sine = np.load('/data/ori_seqs/ori_data_seqs_WGAN_GP_sine_shift_40.npy')
real_bike_share = np.load('/data/ori_seqs/ori_data_seqs_WGAN_GP_bike_share_casual.npy')
stock_data = np.load('/data/ori_seqs/ori_data_seqs_TimeGAN_stock_data_Close.npy')
air_quality = np.load('/data/ori_seqs/ori_data_seqs_Diffusion_air_quality_RH.npy')
merton_process = np.load('/data/ori_seqs/ori_data_seqs_TimeGAN_merton_process_lamJ_1.6_sigJ_0.9.npy')


# Shuffle along axis 0 and pick first 5
np.random.seed(42)

shuffled_sine = np.random.permutation(real_sine)
shuffled_bike_share = np.random.permutation(real_bike_share)
shuffled_stock_data = np.random.permutation(stock_data)
shuffled_air_quality = np.random.permutation(air_quality)
merton_process = np.random.permutation(merton_process)


# Assign to dict
real_series_dict = {
    "sine": shuffled_sine,
    "bike_share": shuffled_bike_share,
    "stock_data": shuffled_stock_data,
    "air_quality": shuffled_air_quality,
    "merton_process": merton_process,


}

# Load generated synthetic data (all from data/syn_seqs/)
generated_series_dicts = {
    "sine": {
        "WGAN-XIRP": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_sine_shift_40_columns_random.npy'),
        "WGAN-GASF": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_sine_shift_40_columns_random_gasf.npy'),
        "WGAN-Naive": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_sine_shift_40_columns_random_unthresholded.npy'),
        "TimeGAN": np.load('data/syn_seqs/syn_data_seqs_TimeGAN_sine_shift_40_.npy'),
        "Diffusion": np.load('data/syn_seqs/syn_data_seqs_Diffusion_sine_shift_40_columns_mean.npy'),
    },
    "bike_share": {
        "WGAN-XIRP": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_bike_share_casual_columns_random_xirp.npy'),
        "WGAN-GASF": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_bike_share_casual_columns_random_gasf.npy'),
        "WGAN-Naive": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_bike_share_casual_columns_random_unthresholded.npy'),
        "TimeGAN": np.load('data/syn_seqs/syn_data_seqs_TimeGAN_bike_share_casual_.npy'),
        "Diffusion": np.load('data/syn_seqs/syn_data_seqs_Diffusion_bike_share_casual_columns_mean.npy'),
    },
    "stock_data": {
        "WGAN-XIRP": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_stock_data_Close_columns_random_xirp.npy'),
        "WGAN-GASF": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_stock_data_Close_columns_random_gasf.npy'),
        "WGAN-Naive": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_stock_data_Close_columns_random_unthresholded.npy'),
        "TimeGAN": np.load('data/syn_seqs/syn_data_seqs_TimeGAN_stock_data_Close_.npy'),
        "Diffusion": np.load('data/syn_seqs/syn_data_seqs_Diffusion_stock_data_Close_columns_mean.npy'),
    },
    "air_quality": {
        "WGAN-XIRP": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_air_quality_RH_columns_random_xirp.npy'),
        "WGAN-GASF": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_air_quality_RH_columns_random_gasf.npy'),
        "WGAN-Naive": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_air_quality_RH_columns_random_unthresholded.npy'),
        "TimeGAN": np.load('data/syn_seqs/syn_data_seqs_TimeGAN_air_quality_RH_.npy'),
        "Diffusion": np.load('data/syn_seqs/syn_data_seqs_Diffusion_air_quality_RH_columns_mean.npy'),
    },
    "merton_process": {
        "WGAN-XIRP": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_merton_process_lamJ_1.6_sigJ_0.9_columns_random_xirp.npy'),
        "WGAN-GASF": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_merton_process_lamJ_1.6_sigJ_0.9_columns_random_gasf.npy'),
        "WGAN-Naive": np.load('data/syn_seqs/syn_data_seqs_WGAN_GP_merton_process_lamJ_1.6_sigJ_0.9_columns_random_unthresholded.npy'),
        "TimeGAN": np.load('data/syn_seqs/syn_data_seqs_TimeGAN_merton_process_lamJ_1.6_sigJ_0.9_.npy'),
        "Diffusion": np.load('data/syn_seqs/syn_data_seqs_Diffusion_merton_process_lamJ_1.6_sigJ_0.9_columns_mean.npy'),
    }
}

def plot_time_series_grid_flipped(real_series_dict, generated_series_dicts, n_real=5):
    """
    Plot a flipped grid where each row is either a real or GAN model and each column is a time series type.
    For 'Real' rows, show multiple original samples; for model rows, show the best match based on Wasserstein distance.
    Optimized for DIN A4 portrait.

    Parameters:
    - real_series_dict: dict[str, list[np.ndarray]]
    - generated_series_dicts: dict[str, dict[str, list[np.ndarray]]]
    - n_real: how many real samples to show (per type) in the 'Real' row
    """

    ts_types = list(real_series_dict.keys())
    model_names = list(next(iter(generated_series_dicts.values())).keys())
    row_labels = ['Real'] * n_real + model_names
    n_rows = n_real + len(model_names)
    n_cols = len(ts_types)

    # DIN A4 portrait
    figsize = (10.27, 14.69)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for j, ts_type in enumerate(ts_types):
        real_series = real_series_dict[ts_type]
        reference = real_series[n_real - 1]

        # Plot real samples
        for i in range(n_real):
            ax = axes[i, j]
            ax.plot(real_series[i], color='#CC9633')

            ax.set_box_aspect(.75)
            if i == 0:
                formatted_type = ' '.join(part.capitalize() for part in ts_type.split('_'))
                ax.set_title(formatted_type, fontsize=10)

        # Plot best generated samples
        for k, model_name in enumerate(model_names):
            samples = generated_series_dicts[ts_type][model_name]
            distances = [wasserstein_distance(reference, g) for g in samples]
            best_sample = samples[np.argmin(distances)]

            row_idx = n_real + k
            ax = axes[row_idx, j]
            ax.plot(best_sample, color='#70B2E4')

            ax.set_box_aspect(.75)

    # Custom formatting for row labels
    def format_model_label(name):
        name_lower = name.lower()
        print(name_lower)
        if name_lower == "wgan-xirp":
            return "WGAN-GP \n (XIRP)"
        elif name_lower == "wgan-gasf":
            return "WGAN-GP \n (GASF)"
        elif name_lower == "wgan-naive":
            return "WGAN-GP \n (Naive)"
        elif name_lower == "timegan":
            return "TimeGAN \n    "
        elif name_lower == "diffusion":
            return "Diffusion \n    "
        else:
            return ' '.join(part.capitalize() for part in name.split('_'))

    # Set row labels with left alignment
    for i in range(n_rows):
        if i < n_real:
            label = f"Original {i+1} \n   "
        else:
            model_name = model_names[i - n_real]
            label = format_model_label(model_name)
        axes[i, 0].text(-0.8, 0.5, label,
                fontsize=10,
                rotation=90,
                va='center',
                ha='center',  # center-align entire text block
                multialignment='center',
                transform=axes[i, 0].transAxes)


    plt.subplots_adjust(wspace=0.1, hspace=0.375)
    plt.savefig("time_series_grid_A4_flipped.pdf", bbox_inches='tight')  # Optional
    plt.show()

plot_time_series_grid_flipped(real_series_dict, generated_series_dicts, n_real=5)