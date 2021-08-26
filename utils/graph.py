'''
Some code snippets are borrowed from LEAF.

LEAF: A Benchmark for Federated Settings
Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li,
Jakub Konečný, H. Brendan McMahan, Virginia Smith, and Ameet Talwalkar.
Workshop on Federated Learning for Data Privacy and Confidentiality (2019).

https://leaf.cmu.edu/
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from log_keys import (
    NUM_ROUND_KEY,
    ACCURACY_KEY,
    LOSS_KEY,
    NUM_SAMPLES_KEY,
)

def load_data_from_file(log_path=None):
    if log_path is None:
        raise Exception('Cannot file log file from given path.')

    data = pd.read_csv(log_path)
    return data

def plot_accuracy_by_round_number(data, weighted=False, plot_every_n_rounds=20, plot_stds=False, figure_size=(10, 8), title_fontsize=16, **kwargs):
    # Initialize plot with figure_size.
    plt.figure(figsize=figure_size)

    # Initialize plot title, showing whether averaging over clients are weighted or not.
    title_weighted = 'Weighted' if weighted else 'Unweighted'
    plt.title(f'Accuracy vs Round Number ({title_weighted})', fontsize=title_fontsize)

    # Calculate accuracy's mean and std.
    if weighted:
        accuracies = data.groupby(NUM_ROUND_KEY).apply(_weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
        accuracies = accuracies.reset_index(name=ACCURACY_KEY)

        stds = data.groupby(NUM_ROUND_KEY).apply(_weighted_std, ACCURACY_KEY, NUM_SAMPLES_KEY)
        stds = stds.reset_index(name=ACCURACY_KEY)
    else:
        accuracies = data.groupby(NUM_ROUND_KEY, as_index=False).mean()
        stds = data.groupby(NUM_ROUND_KEY, as_index=False).std()
    
    # Filter out data by every n rounds.
    accuracies = accuracies.iloc[::plot_every_n_rounds]
    stds = stds.iloc[::plot_every_n_rounds]

    # Plot mean accuracy (and std if plot_stds is True),
    if plot_stds:
        plt.errorbar(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY], stds[ACCURACY_KEY])
    else:
        plt.plot(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY])
    
    # Calculate accuracy's 10/90th percentile,
    percentile_10 = data.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.1)
    percentile_90 = data.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.9)

    # Filter out data by every n rounds.
    percentile_10 = percentile_10.iloc[::plot_every_n_rounds]
    percentile_90 = percentile_90.iloc[::plot_every_n_rounds]

    # Plot accuracy's 10/90th percentile.
    plt.plot(percentile_10[NUM_ROUND_KEY], percentile_10[ACCURACY_KEY], linestyle=':')
    plt.plot(percentile_90[NUM_ROUND_KEY], percentile_90[ACCURACY_KEY], linestyle=':')

    # Draw legend.
    plt.legend(['Mean', '10th percentile', '90th percentile'], loc='upper left')

    # Draw x and y labels.
    plt.ylabel('Accuracy')
    plt.xlabel('Round Number')

    # Set extra properties such as xlim, ylim, xlabel, and ylabel.
    _set_plot_properties(kwargs)

    # Show the graph.
    plt.show()

def plot_loss_by_round_number(data, weighted=False, plot_stds=False, figsize=(10, 8), title_fontsize=16):
    raise NotImplementedError

def _set_plot_properties(properties):
    """Sets some plt properties."""
    if 'xlim' in properties:
        plt.xlim(properties['xlim'])
    if 'ylim' in properties:
        plt.ylim(properties['ylim'])
    if 'xlabel' in properties:
        plt.xlabel(properties['xlabel'])
    if 'ylabel' in properties:
        plt.ylabel(properties['ylabel'])

def _weighted_mean(df, metric_name, weight_name):
    d = df[metric_name]
    w = df[weight_name]
    try:
        return (w * d).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


def _weighted_std(df, metric_name, weight_name):
    d = df[metric_name]
    w = df[weight_name]
    try:
        weigthed_mean = (w * d).sum() / w.sum()
        return (w * ((d - weigthed_mean) ** 2)).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan