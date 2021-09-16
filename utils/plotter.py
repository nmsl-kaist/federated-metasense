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

from utils.log_keys import (
    NUM_ROUND_KEY,
    ACCURACY_KEY,
    LOSS_KEY,
    NUM_SAMPLES_KEY,
)

class Plotter():
    def __init__(self, log_path=None):
        if log_path is None:
            raise Exception('Cannot find log file from given path.')

        self.data = pd.read_csv(log_path)

    def final_accuracy(self, num_total_clients):
        validation_set = self.data.loc[self.data['cid'] < num_total_clients // 2]
        test_set = self.data.loc[self.data['cid'] >= num_total_clients // 2]

        validation_grouped = validation_set.groupby(NUM_ROUND_KEY, as_index=False).mean()
        test_grouped = test_set.groupby(NUM_ROUND_KEY, as_index=False).mean()

        return test_grouped[validation_grouped['loss'] == validation_grouped['loss'].min()]

    def plot_accuracy_distribution_from_round(self, round_number, figsize=(10, 8), color='gray', sort=False, **kwargs):
        plt.figure(figsize=figsize)
        plt.title(f'Accuracy from round {round_number}')

        data = self.data.loc[self.data[NUM_ROUND_KEY] == round_number]
        if sort:
            data = data.sort_values(by=ACCURACY_KEY)
            data = data.reset_index(drop=True)
        print(data)

        plt.bar(data.index if sort else data['cid'], data[ACCURACY_KEY], color=color)

        plt.ylabel('Accuracy')
        plt.xlabel('cid')

        self._set_plot_properties(kwargs)

        plt.show()

    def plot_accuracy_by_round_number(self, weighted=False, plot_every_n_rounds=10, plot_stds=False, figsize=(10, 8), title_fontsize=16, **kwargs):
        # Initialize plot with figure_size.
        plt.figure(figsize=figsize)

        # Initialize plot title, showing whether averaging over clients are weighted or not.
        title_weighted = 'Weighted' if weighted else 'Unweighted'
        plt.title(f'Accuracy vs Round Number ({title_weighted})', fontsize=title_fontsize)

        # Calculate accuracy's mean and std.
        if weighted:
            accuracies = self.data.groupby(NUM_ROUND_KEY).apply(self._weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY)
            accuracies = accuracies.reset_index(name=ACCURACY_KEY)

            stds = self.data.groupby(NUM_ROUND_KEY).apply(self._weighted_std, ACCURACY_KEY, NUM_SAMPLES_KEY)
            stds = stds.reset_index(name=ACCURACY_KEY)
        else:
            accuracies = self.data.groupby(NUM_ROUND_KEY, as_index=False).mean()
            stds = self.data.groupby(NUM_ROUND_KEY, as_index=False).std()

        # Filter out data by every n rounds.
        accuracies = accuracies.iloc[::plot_every_n_rounds]
        stds = stds.iloc[::plot_every_n_rounds]

        # Plot mean accuracy (and std if plot_stds is True),
        if plot_stds:
            plt.errorbar(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY], stds[ACCURACY_KEY])
        else:
            plt.plot(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY])

        # Calculate accuracy's 10/90th percentile,
        percentile_10 = self.data.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.1)
        percentile_90 = self.data.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.9)

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
        self._set_plot_properties(kwargs)

        # Show the graph.
        plt.show()

        print(accuracies)
        print(stds)

    def plot_loss_by_round_number(self, weighted=False, plot_every_n_rounds=10, plot_stds=False, figsize=(10, 8), title_fontsize=16, **kwargs):
        # Initialize plot with figure_size.
        plt.figure(figsize=figsize)

        # Initialize plot title, showing whether averaging over clients are weighted or not.
        title_weighted = 'Weighted' if weighted else 'Unweighted'
        plt.title(f'Loss vs Round Number ({title_weighted})', fontsize=title_fontsize)

        # Calculate loss's mean and std.
        if weighted:
            losses = self.data.groupby(NUM_ROUND_KEY).apply(self._weighted_mean, LOSS_KEY, NUM_SAMPLES_KEY)
            losses = losses.reset_index(name=LOSS_KEY)

            stds = self.data.groupby(NUM_ROUND_KEY).apply(self._weighted_std, LOSS_KEY, NUM_SAMPLES_KEY)
            stds = stds.reset_index(name=LOSS_KEY)
        else:
            losses = self.data.groupby(NUM_ROUND_KEY, as_index=False).mean()
            stds = self.data.groupby(NUM_ROUND_KEY, as_index=False).std()

        # Filter out data by every n rounds.
        losses = losses.iloc[::plot_every_n_rounds]
        stds = stds.iloc[::plot_every_n_rounds]

        # Plot mean loss (and std if plot_stds is True),
        if plot_stds:
            plt.errorbar(losses[NUM_ROUND_KEY], losses[LOSS_KEY], stds[LOSS_KEY])
        else:
            plt.plot(losses[NUM_ROUND_KEY], losses[LOSS_KEY])

        # Calculate accuracy's 10/90th percentile,
        percentile_10 = self.data.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.1)
        percentile_90 = self.data.groupby(NUM_ROUND_KEY, as_index=False).quantile(0.9)

        # Filter out data by every n rounds.
        percentile_10 = percentile_10.iloc[::plot_every_n_rounds]
        percentile_90 = percentile_90.iloc[::plot_every_n_rounds]

        # Plot accuracy's 10/90th percentile.
        plt.plot(percentile_10[NUM_ROUND_KEY], percentile_10[LOSS_KEY], linestyle=':')
        plt.plot(percentile_90[NUM_ROUND_KEY], percentile_90[LOSS_KEY], linestyle=':')

        # Draw legend.
        plt.legend(['Mean', '10th percentile', '90th percentile'], loc='upper left')

        # Draw x and y labels.
        plt.ylabel('Loss')
        plt.xlabel('Round Number')

        # Set extra properties such as xlim, ylim, xlabel, and ylabel.
        self._set_plot_properties(kwargs)

        # Show the graph.
        plt.show()

    def _set_plot_properties(self, properties):
        """Sets some plt properties."""
        if 'xlim' in properties:
            plt.xlim(properties['xlim'])
        if 'ylim' in properties:
            plt.ylim(properties['ylim'])
        if 'xlabel' in properties:
            plt.xlabel(properties['xlabel'])
        if 'ylabel' in properties:
            plt.ylabel(properties['ylabel'])

    def _weighted_mean(self, df, metric_name, weight_name):
        d = df[metric_name]
        w = df[weight_name]
        try:
            return (w * d).sum() / w.sum()
        except ZeroDivisionError:
            return np.nan

    def _weighted_std(self, df, metric_name, weight_name):
        d = df[metric_name]
        w = df[weight_name]
        try:
            weigthed_mean = (w * d).sum() / w.sum()
            return (w * ((d - weigthed_mean) ** 2)).sum() / w.sum()
        except ZeroDivisionError:
            return np.nan