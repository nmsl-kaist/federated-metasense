import os
from typing import Dict

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.server.strategy import FedAvg

from utils.dataset import Dataset
from utils.logger import Logger
from utils.config import get_configuration
from utils.generate_model import generate_model

from clients.RayDefaultClient import RayDefaultClient
from clients.RayMetaClient import RayMetaClient

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Use minimal memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Start Ray simulation (a _default server_ will be created)
# This code does:
# 1. Load FEMNIST dataset (The dataset should be downloaded and sampled in advance.)
# 2. Starts a Ray-based simulation where a % of clients are sampled each round.
# 3. Every round, the global model is evaluated on each test client's data. 
#    This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    # Get configuration
    config = get_configuration()

    # Load the dataset
    femnist_dataset = Dataset(config['dataset_name'])

    # Initialize logger
    logger = Logger(config)

    # The number of requested clients must be smaller than dataset's number of client
    # If this assertion fails, try resample dataset with larger sampling ratio, or request smaller number of users.
    print(f"Number of dataset's training clients: {femnist_dataset.get_num_train_clients()}")
    print(f"Number of dataset's testing clients: {femnist_dataset.get_num_test_clients()}")
    print(f"Number of requested clients: {config['num_clients']}")
    assert config['num_clients'] < femnist_dataset.get_num_train_clients(), "Try resample dataset with larger sampling ratio, or request smaller number of users"
    assert config['num_clients'] < femnist_dataset.get_num_test_clients(), "Try resample dataset with larger sampling ratio, or request smaller number of users"

    # Initialize fit/eval config functions
    def fit_config(rnd: int) -> Dict[str, str]:
        """
        Return a fit configuration.
        The returned configuration is passed as an argument ''config'' in client's ''fit'' function.
        """
        config["fit_config"]["rnd"] = str(rnd)
        return config["fit_config"]

    def eval_config(rnd: int) -> Dict[str, str]:
        """
        Return an evaluation configuration.
        The returned configuration is passed as an argument ''config'' in client's ''evaluate'' function.
        """
        config["eval_config"]["rnd"] = str(rnd)
        return config["eval_config"]

    def client_fn(cid: str):
        """
        Create a single client instance.
        """
        # Create and return client
        if config["args_extra_label"] == 'meta':
            return RayMetaClient(cid, logger)
        else:
            return RayDefaultClient(cid, logger)

    # Initialize the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=config['fraction_fit'],
        fraction_eval=config['fraction_eval'],
        min_available_clients=config['num_clients'],  # All clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
    )

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config['num_clients'],
        client_resources=config['client_resources'],
        num_rounds=config['num_rounds'],
        strategy=strategy,
        ray_init_args=config['ray_config'],
    )
