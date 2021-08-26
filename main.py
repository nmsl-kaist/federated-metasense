import os
import time
import sys
from pathlib import Path
from multiprocessing import Process
from typing import Tuple, Dict

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.server.strategy import FedAvg

from utils.dataset import Dataset
from utils.logger import Logger
from utils.generate_model import generate_model

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Use minimal memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Define a Flower client
class RayClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.model = generate_model()

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training samples."""
        print(f"Round: {config['rnd']} - fit")
        self.model.set_weights(parameters)
        train_data = femnist_dataset.get_train_data(self.cid)
        history = self.model.fit(
            x=train_data['x'],
            y=train_data['y'],
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
            verbose=2)
        logger.log_train_data(
            round_number=config['rnd'],
            cid=self.cid,
            num_samples=len(train_data['x']),
            history=history)
        return self.model.get_weights(), len(train_data['x']), {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        print(f"Round: {config['rnd']} - evaluate")
        self.model.set_weights(parameters)
        test_data = femnist_dataset.get_test_data(self.cid)
        loss, accuracy = self.model.evaluate(
            x=test_data['x'],
            y=test_data['y'],
            verbose=2)
        logger.log_test_data(
            round_number=config['rnd'],
            cid=self.cid,
            num_samples=len(test_data['x']),
            loss=loss,
            accuracy=accuracy)
        return loss, len(test_data['x']), {"accuracy": accuracy}


def fit_config(rnd: int) -> Dict[str, str]:
    """
    Return a fit configuration.
    The returned configuration is passed as an argument ''config'' in client's ''fit'' function.
    """
    config = {
        "rnd": str(rnd),
        "epochs": str(1),
        "batch_size": str(10),
    }
    return config

def eval_config(rnd: int) -> Dict[str, str]:
    """
    Return an evaluation configuration.
    The returned configuration is passed as an argument ''config'' in client's ''evaluate'' function.
    """
    config = {
        "rnd": str(rnd),
    }
    return config

def client_fn(cid: str):
    """
    Create a single client instance.
    """
    return RayClient(cid)


# Start Ray simulation (a _default server_ will be created)
# This code does:
# 1. Load FEMNIST dataset (The dataset should be downloaded and sampled in advance.)
# 2. Starts a Ray-based simulation where a % of clients are sampled each round.
# 3. Every round, the global model is evaluated on each test client's data. 
#    This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    # configuration
    config = {
        "name": "test_experiment_2", # This field should exist.
        "num_clients": 100, # number of total clients
        "fraction_fit": 0.1, # {fraction_fit * num_clients} clients are used for training. Only number matters since dataset are split.
        "fraction_eval": 0.1, # {fraction_eval * num_clients} clients are used for testing. Only number matters since dataset are split.
        "client_resources": {"num_gpus": 1/4}, # 1/n means n clients are assigned to 1 physical gpu. Too large n may cause gpu oom error.
        "dataset": "femnist",
        "num_rounds": 100,
        "ray_config": {"include_dashboard": True},
    }

    # Load the dataset
    femnist_dataset = Dataset(config['dataset'])

    # Initialize logger
    logger = Logger(config)

    # The number of requested clients must be smaller than dataset's number of client
    # If this assertion fails, try resample dataset with larger sampling ratio, or request smaller number of users.
    print(f"Number of dataset's training clients: {femnist_dataset.get_num_train_clients()}")
    print(f"Number of dataset's testing clients: {femnist_dataset.get_num_test_clients()}")
    print(f"Number of requested clients: {config['num_clients']}")
    assert config['num_clients'] < femnist_dataset.get_num_train_clients(), "Try resample dataset with larger sampling ratio, or request smaller number of users"
    assert config['num_clients'] < femnist_dataset.get_num_test_clients(), "Try resample dataset with larger sampling ratio, or request smaller number of users"

    # initialize the strategy
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
