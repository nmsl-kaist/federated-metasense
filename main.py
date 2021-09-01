import os
import time
import sys
import argparse
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

# Define a Flower client - fedmeta
class RayMetaClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.model = generate_model()

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def _add_parameters(self, parameter1, parameter2):
        return [p1 + p2 for p1, p2 in zip(parameter1, parameter2)]

    def _subtract_parameters(self, parameter1, parameter2):
        return [p1 - p2 for p1, p2 in zip(parameter1, parameter2)]

    def _pick_two_chunks_of_random_k_samples(self, data, k1, k2):
        indices = np.random.choice(data['x'].shape[0], size=k1+k2, replace=False)
        indices_for_chunk1 = indices[:k1]
        indices_for_chunk2 = indices[k1:]
        chunk1 = {'x': data['x'][indices_for_chunk1], 'y': data['y'][indices_for_chunk1]}
        chunk2 = {'x': data['x'][indices_for_chunk2], 'y': data['y'][indices_for_chunk2]}
        return chunk1, chunk2

    def fit(self, parameters, config):
        """Fit model and return new weights."""
        print(f"Round: {config['rnd']} - fit")
        print(config['alpha'], config['meta_learn_epochs'])
        # Get training data.
        train_data = femnist_dataset.get_train_data(self.cid)
        # Pick k1, k2 random samples respectively for meta-learn and meta-update steps.
        train_data_meta_learn, train_data_meta_update = self._pick_two_chunks_of_random_k_samples(train_data, k1=10, k2=20)
        train_data_meta_update = train_data
        # Recompile the model with meta-learn learning rate (alpha).
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["alpha"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
        # Load the parameters.
        self.model.set_weights(parameters)
        # First, get user(task) specific parameters theta_prime with a few gradient descent with meta-learn samples.
        self.model.fit(
            x=train_data_meta_learn['x'],
            y=train_data_meta_learn['y'],
            batch_size=train_data_meta_learn['x'].shape[0], # Full batch
            epochs=int(config["meta_learn_epochs"]), # gradient steps
            verbose=0)
        theta_prime = self.model.get_weights()
        # For logging purpose, evaluate the model with theta_prime with training data.
        loss, accuracy = self.model.evaluate(
            x=train_data['x'],
            y=train_data['y'],
            verbose=0)
        # Log the evaluation result with training data.
        logger.log_train_data(
            round_number=config['rnd'],
            cid=self.cid,
            num_samples=len(train_data['x']),
            loss=loss,
            accuracy=accuracy)
        # Second, evaluate the user(task) specific parameters with meta-update samples, and get the gradient from user(task) specific parameters.
        # Recompile the model with meta-update learning rate (beta).
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["beta"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
        self.model.fit(
            x=train_data_meta_update['x'],
            y=train_data_meta_update['y'],
            batch_size=int(config["batch_size"]),
            epochs=int(config["meta_update_epochs"]),
            verbose=0)
        updated_weight = self.model.get_weights()
        # Final gradient is updated_weight - theta_prime.
        gradient = self._subtract_parameters(updated_weight, theta_prime)
        # Finally, return the parameter to the server. They will be averaged by server.
        # 1 for sample size means unweighted average among clients.
        return self._add_parameters(parameters, gradient), 1, {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        print(f"Round: {config['rnd']} - evaluate")
        # Get testing data.
        test_data = femnist_dataset.get_test_data(self.cid)
        # Pick k random samples for adaptation.
        test_data_adaptation, _ = self._pick_two_chunks_of_random_k_samples(test_data, k1=10, k2=0)
        # Recompile the model with adaptation rate and load the parameters.
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["alpha"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
        self.model.set_weights(parameters)
        # Adapt the model using samples.
        self.model.fit(
            x=test_data_adaptation['x'],
            y=test_data_adaptation['y'],
            batch_size=test_data_adaptation['x'].shape[0], # Full batch
            epochs=int(config["adaptation_epochs"]), # gradient steps
            verbose=0)
        # Evaluate the adapted model using full test dataset.
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


# Start Ray simulation (a _default server_ will be created)
# This code does:
# 1. Load FEMNIST dataset (The dataset should be downloaded and sampled in advance.)
# 2. Starts a Ray-based simulation where a % of clients are sampled each round.
# 3. Every round, the global model is evaluated on each test client's data. 
#    This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-alpha',
                    help='learning rate alpha;',
                    type=float,
                    required=True)
    parser.add_argument('--meta-learn-epochs',
                    help='number of epochs for meta-learn and adaptation;',
                    type=int,
                    required=True)
    args = parser.parse_args()

    # configuration
    config = {
        "name": f"test_experiment_meta_{args.alpha}_{args.meta_learn_epochs}epochs", # This field should exist.
        "num_clients": 100, # number of total clients
        "fraction_fit": 0.1, # {fraction_fit * num_clients} clients are used for training. Only number matters since dataset are split.
        "fraction_eval": 0.1, # {fraction_eval * num_clients} clients are used for testing. Only number matters since dataset are split.
        "client_resources": {"num_gpus": 1/4}, # 1/n means n clients are assigned to 1 physical gpu. Too large n may cause gpu oom error.
        "dataset": "femnist",
        "num_rounds": 1000,
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

    # Initialize fit/eval config functions
    def fit_config(rnd: int) -> Dict[str, str]:
        """
        Return a fit configuration.
        The returned configuration is passed as an argument ''config'' in client's ''fit'' function.
        """
        config = {
            "rnd": str(rnd),
            "meta_learn_epochs": str(args.meta_learn_epochs),
            "meta_update_epochs": str(1),
            "batch_size": str(10),
            "alpha": str(args.alpha),
            "beta": str(0.010),
        }
        return config

    def eval_config(rnd: int) -> Dict[str, str]:
        """
        Return an evaluation configuration.
        The returned configuration is passed as an argument ''config'' in client's ''evaluate'' function.
        """
        config = {
            "rnd": str(rnd),
            "adaptation_epochs": str(args.meta_learn_epochs),
            "alpha": str(args.alpha),
        }
        return config

    def client_fn(cid: str):
        """
        Create a single client instance.
        """
        return RayMetaClient(cid)

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
