import os
import time
import sys
from pathlib import Path
from multiprocessing import Process
from typing import Tuple

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.server.strategy import FedAvg

import data_utils

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Use minimal memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# loaded dataset
femnist_dataset = None

# image size
IMAGE_SIZE = 28
NUM_CLASSES = 62

# dataset class
class Dataset():
    def __init__(self, train_path, test_path):
        train_client_ids, test_client_ids, train_data_dict, test_data_dict = data_utils.read_data(train_path, test_path)
        train_data = map(lambda id: train_data_dict.get(id), train_client_ids)
        test_data = map(lambda id: test_data_dict.get(id), test_client_ids)
        self.train_data = list(map(lambda data: {k:np.array(v) for k, v in data.items()}, train_data))
        self.test_data = list(map(lambda data: {k:np.array(v) for k, v in data.items()}, test_data))

    def get_train_data(self, cid: str):
        return self.train_data[int(cid)]

    def get_test_data(self, cid: str):
        return self.test_data[int(cid)]

    def get_num_train_clients(self):
        return len(self.train_data)

    def get_num_test_clients(self):
        return len(self.test_data)

# CNN model generator using sequential
def generate_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1), input_shape=(IMAGE_SIZE * IMAGE_SIZE,)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
    model.add(tf.keras.layers.Dense(units=NUM_CLASSES))
    return model

# Define a Flower client
class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        # Load and compile a Keras model
        self.model = generate_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training
        examples."""
        self.model.set_weights(parameters)
        # Remove steps_per_epoch if you want to train over the full dataset
        # https://keras.io/api/models/model_training_apis/#fit-method
        train_data = femnist_dataset.get_train_data(self.cid)
        self.model.fit(train_data['x'], train_data['y'], epochs=1, batch_size=10)
        return self.model.get_weights(), len(train_data['x']), {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        self.model.set_weights(parameters)
        test_data = femnist_dataset.get_test_data(self.cid)
        loss, accuracy = self.model.evaluate(test_data['x'], test_data['y'])
        return loss, len(test_data['x']), {"accuracy": accuracy}


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a Ray-based simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    num_clients = 100  # number of total clients
    fraction_fit = 0.8 # {fraction_fit * num_clients} clients are used for training. Only number matters since dataset are split.
    fraction_eval = 0.4 # {fraction_eval * num_clients} clients are used for testing. Only number matters since dataset are split.
    client_resources = {"num_gpus": 1/4} # 1/n means n clients are assigned to 1 physical gpu. Too large n may cause gpu oom error.

    # load the dataset
    train_path = os.path.join('..', 'data', 'femnist', 'data', 'train')
    test_path = os.path.join('..', 'data', 'femnist', 'data', 'test')
    femnist_dataset = Dataset(train_path, test_path)

    # The number of clients must be smaller than dataset's number of client
    # If this assertion fails, try resample dataset with larger sampling ratio.
    print(f"Number of dataset's training clients: {femnist_dataset.get_num_train_clients()}")
    print(f"Number of dataset's testing clients: {femnist_dataset.get_num_test_clients()}")
    print(f"Number of requested clients: {num_clients}")
    assert num_clients < femnist_dataset.get_num_train_clients(), "Try resample dataset with larger sampling ratio"
    assert num_clients < femnist_dataset.get_num_test_clients(), "Try resample dataset with larger sampling ratio"

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_eval=fraction_eval,
        min_available_clients=num_clients,  # All clients should be available
    )

    def client_fn(cid: str):
        # create a single client instance
        return CifarRayClient(cid)

    # (optional) specify ray config
    ray_config = {"include_dashboard": True}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources=client_resources,
        num_rounds=3000,
        strategy=strategy,
        ray_init_args=ray_config,
    )
