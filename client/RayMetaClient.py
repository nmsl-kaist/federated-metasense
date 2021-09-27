import flwr as fl
import tensorflow as tf
import numpy as np

from utils.generate_model import generate_model

# Define a Flower client - fedmeta
class RayMetaClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset, logger):
        self.cid = cid
        self.dataset = dataset
        self.logger = logger
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
        print('fit - meta')
        train_data = self.dataset.get_train_data(self.cid)
        # Pick k1, k2 random samples respectively for meta-learn and meta-update steps.
        train_data_meta_learn, train_data_meta_update = self._pick_two_chunks_of_random_k_samples(train_data, k1=10, k2=train_data['x'].shape[0]-20)
        # Recompile the model with meta-learn learning rate (alpha).
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["alpha"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        # Load the parameters.
        self.model.set_weights(parameters)
        # First, get user(task) specific parameters theta_prime with a few gradient descent with meta-learn samples.
        self.model.fit(
            x=train_data_meta_learn['x'], y=train_data_meta_learn['y'],
            batch_size=train_data_meta_learn['x'].shape[0], # Full batch
            epochs=int(config["meta_learn_epochs"]), # gradient steps
            verbose=0)
        theta_prime = self.model.get_weights()
        # For logging purpose, evaluate the model with theta_prime across full training data.
        loss, accuracy = self.model.evaluate(
            x=train_data['x'], y=train_data['y'], verbose=0)
        # Log the result of evaluation across full training data.
        self.logger.log_train_data(
            round_number=config['rnd'], cid=self.cid,
            num_samples=len(train_data['x']), loss=loss, accuracy=accuracy)
        # Second, evaluate the user(task) specific parameters with meta-update samples, and get the gradient from user(task) specific parameters.
        # Recompile the model with meta-update learning rate (beta).
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["beta"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        self.model.fit(
            x=train_data_meta_update['x'], y=train_data_meta_update['y'],
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
        """This function is not called if centralized evaluation function is defined."""
        test_data = self.dataset.get_test_data(self.cid)
        # Pick k random samples for adaptation.
        data_adaptation, _ = self._pick_two_chunks_of_random_k_samples(test_data, k1=10, k2=0)
        # Recompile the model with adaptation rate and load the parameters.
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["alpha"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        self.model.set_weights(parameters)
        # Adapt the model using random samples.
        self.model.fit(
            x=data_adaptation['x'], y=data_adaptation['y'],
            batch_size=data_adaptation['x'].shape[0], # Full batch
            epochs=int(config["adaptation_epochs"]), # gradient steps
            verbose=0)
        # Evaluate the adapted model using full test dataset.
        loss, accuracy = self.model.evaluate(
            x=test_data['x'], y=test_data['y'], verbose=0)
        self.logger.log_test_data(
            round_number=config['rnd'], cid=self.cid,
            num_samples=len(test_data['x']), loss=loss, accuracy=accuracy)
        return loss, len(test_data['x']), {"accuracy": accuracy}
