import flwr as fl
import tensorflow as tf
import numpy as np

from utils.datasetloader import DatasetLoader
from utils.generate_model_femnist import generate_model_femnist
from utils.generate_model_femnist import NUM_CLASSES

# Define a Flower client - fedmeta
class SpecialClientWithSeparationFixedClass(fl.client.NumPyClient):
    def __init__(self, cid: str, logger):
        self.cid = cid
        self.logger = logger
        self.model = generate_model_femnist()

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
        try:
            if config["mode"] == "per-cond":
                return self.fit_per_cond(parameters, config)
            elif config["mode"] == "multi-cond-learn":
                return self.fit_multi_cond_learn(parameters, config)
            else:
                return self.fit_multi_cond_update(parameters, config)
        except Exception as e:
            print(e)
    
    def fit_per_cond(self, parameters, config):
        """Fit model and return new weights."""
        print('fit - special - per-cond')
        train_data = DatasetLoader(self.cid, 'train').get_data()
        # Pick k1, k2 random samples respectively for meta-learn and meta-update steps.
        train_data_meta_learn, train_data_meta_update = self._pick_two_chunks_of_random_k_samples(train_data, k1=62, k2=train_data['x'].shape[0]-62)
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

    def filter_dataset_by_class_range(self, data, this_class_range, max_class_range):
        class_min = NUM_CLASSES * this_class_range / max_class_range
        class_max = NUM_CLASSES * (this_class_range + 1) / max_class_range
        indices = np.argwhere(np.logical_and(data['y'] < class_max, data['y'] >= class_min)).flatten()
        return {'x': data['x'][indices], 'y': data['y'][indices]}

    def fit_multi_cond_learn(self, parameters, config):
        print('fit - special - multi-cond-learn')
        train_data = DatasetLoader(self.cid, 'train').get_data()
        train_data_filtered_by_class = self.filter_dataset_by_class_range(train_data, int(config["this-class-range"]), int(config["max-class-range"]))
        # Pick k random samples for meta-learn step. Use only first 50% of data.
        size = int(train_data_filtered_by_class['y'].shape[0] * 0.5)
        train_data_filtered_by_class = {'x': train_data_filtered_by_class['x'][:size], 'y': train_data_filtered_by_class['y'][:size]}
        k = min(62, train_data_filtered_by_class['x'].shape[0])
        train_data_meta_learn, _ = self._pick_two_chunks_of_random_k_samples(train_data_filtered_by_class, k1=k, k2=0)
        # Recompile the model with meta-learn learning rate (alpha).
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["alpha"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        # Load the parameters.
        self.model.set_weights(parameters)
        # Perform meta-learn
        if train_data_meta_learn['x'].shape[0] > 0:
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
        return theta_prime, 1, {}

    def fit_multi_cond_update(self, parameters, config):
        print('fit - special - multi-cond-update')
        train_data = DatasetLoader(self.cid, 'train').get_data()
        train_data_filtered_by_class = self.filter_dataset_by_class_range(train_data, int(config["this-class-range"]), int(config["max-class-range"]))
        # Pick k random samples for meta-update step. Use only last 50% of data.
        size = int(train_data_filtered_by_class['y'].shape[0] * 0.5)
        train_data_filtered_by_class = {'x': train_data_filtered_by_class['x'][size:], 'y': train_data_filtered_by_class['y'][size:]}
        k = min(1000, train_data_filtered_by_class['x'].shape[0])
        train_data_meta_update, _ = self._pick_two_chunks_of_random_k_samples(train_data_filtered_by_class, k1=k, k2=0)
        # Evaluate the user(task) specific parameters with meta-update samples.
        # Recompile the model with meta-update learning rate (beta).
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["beta"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        # Load the parameters.
        self.model.set_weights(parameters)
        # Perform meta-update
        if train_data_meta_update['x'].shape[0] > 0:
            self.model.fit(
                x=train_data_meta_update['x'], y=train_data_meta_update['y'],
                batch_size=int(config["batch_size"]),
                epochs=int(config["meta_update_epochs"]),
                verbose=0)
        updated_weight = self.model.get_weights()
        # No logging here.
        # Return the result
        return updated_weight, 1, {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        test_data = DatasetLoader(self.cid, 'test').get_data()
        # Pick k random samples for adaptation.
        data_adaptation, data_evaluation = self._pick_two_chunks_of_random_k_samples(test_data, k1=62, k2=test_data['x'].shape[0]-62)
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
            x=data_evaluation['x'], y=data_evaluation['y'], verbose=0)
        self.logger.log_test_data(
            round_number=config['rnd'], cid=self.cid,
            num_samples=len(data_evaluation['x']), loss=loss, accuracy=accuracy)
        return loss, len(test_data['x']), {"accuracy": accuracy}
