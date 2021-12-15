import flwr as fl
import tensorflow as tf
import numpy as np
import math

from utils.datasetloader import DatasetLoader
from models.generate_model import generate_model

# Define a Flower client - fedmeta
class SpecialClientWithSeparation(fl.client.NumPyClient):
    def __init__(self, cid, logger, dataset_name):
        self.cid = cid
        self.logger = logger
        self.dataset_name = dataset_name
        self.model = generate_model(self.dataset_name)

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def _add_parameters(self, parameter1, parameter2):
        return [p1 + p2 for p1, p2 in zip(parameter1, parameter2)]

    def _subtract_parameters(self, parameter1, parameter2):
        return [p1 - p2 for p1, p2 in zip(parameter1, parameter2)]

    def _sample_two_chunks_k1_k2_datapoints(self, data, k1, k2):
        indices = np.random.choice(data['x'].shape[0], size=k1+k2, replace=False)
        indices_for_chunk1 = indices[:k1]
        indices_for_chunk2 = indices[k1:]
        chunk1 = {'x': data['x'][indices_for_chunk1], 'y': data['y'][indices_for_chunk1]}
        chunk2 = {'x': data['x'][indices_for_chunk2], 'y': data['y'][indices_for_chunk2]}
        return chunk1, chunk2

    def _split_into_k_shots_and_remainder(self, data, k):
        # Return one chunk with k shots and another chunk with remaining data.
        available_classes = list(set(data['y'].tolist()))
        shot_indices = []
        for i in available_classes:
            indices_in_class = []
            for j in range(len(data['y'])):
                if data['y'][j] == i:
                    indices_in_class.append(j)
            k_shot_indices_in_class = np.random.choice(indices_in_class, size=k, replace=False)
            shot_indices.extend(k_shot_indices_in_class.tolist())
        shot_indices = np.array(shot_indices)
        remaining_indices = np.setdiff1d(np.arange(len(data['y'])), shot_indices)
        chunk1 = {'x': data['x'][shot_indices], 'y': data['y'][shot_indices]}
        chunk2 = {'x': data['x'][remaining_indices], 'y': data['y'][remaining_indices]}
        return chunk1, chunk2

    def _sample_two_chunks_k1_k2_shots(self, data, k1, k2):
        chunk1, remaining = self._split_into_k_shots_and_remainder(data, k1)
        chunk2, _ = self._split_into_k_shots_and_remainder(remaining, k2)
        return chunk1, chunk2

    def _split_into_two_chunks_fit(self, data):
        if self.dataset_name == "FEMNIST":
            k = min(62, data['x'].shape[0])
            return self._sample_two_chunks_k1_k2_datapoints(data, k1=k, k2=data['x'].shape[0]-k)
        else:
            return self._sample_two_chunks_k1_k2_shots(data, k1=3, k2=3)

    def _split_into_two_chunks_evaluate(self, data):
        if self.dataset_name == "FEMNIST":
            return self._sample_two_chunks_k1_k2_datapoints(data, k1=62, k2=data['x'].shape[0]-62)
        else:
            return self._split_into_k_shots_and_remainder(data, k=3)

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
        train_data = DatasetLoader(self.dataset_name, self.cid, 'train').get_data()
        optimizer_outer = tf.keras.optimizers.Adam(learning_rate=float(config["beta"]))
        lr_inner = float(config["alpha"])
        self.model.set_weights(parameters)
        for _ in range(10):
            # Split into two chunks for meta-learn and meta-update steps.
            train_data_meta_learn, train_data_meta_update = self._split_into_two_chunks_fit(train_data)
            with tf.GradientTape() as t_outer:
                # Take one gradient step
                with tf.GradientTape() as t_inner:
                    logits = self.model(train_data_meta_learn['x'], training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        train_data_meta_learn['y'], logits, from_logits=True
                    )
                grad = t_inner.gradient(loss, self.model.trainable_variables)
                # Copy model and calculate theta_prime
                model_copy = generate_model(self.dataset_name)
                k = 0
                for i in range(len(model_copy.layers)):
                    if model_copy.layers[i].name.startswith('conv') or model_copy.layers[i].name.startswith('dense'):
                        model_copy.layers[i].kernel = tf.subtract(self.model.layers[i].kernel, tf.multiply(lr_inner, grad[k]))
                        model_copy.layers[i].bias = tf.subtract(self.model.layers[i].bias, tf.multiply(lr_inner, grad[k + 1]))
                        k += 2
                    if model_copy.layers[i].name.startswith('batch'):
                        model_copy.layers[i].gamma = tf.subtract(self.model.layers[i].gamma, tf.multiply(lr_inner, grad[k]))
                        model_copy.layers[i].beta = tf.subtract(self.model.layers[i].beta, tf.multiply(lr_inner, grad[k + 1]))
                        k += 2
                # Calculate meta loss
                logits = model_copy(train_data_meta_update['x'], training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        train_data_meta_update['y'], logits, from_logits=True
                    )
                )
            grad = t_outer.gradient(loss, self.model.trainable_variables)
            optimizer_outer.apply_gradients(zip(grad, self.model.trainable_variables))

        return self.model.get_weights(), 1, {}

    def filter_dataset_by_class_list(self, data, class_list):
        indices = np.argwhere(np.isin(data['y'], class_list)).flatten()
        return {'x': data['x'][indices], 'y': data['y'][indices]}

    def fit_multi_cond_learn(self, parameters, config):
        print('fit - special - multi-cond-learn')
        train_data = DatasetLoader(self.dataset_name, self.cid, 'train').get_data()
        train_data_filtered_by_class = self.filter_dataset_by_class_list(train_data, config["selected-class"])
        # Pick k random samples for meta-learn step. Use only first 50% of data.
        #size = int(train_data_filtered_by_class['y'].shape[0] * 0.5)
        #train_data_filtered_by_class = {'x': train_data_filtered_by_class['x'][:size], 'y': train_data_filtered_by_class['y'][:size]}
        optimizer_inner = tf.keras.optimizers.SGD(learning_rate=float(config["alpha"]))
        self.model.set_weights(parameters)
        train_data_meta_learn, _ = self._split_into_two_chunks_fit(train_data_filtered_by_class)
        for _ in range(1):
            # Split into two chunks for meta-learn and meta-update steps.
            # Take one gradient step
            with tf.GradientTape() as t_inner:
                logits = self.model(train_data_meta_learn['x'], training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    train_data_meta_learn['y'], logits, from_logits=True
                )
            grad = t_inner.gradient(loss, self.model.trainable_variables)
            optimizer_inner.apply_gradients(zip(grad, self.model.trainable_variables))

        return self.model.get_weights(), 1, {}

    def fit_multi_cond_update(self, parameters, config):
        print('fit - special - multi-cond-update')
        train_data = DatasetLoader(self.dataset_name, self.cid, 'train').get_data()
        train_data_filtered_by_class = self.filter_dataset_by_class_list(train_data, config["selected-class"])
        # Pick k random samples for meta-update step. Use only last 50% of data.
        #size = int(train_data_filtered_by_class['y'].shape[0] * 0.5)
        #train_data_filtered_by_class = {'x': train_data_filtered_by_class['x'][size:], 'y': train_data_filtered_by_class['y'][size:]}
        optimizer_inner = tf.keras.optimizers.Adam(learning_rate=float(config["beta"]))
        self.model.set_weights(parameters)
        _, train_data_meta_update = self._split_into_two_chunks_fit(train_data_filtered_by_class)
        for _ in range(10):
            # Split into two chunks for meta-learn and meta-update steps.
            # Take one gradient step
            with tf.GradientTape() as t_inner:
                logits = self.model(train_data_meta_update['x'], training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    train_data_meta_update['y'], logits, from_logits=True
                )
            grad = t_inner.gradient(loss, self.model.trainable_variables)
            optimizer_inner.apply_gradients(zip(grad, self.model.trainable_variables))

        return self.model.get_weights(), 1, {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        """This function is not called if centralized evaluation function is defined."""
        test_data = DatasetLoader(self.dataset_name, self.cid, 'test').get_data()
        optimizer_inner = tf.keras.optimizers.SGD(learning_rate=float(config["alpha"]))
        self.model.set_weights(parameters)
        # Pick k random samples for adaptation.
        data_adaptation, data_evaluation = self._split_into_two_chunks_evaluate(test_data)
        # Manual testing to not use moving statistics in batchnorm.
        # Copy model.
        model_copy = generate_model(self.dataset_name)
        model_copy.set_weights(self.model.get_weights())
        # Take a few gradient step
        for _ in range(5):
            with tf.GradientTape() as t_inner:
                logits = model_copy(data_adaptation['x'], training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    data_adaptation['y'], logits, from_logits=True
                )
            grad = t_inner.gradient(loss, model_copy.trainable_variables)
            optimizer_inner.apply_gradients(zip(grad, model_copy.trainable_variables))
        # Evaluate the adapted model
        logits = model_copy(data_evaluation['x'], training=True) # training=True to not use moving statistics in batchnorm
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            data_evaluation['y'], logits, from_logits=True
        )
        loss = float(tf.reduce_mean(loss).numpy())
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state(data_evaluation['y'], logits)
        accuracy = float(m.result().numpy())
        self.logger.log_test_data(
            round_number=config['rnd'], cid=self.cid,
            num_samples=len(data_evaluation['x']), loss=loss, accuracy=accuracy)
        return loss, len(data_evaluation['x']), {"accuracy": accuracy}

