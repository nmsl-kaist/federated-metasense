import flwr as fl
import tensorflow as tf

from utils.datasetloader import DatasetLoader
from models.generate_model import generate_model

class DefaultClient(fl.client.NumPyClient):
    def __init__(self, cid, logger, dataset_name):
        self.cid = cid
        self.logger = logger
        self.dataset_name = dataset_name
        self.model = generate_model(self.dataset_name)

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training samples."""
        print('fit - default')
        train_data = DatasetLoader(self.dataset_name, self.cid, 'train').get_data()
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=float(config["beta"])),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        self.model.set_weights(parameters)
        history = self.model.fit(
            x=train_data['x'], y=train_data['y'],
            batch_size=int(config["batch_size"]), epochs=int(config["meta_update_epochs"]), verbose=0)
        # For logging purpose, evaluate the model with full training data.
        loss, accuracy = self.model.evaluate(
            x=train_data['x'], y=train_data['y'], verbose=0)
        # Log the result of evaluation across full training data.
        self.logger.log_train_data(
            round_number=config['rnd'], cid=self.cid,
            num_samples=len(train_data['x']), loss=loss, accuracy=accuracy)
        return self.model.get_weights(), len(train_data['x']), {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        """This function is not called if centralized evaluation function is defined."""
        test_data = DatasetLoader(self.dataset_name, self.cid, 'test').get_data()
        self.model.set_weights(parameters)
        # Manual testing to not use moving statistics in batchnorm.
        logits = self.model(test_data['x'], training=True) # training=True to not use moving statistics in batchnorm
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            test_data['y'], logits, from_logits=True
        )
        loss = float(tf.reduce_mean(loss).numpy())
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state(test_data['y'], logits)
        accuracy = float(m.result().numpy())
        self.logger.log_test_data(
            round_number=config['rnd'], cid=self.cid,
            num_samples=len(test_data['x']), loss=loss, accuracy=accuracy)
        return loss, len(test_data['x']), {"accuracy": accuracy}
