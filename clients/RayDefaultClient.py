import flwr as fl

from utils.generate_model import generate_model

# Define a Flower client
class RayDefaultClient(fl.client.NumPyClient):
    def __init__(self, cid: str, logger):
        self.cid = cid
        self.model = generate_model()
        self.logger = logger

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training samples."""
        train_data = femnist_dataset.get_train_data(self.cid)
        self.model.set_weights(parameters)
        history = self.model.fit(
            x=train_data['x'], y=train_data['y'],
            batch_size=int(config["batch_size"]), epochs=1, verbose=0)
        self.logger.log_train_data(
            round_number=config['rnd'], cid=self.cid,
            num_samples=len(self.x_train), history=history)
        return self.model.get_weights(), len(train_data['x']), {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        """This function is not called if centralized evaluation function is defined."""
        test_data = femnist_dataset.get_test_data(self.cid)
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(
            x=test_data['x'], y=test_data['y'], verbose=0)
        self.logger.log_test_data(
            round_number=config['rnd'], cid=self.cid,
            num_samples=len(test_data['x']), loss=loss, accuracy=accuracy)
        return loss, len(test_data['x']), {"accuracy": accuracy}
