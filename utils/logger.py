from typing import Dict
import os
import json

from utils.log_keys import (
    NUM_ROUND_KEY,
    ACCURACY_KEY,
    LOSS_KEY,
    CLIENT_ID_KEY,
    NUM_SAMPLES_KEY,
)

class Logger():
    # Initializes logger.
    # Create a folder named ''config['name']'' into logs folder.
    # Generate a config log file into the created folder.
    def __init__(self, config):
        # Initialize file names
        self.config_file_name = 'config.json'
        self.train_file_name = 'train.csv'
        self.test_file_name = 'test.csv'
        # Initialize log folder path
        self.log_folder = os.path.join('.', 'log', config['name'])
        os.makedirs(self.log_folder)
        # Initialize log keys order. Logs are written in this order.
        self.log_keys_list = [NUM_ROUND_KEY, CLIENT_ID_KEY, NUM_SAMPLES_KEY, LOSS_KEY, ACCURACY_KEY]
        # Create initial files
        self._write_config_file(config)
        self._write_train_file_header()
        self._write_test_file_header()

    def _join_strings_by_comma(self, strings):
        return ','.join(strings) + '\n'
    
    def _write_config_file(self, config: Dict[str, str]):
        with open(os.path.join(self.log_folder, self.config_file_name), 'w') as config_file:
            config_file.write(json.dumps(config))
    
    def _write_train_file_header(self):
        with open(os.path.join(self.log_folder, self.train_file_name), 'a') as train_file:
            train_file.write(self._join_strings_by_comma(self.log_keys_list))

    def _write_test_file_header(self):
        with open(os.path.join(self.log_folder, self.test_file_name), 'a') as test_file:
            test_file.write(self._join_strings_by_comma(self.log_keys_list))

    def _get_log_string(self, log_data_dict: Dict[str, str]):
        log_data_list = list(map(lambda key: log_data_dict[key], self.log_keys_list))
        return self._join_strings_by_comma(log_data_list)

    def log_train_data(self, round_number, cid, num_samples, history):
        if self.log_folder is None:
            return
        with open(os.path.join(self.log_folder, self.train_file_name), 'a') as train_file:
            loss = history.history['loss'][-1] # Last epoch
            accuracy = history.history['accuracy'][-1] # Last epoch
            log_data_dict = {
                NUM_ROUND_KEY: str(round_number),
                CLIENT_ID_KEY: str(cid),
                NUM_SAMPLES_KEY: str(num_samples),
                LOSS_KEY: str(loss),
                ACCURACY_KEY: str(accuracy),}
            train_file.write(self._get_log_string(log_data_dict))

    def log_test_data(self, round_number, cid, num_samples, loss, accuracy):
        if self.log_folder is None:
            return
        with open(os.path.join(self.log_folder, self.test_file_name), 'a') as test_file:
            log_data_dict = {
                NUM_ROUND_KEY: str(round_number),
                CLIENT_ID_KEY: str(cid),
                NUM_SAMPLES_KEY: str(num_samples),
                LOSS_KEY: str(loss),
                ACCURACY_KEY: str(accuracy),}
            test_file.write(self._get_log_string(log_data_dict))

# For testing purpose
if __name__ == "__main__":
    pass