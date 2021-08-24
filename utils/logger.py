from typing import Dict
import os
import json

class Logger():
    # Initializes logger.
    # Create a folder named ''experiment_name'' into logs folder.
    # Generate a config log file into the created folder.
    def __init__(self, config):
        # Initialize file names
        self.config_file_name = 'config.json'
        self.train_file_name = 'train.csv'
        self.test_file_name = 'test.csv'
        # Initialize log folder path
        self.log_folder = os.path.join('.', 'log', config['name'])
        os.makedirs(self.log_folder)
        # Create initial files
        self._write_config_file(config)
        self._write_train_file_header()
        self._write_test_file_header()
    
    def _write_config_file(self, config: Dict[str, str]):
        with open(os.path.join(self.log_folder, self.config_file_name), 'w') as config_file:
            config_file.write(json.dumps(config))
    
    def _write_train_file_header(self):
        with open(os.path.join(self.log_folder, self.train_file_name), 'a') as train_file:
            train_file.write(f'round_number,cid,loss,accuracy\n')

    def _write_test_file_header(self):
        with open(os.path.join(self.log_folder, self.test_file_name), 'a') as test_file:
            test_file.write(f'round_number,cid,loss,accuracy\n')

    def log_train_data(self, round_number, cid, history):
        if self.log_folder is None:
            return
        with open(os.path.join(self.log_folder, self.train_file_name), 'a') as train_file:
            loss = history.history['loss'][-1] # Last epoch
            accuracy = history.history['accuracy'][-1] # Last epoch
            train_file.write(f'{round_number},{cid},{loss},{accuracy}\n')

    def log_test_data(self, round_number, cid, loss, accuracy):
        if self.log_folder is None:
            return
        with open(os.path.join(self.log_folder, self.test_file_name), 'a') as test_file:
            test_file.write(f'{round_number},{cid},{loss},{accuracy}\n')

# For testing purpose
if __name__ == "__main__":
    pass