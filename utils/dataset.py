'''
Some code snippets are borrowed from LEAF.

LEAF: A Benchmark for Federated Settings
Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li,
Jakub Konečný, H. Brendan McMahan, Virginia Smith, and Ameet Talwalkar.
Workshop on Federated Learning for Data Privacy and Confidentiality (2019).

https://leaf.cmu.edu/
'''

import os
import json
from collections import defaultdict

import numpy as np

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    return train_clients, test_clients, train_data, test_data

class Dataset():
    def __init__(self, dataset_name):
        train_path = os.path.join('.', 'data', dataset_name, 'data', 'train')
        test_path = os.path.join('.', 'data', dataset_name, 'data', 'test')
        train_client_ids, test_client_ids, train_data_dict, test_data_dict = read_data(train_path, test_path)
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
