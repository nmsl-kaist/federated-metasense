import os
from typing import Dict

import flwr as fl
import numpy as np
import tensorflow as tf

from strategy.fedavg import FedAvg
#from strategy.fedmetasequential import FedMetaSequential
#from strategy.fedmetaaverage import FedMetaAverage
from strategy.fedmetaaverage2 import FedMetaAverage2
#from strategy.fedmetawhole import FedMetaWhole

from utils.logger import Logger
from utils.config import get_configuration

from client.DefaultClient import DefaultClient
from client.MetaClient import MetaClient
from client.SpecialClientWithSeparation import SpecialClientWithSeparation
#from client.SpecialClientNoSeparation import SpecialClientNoSeparation

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Use minimal memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Start Ray simulation
if __name__ == "__main__":

    # Get configuration
    config = get_configuration()

    # Initialize logger
    logger = Logger(config)

    # Print num_clients
    print(f"Number of requested clients: {config['num_clients']}")

    # Initialize fit/eval config functions
    def fit_config(rnd: int) -> Dict[str, str]:
        """
        Return a fit configuration.
        The returned configuration is passed as an argument ''config'' in client's ''fit'' function.
        """
        config["fit_config"]["rnd"] = str(rnd)
        return config["fit_config"]

    def eval_config(rnd: int) -> Dict[str, str]:
        """
        Return an evaluation configuration.
        The returned configuration is passed as an argument ''config'' in client's ''evaluate'' function.
        """
        config["eval_config"]["rnd"] = str(rnd)
        return config["eval_config"]

    def client_fn(cid: str):
        """
        Create a single client instance.
        """
        if config["args_suffix"] == 'average2':
            return SpecialClientWithSeparation(cid, logger, config["dataset_name"])
        elif config["args_suffix"] == 'meta':
            return MetaClient(cid, logger, config["dataset_name"])
        else:
            return DefaultClient(cid, logger, config["dataset_name"])

    # Initialize the strategy
    if config["args_suffix"] == 'average2':
        strategy = FedMetaAverage2(
            fit_clients=config["fit_clients"],
            eval_clients=config["eval_clients"],
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_every=config['strategy_config']['num_per_cond_per_cycle'] + config['strategy_config']['num_partitions_for_multi_cond_task'] * 2,
            available_fit_client_id=config["strategy_config"]["available_fit_client_id"],
            available_eval_client_id=config["strategy_config"]["available_eval_client_id"],
            num_per_cond_per_cycle=config['strategy_config']['num_per_cond_per_cycle'],
            num_partitions_for_multi_cond_task=config['strategy_config']['num_partitions_for_multi_cond_task'],
            multi_cond_multiplier=config['strategy_config']['multi_cond_multiplier'],
            num_classes=config['num_classes'],
        )
    else:
        strategy = FedAvg(
            fit_clients=config["fit_clients"],
            eval_clients=config["eval_clients"],
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_every=config["evaluate_every"],
            available_fit_client_id=config["strategy_config"]["available_fit_client_id"],
            available_eval_client_id=config["strategy_config"]["available_eval_client_id"],
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
