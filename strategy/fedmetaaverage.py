# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple
import copy
import random

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from strategy.criteria import TrainingClientLimitCriterion

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""

class FedMetaAverage(Strategy):
    """Configurable FedMeta strategy implementation."""

    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        num_per_cond_per_cycle: int = 4,
        num_partitions_for_multi_cond_task: int = 4,
        multi_cond_multiplier: int = 1,
    ) -> None:
        """FedMeta strategy.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.num_per_cond_per_cycle = num_per_cond_per_cycle
        self.num_partitions_for_multi_cond_task = num_partitions_for_multi_cond_task
        self.multi_cond_multiplier = multi_cond_multiplier
        self.total_rnd_per_cycle = num_per_cond_per_cycle + num_partitions_for_multi_cond_task * 2

    def __repr__(self) -> str:
        rep = f"FedMetaAverage(accept_failures={self.accept_failures})"
        return rep

    def _add_weights(self, weight1, weight2):
        return [p1 + p2 for p1, p2 in zip(weight1, weight2)]

    def _subtract_weights(self, weight1, weight2):
        return [p1 - p2 for p1, p2 in zip(weight1, weight2)]

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """This function is used for centralized evaluation.
        As we will not use centralized evaluation, 
        this function will always return None."""
        return None

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        round_in_cycle = (rnd - 1) % self.total_rnd_per_cycle
        if round_in_cycle < self.num_per_cond_per_cycle:
            mode = "per-cond"
            round_in_phase = round_in_cycle
        elif round_in_cycle < self.num_per_cond_per_cycle + self.num_partitions_for_multi_cond_task:
            mode = "multi-cond-learn"
            round_in_phase = round_in_cycle - self.num_per_cond_per_cycle
        else:
            mode = "multi-cond-update"
            round_in_phase = round_in_cycle - self.num_per_cond_per_cycle - self.num_partitions_for_multi_cond_task
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        if mode == "multi-cond-learn":
            sample_size = sample_size * self.multi_cond_multiplier
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients, criterion=TrainingClientLimitCriterion(10)
        )

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        
        config["mode"] = mode
        if mode == "per-cond":
            fit_ins = FitIns(parameters, config)
            # Return client/config pairs
            return [(client, fit_ins) for client in clients]
        elif mode == "multi-cond-learn":
            if round_in_phase == 0: # First round
                # Initialize class partitions for each multi-cond task.
                self.class_partitions = [
                    self.get_class_partition(62, self.num_partitions_for_multi_cond_task)
                    for client in clients
                ]
                # Record beginning parameter at this point
                self.beginning_parameters = parameters
                # Initialize list of parameters
                self.theta = [[] for i in range(len(clients))]
                # Initialize record for selected clients
                self.clients_record = {}
            self.clients_record[rnd] = clients
            fit_configuration = []
            for i in range(len(clients)):
                config_new = copy.deepcopy(config)
                config_new['selected-class'] = self.class_partitions[i][round_in_phase]
                fit_configuration.append((clients[i], FitIns(self.beginning_parameters, config_new)))
            return fit_configuration
        else:
            if round_in_phase == 0: # First round
                # Copy theta_prime at this point into theta
                self.theta = [
                    Parameters(tensors=t.tensors, tensor_type=t.tensor_type)
                    for t in self.theta_prime
                ]
            # Don't use selected clients. Use recorded clients.
            clients = self.clients_record[rnd - self.num_partitions_for_multi_cond_task]
            fit_configuration = []
            for i in range(len(clients)):
                config_new = copy.deepcopy(config)
                config_new['selected-class'] = self.class_partitions[i][round_in_phase]
                fit_configuration.append((clients[i], FitIns(self.theta[i], config_new)))
            return fit_configuration

    def get_class_partition(self, num_classes, n):  
        lst = list(range(num_classes))
        random.shuffle(lst)
        division = len(lst) / n
        return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        round_in_cycle = (rnd - 1) % self.total_rnd_per_cycle
        if round_in_cycle < self.num_per_cond_per_cycle:
            mode = "per-cond"
        elif round_in_cycle < self.num_per_cond_per_cycle + self.num_partitions_for_multi_cond_task:
            mode = "multi-cond-learn"
        else:
            mode = "multi-cond-update"

        if mode == "per-cond":
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            return weights_to_parameters(aggregate(weights_results)), {}
        elif mode == "multi-cond-learn":
            results = self.match_order(results, self.clients_record[rnd])
            for i in range(len(results)):
                self.theta[i].append(results[i][1].parameters)
            if round_in_cycle == self.num_per_cond_per_cycle + self.num_partitions_for_multi_cond_task - 1: # Final round
                self.theta_prime = []
                for theta_list in self.theta:
                    weights_results = [
                        (parameters_to_weights(p), 1) for p in theta_list
                    ]
                    self.theta_prime.append(weights_to_parameters(aggregate(weights_results)))
            return None, {}
        else:
            results = self.match_order(results, self.clients_record[rnd - self.num_partitions_for_multi_cond_task])
            self.theta = [
                fit_res.parameters
                for cilent, fit_res in results
            ]
            if round_in_cycle == self.total_rnd_per_cycle - 1: # Final round
                weights_results = [
                    (self._add_weights(
                        self._subtract_weights(parameters_to_weights(self.theta[i]), parameters_to_weights(self.theta_prime[i])),
                        parameters_to_weights(self.beginning_parameters))
                    , 1)
                    for i in range(len(self.theta))
                ]
                return weights_to_parameters(aggregate(weights_results)), {}
            return None, {}

    def match_order(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        clients_record: List[ClientProxy],
    ) -> List[Tuple[ClientProxy, FitRes]]:
        ordered_results = []
        for client in clients_record:
            for res in results:
                if res[0].cid == client.cid:
                    ordered_results.append(res)
        return ordered_results

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Only evaluate every cycle
        if rnd % self.total_rnd_per_cycle != 0:
            return []
        if (rnd // self.total_rnd_per_cycle) % 2 != 0:
            return []
        
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {}
