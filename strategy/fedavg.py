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
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

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

from strategy.criteria import AvailableClientIDCriterion

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""

class FedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fit_clients: int = 1,
        eval_clients: int = 1,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        evaluate_every: int = 1,
        available_fit_client_id = [0],
        available_eval_client_id = [0],
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fit_clients : int, optional
            Number of clients used during training. Defaults to 1.
        eval_clients : int, optional
            Number of clients used during validation. Defaults to 1.
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
        
        self.fit_clients = fit_clients
        self.eval_clients = eval_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.evaluate_every = evaluate_every
        self.available_fit_client_id = available_fit_client_id
        self.available_eval_client_id = available_eval_client_id
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

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

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=self.fit_clients, criterion=AvailableClientIDCriterion(self.available_fit_client_id)
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if rnd % self.evaluate_every != 0:
            return []
        
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=self.eval_clients, criterion=AvailableClientIDCriterion(self.available_eval_client_id)
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

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
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        return weights_to_parameters(aggregate(weights_results)), {}

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

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """This function is used for centralized evaluation.
        As we will not use centralized evaluation, 
        this function will always return None."""
        return None