from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

class AvailableClientIDCriterion(Criterion):
    def __init__(self, available_id_list):
        super().__init__()
        self.available_id_list = available_id_list
    
    def select(self, client: ClientProxy) -> bool:
        return int(client.cid) in self.available_id_list
