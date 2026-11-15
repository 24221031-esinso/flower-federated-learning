"""Defines the ClientApp."""
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from .task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs, learning_rate, device):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device

    def get_parameters(self, config):
        return get_weights(self.net)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.learning_rate,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 1)
    trainloader, testloader = load_data(partition_id, num_partitions)

    local_epochs = context.run_config.get("local-epochs", 1)
    learning_rate = context.run_config.get("learning-rate", 0.01)

    return FlowerClient(
        net, trainloader, testloader, local_epochs, learning_rate, device
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)