
"""Defines the ServerApp."""

import csv
import os

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .task import get_weights, Net


class LoggingFedAvg(FedAvg):

    def aggregate_fit(self, rnd, results, failures):
        if failures:
            print(f"\nFIT FAILURE pada Round {rnd}:")
            for failure in failures:
                print(repr(failure))

        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        if failures:
            print(f"\nEVALUATE FAILURE pada Round {rnd}:")
            for failure in failures:
                print(repr(failure))

        if results:
            losses = [r.loss for _, r in results]
            accuracies = [
                r.metrics["accuracy"] for _, r in results
            ]

            avg_loss = sum(losses) / len(losses)
            avg_acc = sum(accuracies) / len(accuracies)

            os.makedirs("results", exist_ok=True)
            csv_path = "results/metrics.csv"

            write_header = (
                rnd == 1 or not os.path.exists(csv_path)
            )

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)

                if write_header:
                    writer.writerow(
                        ["round", "avg_loss", "avg_accuracy"]
                    )

                writer.writerow([rnd, avg_loss, avg_acc])

        return super().aggregate_evaluate(rnd, results, failures)


def server_fn(context):
    num_rounds = context.run_config.get("num-server-rounds", 3)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)

    net = Net()

    # Ubah list bobot menjadi objek Parameters milik Flower
    parameters = ndarrays_to_parameters(get_weights(net))

    strategy = LoggingFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=3,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


app = ServerApp(server_fn=server_fn)
