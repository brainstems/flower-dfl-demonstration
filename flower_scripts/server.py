import flwr as fl
import pickle
from pathlib import Path
from flwr.common import parameters_to_ndarrays
from logging import INFO, DEBUG
from flwr.common.logger import log

fl.common.logger.configure(identifier="flowerIngredMistral", filename="log.txt")


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples),
    }


class FedAvgWithModelSaving(fl.server.strategy.FedAvg):
    """This is a custom strategy that behaves exactly like
    FedAvg with the difference of storing of the state of
    the global model to disk after each round.
    """

    def __init__(self, save_path: str, *args, **kwargs):
        self.save_path = Path(save_path)
        # Create directory if needed
        self.save_path.mkdir(exist_ok=True, parents=True)
        super().__init__(*args, **kwargs)

    def _save_global_model(self, server_round: int, parameters):
        """A new method to save the parameters to disk."""

        # convert parameters to list of NumPy arrays
        # this will make things easy if you want to load them into a
        # PyTorch or TensorFlow model later
        ndarrays = parameters_to_ndarrays(parameters)
        data = {"globa_parameters": ndarrays}
        filename = str(self.save_path / f"parameters_round_{server_round}.pkl")
        with open(filename, "wb") as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
        log(1, f"Checkpoint saved to: {filename}")

    def evaluate(self, server_round: int, parameters):
        """Evaluate model parameters using an evaluation function."""
        # save the parameters to disk using a custom method
        self._save_global_model(server_round, parameters)

        # call the parent method so evaluation is performed as
        # FedAvg normally does.
        return super().evaluate(server_round, parameters)


# Create strategy and run server
strategy = FedAvgWithModelSaving(
    save_path="./models",  # save model to this directory
    fraction_fit=0.8,
    fraction_evaluate=0.8,
    evaluate_metrics_aggregation_fn=weighted_average,
)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=4),
    strategy=strategy,
    certificates=(
        Path(".cache/certificates/ca.crt").read_bytes(),
        Path(".cache/certificates/server.pem").read_bytes(),
        Path(".cache/certificates/server.key").read_bytes(),
    ),
)

# Start with Superlink
"""
flower-superlink
  --ssl-ca-certfile certificates/ca.crt
  --ssl-certfile certificates/server.pem
  --ssl-keyfile certificates/server.key
"""
