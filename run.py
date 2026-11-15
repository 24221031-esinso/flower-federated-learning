from flwr.simulation import run_simulation
from pytorchexample.client_app import app as client_app
from pytorchexample.server_app import app as server_app

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=1,  # Ubah dari 3 menjadi 1
    backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    verbose_logging=True,
)