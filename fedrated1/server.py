import flwr as fl

# Start Flower server with 3 training rounds
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=0)
)