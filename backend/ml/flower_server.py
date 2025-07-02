import flwr as fl
import tensorflow as tf
import os
import pickle
import requests
from flwr.server import ServerConfig
from encryption import decrypt_bytes, get_shared_key
import numpy as np

POCKETBASE_URL = "http://127.0.0.1:8090"
MODEL_DIR = "../models"

PB_TOKEN = "YOUR PB ADMIN TOKEN"
if not PB_TOKEN:
    raise RuntimeError("PB_TOKEN environment variable not set. Please set PB_TOKEN to a valid PocketBase user token.")
headers = {"Authorization": f"Bearer {PB_TOKEN}"}

def decrypt_weights(ciphertext_str: str) -> list:
    decrypted_bytes = decrypt_bytes(ciphertext_str, get_shared_key())
    return pickle.loads(decrypted_bytes)

def get_num_clients_from_pocketbase():
    try:
        res = requests.get(f"{POCKETBASE_URL}/api/collections/global/records", headers=headers)
        res.raise_for_status()
        items = res.json().get("items", [])
        if not items:
            return 0
        latest = items[0]
        clients_str = latest.get("clients", "")
        clients = [c for c in clients_str.split(",") if c.strip()]
        num_clients = len(set(clients))
        return num_clients
    except Exception as e:
        print(f"Failed to fetch clients from PocketBase: {e}")
        return 0
    
def build_global_model(input_shape: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class SecureFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def aggregate_fit(self, server_round, results, failures):
        print(f"[DEBUG] aggregate_fit called for round {server_round} with {len(results)} results and {len(failures)} failures.")
        decrypted_results = []
        contributing_clients = set()

        for idx, (client_proxy, fit_res) in enumerate(results):
            print(f"[DEBUG] Processing result {idx+1}/{len(results)}")
            try:
                encrypted_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)[0].item()
                print(f"[DEBUG] Encrypted weights received from client.")
                weights = decrypt_weights(encrypted_weights)
                print(f"[DEBUG] Weights decrypted successfully.")
                decrypted_results.append((weights, fit_res.num_examples))
                client_id = fit_res.metrics.get("client_id", "unknown")
                print(f"[DEBUG] Client ID: {client_id}, num_examples: {fit_res.num_examples}")
                contributing_clients.add(client_id)
            except Exception as e:
                print(f"[ERROR] Failed to process client result: {e}")

        print(f"[DEBUG] Aggregating weights from {len(decrypted_results)} clients.")
        def weighted_average(results):
            total_examples = sum(num_examples for _, num_examples in results)
            if total_examples == 0:
                return None
            avg = [np.zeros_like(w) for w in results[0][0]]
            for weights, num_examples in results:
                for i, w in enumerate(weights):
                    avg[i] += w * (num_examples / total_examples)
            return avg

        if decrypted_results:
            aggregated_weights = weighted_average(decrypted_results)
            if aggregated_weights is not None:
                print(f"[DEBUG] Aggregation successful. Setting model weights.")
                aggregated_weights_params = fl.common.ndarrays_to_parameters(aggregated_weights)
                self.model.set_weights(aggregated_weights)

                model_path = os.path.join(MODEL_DIR, f"global_model_round_{server_round}_clients_{max(1, NUM_CLIENTS)}.h5")
                self.model.save(model_path)
                print(f"Global model saved: {model_path}")

                try:
                    with open(model_path, "rb") as f:
                        files = {"global_model": f}
                        data = {"round": str(server_round), "clients": str(max(1, NUM_CLIENTS))}
                        res = requests.post(f"{POCKETBASE_URL}/api/collections/global/records", data=data, files=files, headers=headers)
                    if res.ok:
                        print(f"Global model uploaded to PocketBase.")
                    else:
                        print(f"Failed to upload global model: {res.text}")
                except Exception as e:
                    print(f"[ERROR] Exception during upload to PocketBase: {e}")
                return aggregated_weights_params, {"num_clients": len(decrypted_results)}
            else:
                print(f"[ERROR] Aggregation failed, no weights to set.")
                return None, {"num_clients": 0}
        else:
            print(f"[ERROR] No decrypted results to aggregate.")
            return None, {"num_clients": 0}

    def upload_to_pocketbase(self, model_path, clients_set):
        res = requests.get(f"{POCKETBASE_URL}/api/collections/global/records", headers=headers)

        if not res.ok:
            raise RuntimeError(f"Global record fetch failed: {res.text}")

        items = res.json().get("items", [])
        if not items:
            raise RuntimeError("No global model record exists in PocketBase")

        global_id = items[0]["id"]
        prev_clients = set(items[0].get("clients", "").split(",")) if items[0].get("clients") else set()
        all_clients = prev_clients.union(clients_set)

        with open(model_path, "rb") as f:
            files = {"global_model": f}
            data = {"clients": ",".join(all_clients)}

            update_res = requests.patch(
                f"{POCKETBASE_URL}/api/collections/global/records/{global_id}",
                data=data,
                files=files,
                headers=headers
            )

        if not update_res.ok:
            raise RuntimeError(f"Update failed: {update_res.text}")
        print(f"[☁️] Uploaded global model to PocketBase with {len(all_clients)} clients.")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    input_shape = 13

    NUM_CLIENTS = get_num_clients_from_pocketbase()
    print(f"Found {NUM_CLIENTS} unique clients in PocketBase global record.")

    model = build_global_model(input_shape)
    strategy = SecureFedAvg(
        model=model,
        fraction_fit=1.0,
        min_fit_clients=max(1, NUM_CLIENTS),
        min_available_clients=max(1, NUM_CLIENTS),
        on_fit_config_fn=lambda rnd: {"rnd": rnd}
    )

    print("Starting Flower server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=1)
    )

