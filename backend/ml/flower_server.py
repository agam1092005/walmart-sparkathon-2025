import flwr as fl
import tensorflow as tf
from encryption import decrypt_weights
import numpy as np
import os
from encryption import generate_shared_key, decrypt_bytes

shared_key = generate_shared_key()

def decrypt_weights(ciphertext_str):
    import pickle
    decrypted_bytes = decrypt_bytes(ciphertext_str, shared_key)
    return pickle.loads(decrypted_bytes)

MODEL_DIR = "../models"
NUM_CLIENTS = 2

def build_global_model(input_shape):
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
    def aggregate_fit(self, rnd, results, failures):
        decrypted_results = []
        for parameters, num_examples, metrics in results:
            decrypted = decrypt_weights(fl.common.parameters_to_ndarrays(parameters))
            decrypted_results.append((decrypted, num_examples))
        aggregated = fl.server.strategy.aggregate_ecdsa(decrypted_results)
        if aggregated:
            aggregated_weights = fl.common.ndarrays_to_parameters(aggregated)
            self.model.set_weights(aggregated)
            model_path = os.path.join(MODEL_DIR, f"global_model_round_{rnd}_clients_{NUM_CLIENTS}.h5")
            self.model.save(model_path)
            print(f"[💾] Global model saved: {model_path}")
        return super().aggregate_fit(rnd, results, failures)

if __name__ == "__main__":
    input_shape = 100  # Set dynamically or from first client
    strategy = SecureFedAvg(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"rnd": rnd}
    )
    model = build_global_model(input_shape)
    strategy.model = model
    print("🚀 Starting Flower server...")
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config={"num_rounds": 15})
