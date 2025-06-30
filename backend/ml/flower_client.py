# flower_client.py
import flwr as fl
import tensorflow as tf
import joblib
import numpy as np
import sys
import os
from encryption import encrypt_weights
from encryption import generate_shared_key, decrypt_bytes

shared_key = generate_shared_key()

def encrypt_weights(weights):
    import pickle
    raw = pickle.dumps(weights)
    return encrypt_bytes(raw, shared_key)

MODEL_DIR = "../models"

class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        weights = self.model.get_weights()
        encrypted = encrypt_weights(weights)
        return encrypted

    def fit(self, parameters, config):
        decrypted = parameters  # since mock, no decryption
        self.model.set_weights(decrypted)
        self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=32)
        encrypted = encrypt_weights(self.model.get_weights())
        return encrypted, len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def start_client(company):
    model_path = os.path.join(MODEL_DIR, f"local_{company}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model not found for {company}")

    model, preprocessor = joblib.load(model_path)

    # Mock data loading (replace with actual persisted data)
    import pandas as pd
    df = pd.read_csv(f"../datasets/{company}.csv")
    X = df.drop("label_is_fraud", axis=1)
    y = df["label_is_fraud"]
    X = preprocessor.transform(X).astype(np.float32)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    client = FraudDetectionClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python flower_client.py <company>")
    else:
        start_client(sys.argv[1])
