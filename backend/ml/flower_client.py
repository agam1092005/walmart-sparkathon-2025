# ml/flower_client.py
import flwr as fl
import tensorflow as tf
import joblib
import numpy as np
import sys
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from encryption import encrypt_bytes, get_shared_key
from client_helper import build_model 
import requests

MODEL_DIR = "../models"

class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=32)
        weights_bytes = pickle.dumps(self.model.get_weights())
        encrypted = encrypt_bytes(weights_bytes, get_shared_key())
        return [encrypted], len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def download_model_and_preprocessor_and_dataset(company, pb_token):
    """
    Download model weights (.h5), preprocessor (.pkl), and dataset (.csv) for the given company from PocketBase.
    Save as ../models/weights_<company>.h5, ../models/preprocessor_<company>.pkl, ../datasets/<company>.csv
    """
    POCKETBASE_URL = "http://127.0.0.1:8090"
    headers = {"Authorization": f"Bearer {pb_token}"}
    url = f"{POCKETBASE_URL}/api/collections/data/records?filter=company='{company}'"
    try:
        resp = requests.get(url, headers=headers)
        if not resp.ok:
            print(f"Failed to query PocketBase: {resp.text}")
            return False
        items = resp.json().get("items", [])
        if not items:
            print(f"No PocketBase record found for {company}")
            return False
        record = items[0]
        collection_id = record.get("collectionId", "data")
        record_id = record["id"]
        if "model" in record and record["model"]:
            model_filename = record["model"]
            model_url = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{model_filename}"
            print(f"Downloading model weights from {model_url}")
            model_resp = requests.get(model_url, headers=headers)
            if model_resp.ok:
                model_path = os.path.join(MODEL_DIR, f"weights_{company}.h5")
                with open(model_path, "wb") as f:
                    f.write(model_resp.content)
                print(f"Model weights saved to {model_path}")
            else:
                print(f"Failed to download model weights: {model_resp.text}")
                return False
        else:
            print(f"No model weights found in PocketBase for {company}")
            return False
        if "preprocessor" in record and record["preprocessor"]:
            preproc_filename = record["preprocessor"]
            preproc_url = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{preproc_filename}"
            print(f"Downloading preprocessor from {preproc_url}")
            preproc_resp = requests.get(preproc_url, headers=headers)
            if preproc_resp.ok:
                preproc_path = os.path.join(MODEL_DIR, f"preprocessor_{company}.pkl")
                with open(preproc_path, "wb") as f:
                    f.write(preproc_resp.content)
                print(f"Preprocessor saved to {preproc_path}")
            else:
                print(f"Failed to download preprocessor: {preproc_resp.text}")
                return False
        else:
            print(f"No preprocessor found in PocketBase for {company}")
            return False
        if "dataset" in record and record["dataset"]:
            dataset_filename = record["dataset"]
            dataset_url = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{dataset_filename}"
            print(f"Downloading dataset from {dataset_url}")
            dataset_resp = requests.get(dataset_url, headers=headers)
            if dataset_resp.ok:
                os.makedirs("../datasets", exist_ok=True)
                dataset_path = os.path.join("../datasets", f"{company}.csv")
                with open(dataset_path, "wb") as f:
                    f.write(dataset_resp.content)
                print(f"Dataset saved to {dataset_path}")
            else:
                print(f"Failed to download dataset: {dataset_resp.text}")
                return False
        else:
            print(f"No dataset found in PocketBase for {company}")
            return False
        return True
    except Exception as e:
        print(f"Exception during download: {e}")
        return False

def start_client(company, pb_token=None):
    weight_path = os.path.join(MODEL_DIR, f"weights_{company}.h5")
    preproc_path = os.path.join(MODEL_DIR, f"preprocessor_{company}.pkl")
    dataset_path = os.path.join("../datasets", f"{company}.csv")

    if not (os.path.exists(weight_path) and os.path.exists(preproc_path) and os.path.exists(dataset_path)):
        if pb_token is not None:
            print(f"Required files missing for {company}. Attempting to download from PocketBase...")
            success = download_model_and_preprocessor_and_dataset(company, pb_token)
            if not success:
                print(f"Could not download required files for {company}. Exiting.")
                return
        else:
            print(f"Required files missing for {company} and no PocketBase token provided.")
            return

    try:
        preprocessor = joblib.load(preproc_path)
    except Exception as e:
        print(f"Failed to load preprocessor: {e}")
        return

    try:
        df = pd.read_csv(dataset_path)
        X = df.drop("label_is_fraud", axis=1)
        y = df["label_is_fraud"]
        X = preprocessor.transform(X).astype(np.float32)
    except Exception as e:
        print(f"Failed to load or preprocess dataset: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = X.shape[1]
    model = build_model(input_shape, num_microbatches=32)

    try:
        model.load_weights(weight_path)
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    client = FraudDetectionClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        start_client(sys.argv[1])
    elif len(sys.argv) == 3:
        start_client(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python flower_client.py <company> [pb_token]")
