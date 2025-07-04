import pandas as pd
import numpy as np
import joblib
import sys
import os
import requests
import io
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from client_helper import create_preprocessor, build_model, load_global_preprocessor_from_pocketbase

POCKETBASE_URL = "http://127.0.0.1:8090"


def get_dataset_url(company, pb_token, max_retries=5, delay=2):
    url = f"{POCKETBASE_URL}/api/collections/data/records?filter=company='{company}'"
    headers = {"Authorization": f"Bearer {pb_token}"}
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers)
        print(f"[DEBUG] Querying: {url}")
        print(f"[DEBUG] Response: {resp.text}")
        items = resp.json().get("items", [])
        if items and items[0].get("dataset"):
            record = items[0]
            collection_id = record.get("collectionId", "data")
            record_id = record["id"]
            filename = record["dataset"]
            dataset_url = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{filename}"
            return dataset_url
        print(f"[WARN] Dataset not found for '{company}', retrying ({attempt+1}/{max_retries})...")
        time.sleep(delay)
    raise Exception("Dataset not found after several retries")


def update_pocketbase_status(company, model_path, weight_path, preproc_path, pb_token):
    headers = {"Authorization": f"Bearer {pb_token}"}
    res = requests.get(f"{POCKETBASE_URL}/api/collections/data/records?filter=company='{company}'", headers=headers)
    items = res.json().get("items", [])
    if not items:
        print(f"No PocketBase record found for {company}")
        return

    record_id = items[0]["id"]
    data = {"hasTrained": "true"}
    files = {
        "model": open(model_path, "rb"),
        "weight": open(weight_path, "rb"),
        "preprocessor": open(preproc_path, "rb")
    }
    res = requests.patch(
        f"{POCKETBASE_URL}/api/collections/data/records/{record_id}",
        data=data,
        files=files,
        headers=headers
    )
    if res.ok:
        print(f"PocketBase updated: hasTrained=True, model, weight, and preprocessor uploaded for {company}")
    else:
        print(f"Failed to update PocketBase for {company}: {res.text}")

    try:
        os.remove(model_path)
        print(f"Model file cleaned up: {model_path}")
    except Exception as e:
        print(f"Failed to clean up model file: {e}")
    try:
        os.remove(weight_path)
        print(f"Weight file cleaned up: {weight_path}")
    except Exception as e:
        print(f"Failed to clean up weight file: {e}")
    try:
        os.remove(preproc_path)
        print(f"Preprocessor file cleaned up: {preproc_path}")
    except Exception as e:
        print(f"Failed to clean up preprocessor file: {e}")


def push_to_global_model(company, model, pb_token):
    import subprocess
    import threading
    import time

    print(f"Initiating federated contribution to global model...")

    def run_server():
        subprocess.run(["python3", "ml/flower_server.py", pb_token])

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    time.sleep(5)

    MODEL_DIR = "../models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    latest_model_path = os.path.join(MODEL_DIR, f"local_{company}.pkl")
    joblib.dump(model, latest_model_path)

    result = subprocess.run(["python3", "ml/flower_client.py", company, pb_token])
    if result.returncode != 0:
        print("Client failed to contribute")
        return

    print(f"Federated round complete. Everything is done.")



def run_local_training(company, pb_token):
    dataset_url = get_dataset_url(company, pb_token)
    print(f"Downloading dataset from: {dataset_url}")
    resp = requests.get(dataset_url)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content))

    df = df.dropna(subset=['label_is_fraud'])
    df['label_is_fraud'] = df['label_is_fraud'].astype(int)

    X = df.drop('label_is_fraud', axis=1)
    y = df['label_is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = load_global_preprocessor_from_pocketbase(pb_token, X_train=X_train, company=company)
    X_train = preprocessor.transform(X_train).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)

    try:
        present_classes = np.unique(y_train)
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=present_classes, y=y_train)
        weight_dict = dict(enumerate(class_weights))
    except Exception as e:
        print(f"Class weight computation failed: {e}")
        weight_dict = None

    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1]
    batch_size = min(32, len(X_train_final))
    if batch_size < 1:
        print("[ERROR] Not enough samples to train. Skipping local training.")
        return
    remainder = len(X_train_final) % batch_size
    if remainder != 0:
        X_train_final = X_train_final[:-remainder]
        y_train_final = y_train_final[:-remainder]
    num_microbatches = batch_size
    model = build_model(input_shape, num_microbatches=num_microbatches)

    print(f"Training model for {company} with batch_size={batch_size}...")
    try:
        model.fit(
            X_train_final,
            y_train_final,
            epochs=10,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=weight_dict
        )
    except Exception as e:
        print(f"[ERROR] Exception during model.fit: {e}")
        return

    MODEL_DIR = "../models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save both full model and weights
    model_path = os.path.join(MODEL_DIR, f"model_{company}.h5")
    weight_path = os.path.join(MODEL_DIR, f"weights_{company}.h5")
    model.save(model_path)
    model.save_weights(weight_path)

    preproc_path = os.path.join(MODEL_DIR, f"preprocessor_{company}.pkl")
    joblib.dump(preprocessor, preproc_path)

    print(f"Model saved to: {model_path}")
    print(f"Weights saved to: {weight_path}")
    print(f"Preprocessor saved to: {preproc_path}")

    update_pocketbase_status(company, model_path, weight_path, preproc_path, pb_token)

    push_to_global_model(company, model, pb_token)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python client_runner.py <company> <pb_token>")
    else:
        run_local_training(sys.argv[1], sys.argv[2])
