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
from client_helper import create_preprocessor, build_model
import pickle
from encryption import generate_symmetric_key, encrypt_bytes

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


def update_pocketbase_status(company, model_path, pb_token):
    headers = {"Authorization": f"Bearer {pb_token}"}
    res = requests.get(f"{POCKETBASE_URL}/api/collections/data/records?filter=company='{company}'", headers=headers)
    items = res.json().get("items", [])
    if not items:
        print(f"No PocketBase record found for {company}")
        return

    record_id = items[0]["id"]
    data = {"hasTrained": "true"}
    files = {"model": open(model_path, "rb")}
    res = requests.patch(
        f"{POCKETBASE_URL}/api/collections/data/records/{record_id}",
        data=data,
        files=files,
        headers=headers
    )
    if res.ok:
        print(f"PocketBase updated: hasTrained=True and model uploaded for {company}")
    else:
        print(f"Failed to update PocketBase for {company}: {res.text}")

    try:
        os.remove(model_path)
        print(f"Model file cleaned up: {model_path}")
    except Exception as e:
        print(f"Failed to clean up model file: {e}")


def push_to_global_model(company, model, pb_token):
    print(f"Preparing weights for global model contribution...")

    weights = model.get_weights()
    serialized = pickle.dumps(weights)

    shared_key = generate_symmetric_key()
    encrypted_payload = encrypt_bytes(serialized, shared_key)

    print(f"Contributing to global model via PocketBase API...")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    temp_file_path = f"temp_weights_{company}_{timestamp}.bin"

    try:
        with open(temp_file_path, "wb") as f:
            f.write(encrypted_payload.encode() if isinstance(encrypted_payload, str) else encrypted_payload)

        headers = {"Authorization": f"Bearer {pb_token}"}
        res = requests.get(f"{POCKETBASE_URL}/api/collections/global/records", headers=headers)

        if res.ok:
            records = res.json().get("items", [])
            if records:
                global_id = records[0]["id"]
                current_clients = set(records[0].get("clients", "").split(",")) if records[0].get("clients") else set()
                current_clients.add(company)

                with open(temp_file_path, "rb") as f:
                    files = {"global_model": f}
                    data = {"clients": ",".join(current_clients)}

                    update_res = requests.patch(
                        f"{POCKETBASE_URL}/api/collections/global/records/{global_id}",
                        data=data,
                        files=files,
                        headers=headers
                    )

                    if update_res.ok:
                        print(f"Updated global model record with contribution from {company}")
                    else:
                        print(f"Failed to update global model: {update_res.text}")
            else:
                with open(temp_file_path, "rb") as f:
                    files = {"global_model": f}
                    data = {"clients": company}

                    create_res = requests.post(
                        f"{POCKETBASE_URL}/api/collections/global/records",
                        data=data,
                        files=files,
                        headers=headers
                    )

                    if create_res.ok:
                        print(f"Created new global model record with contribution from {company}")
                    else:
                        print(f"Failed to create global model record: {create_res.text}")
        else:
            print(f"Failed to check global model records: {res.text}")

    except Exception as e:
        print(f"Error contributing to global model: {e}")
    finally:
        try:
            os.remove(temp_file_path)
            print(f"Temporary weights file cleaned up: {temp_file_path}")
        except Exception as e:
            print(f"Failed to clean up temporary file: {e}")


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

    preprocessor = create_preprocessor(X_train)
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
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
    batch_size = 32
    num_microbatches = 32
    model = build_model(input_shape, num_microbatches=num_microbatches)

    remainder = len(X_train_final) % batch_size
    if remainder != 0:
        X_train_final = X_train_final[:-remainder]
        y_train_final = y_train_final[:-remainder]

    print(f"Training model for {company}...")
    model.fit(
        X_train_final,
        y_train_final,
        epochs=10,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=weight_dict
    )

    MODEL_DIR = "../models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_filename = f"local_{company}_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump((model, preprocessor), model_path)
    print(f"Model saved to: {model_path}")

    update_pocketbase_status(company, model_path, pb_token)
    push_to_global_model(company, model, pb_token)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python client_runner.py <company> <pb_token>")
    else:
        run_local_training(sys.argv[1], sys.argv[2])
