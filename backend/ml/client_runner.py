import pandas as pd
import numpy as np
import joblib
import sys
import os
import requests
import io
import time
from client_helper import create_preprocessor, build_model
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from datetime import datetime

# ⛳ Config
DATA_DIR = "../datasets"
MODEL_DIR = "../models"
POCKETBASE_URL = "http://127.0.0.1:8090"
COLLECTION = "data"  # Use DATA collection


def get_dataset_url(company, pb_token, max_retries=5, delay=2):
    url = f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records?filter=company='{company}'"
    headers = {"Authorization": f"Bearer {pb_token}"}
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers)
        print(f"[DEBUG] Querying: {url}")
        print(f"[DEBUG] Response: {resp.text}")
        items = resp.json().get("items", [])
        if items and items[0].get("dataset"):
            record = items[0]
            collection_id = record.get("collectionId", COLLECTION)
            record_id = record["id"]
            filename = record["dataset"]
            dataset_url = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{filename}"
            return dataset_url
        print(f"[WARN] Dataset not found for company '{company}', retrying ({attempt+1}/{max_retries})...")
        time.sleep(delay)
    # Print all items for inspection if not found
    print(f"[ERROR] No matching dataset found after {max_retries} retries. Last response items: {items}")
    raise Exception("Dataset not found for company after several retries")

def update_pocketbase_status(company, model_path, pb_token):
    # Fetch record ID
    headers = {"Authorization": f"Bearer {pb_token}"}
    res = requests.get(f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records?filter=company='{company}'", headers=headers)
    items = res.json().get("items", [])
    if not items:
        print(f"[❌] No PocketBase record found for {company}")
        return
    record_id = items[0]["id"]

    # Prepare form-data for PATCH
    data = {
        "hasTrained": "true"
    }
    files = {
        "model": open(model_path, "rb")
    }
    res = requests.patch(
        f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records/{record_id}",
        data=data,
        files=files,
        headers=headers
    )
    if res.ok:
        print(f"[✅] PocketBase updated: hasTrained=True and model uploaded for {company}")
    else:
        print(f"[⚠️] Failed to update PocketBase for {company}: {res.text}")

def run_local_training(company, pb_token):
    # Download dataset from PocketBase
    dataset_url = get_dataset_url(company, pb_token)
    print(f"[📁] Downloading dataset from: {dataset_url}")
    resp = requests.get(dataset_url)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content))

    # Drop rows with missing labels and ensure labels are integers
    df = df.dropna(subset=['label_is_fraud'])
    df['label_is_fraud'] = df['label_is_fraud'].astype(int)

    # Separate features and labels
    X = df.drop('label_is_fraud', axis=1)
    y = df['label_is_fraud']

    model_path = os.path.join(MODEL_DIR, f"local_{company}.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    preprocessor = create_preprocessor(X_train)
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)

    # Handle class imbalance
    present_classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=present_classes,
        y=y_train
    )
    weight_dict = dict(enumerate(class_weights))

    # Manual validation split for DP safety
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build model with DP-compatible microbatches
    input_shape = X_train.shape[1]
    batch_size = 32
    num_microbatches = 32
    assert batch_size % num_microbatches == 0, "Batch size must be divisible by microbatches"

    model = build_model(input_shape, num_microbatches=num_microbatches)

    remainder = len(X_train_final) % batch_size
    if remainder != 0:
        drop_count = remainder
        print(f"[⚠️] Trimming {drop_count} samples to fit batch size requirements...")
        X_train_final = X_train_final[:-drop_count]
        y_train_final = y_train_final[:-drop_count]
    # Train
    print(f"[🧠] Training model for {company}...")
    model.fit(
        X_train_final,
        y_train_final,
        epochs=10,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=weight_dict
    )

    # Save model and preprocessor
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump((model, preprocessor), model_path)
    print(f"[💾] Model saved to: {model_path}")

    # Upload model to PocketBase
    update_pocketbase_status(company, model_path, pb_token)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❌ Usage: python client_runner.py <company> <pb_token>")
    else:
        run_local_training(sys.argv[1], sys.argv[2])
