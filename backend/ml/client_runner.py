import numpy as np
import joblib
import sys
import os
import requests
from client_helper import load_data, create_preprocessor, build_model
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# ⛳ Config
DATA_DIR = "../datasets"
MODEL_DIR = "../models"
POCKETBASE_URL = "http://127.0.0.1:8090"
COLLECTION = "data"  # Use DATA collection
PB_TOKEN = os.environ.get("PB_TOKEN")  # Auth token must be set in env


def update_pocketbase_status(company, model_path):
    if not PB_TOKEN:
        print("[❌] PB_TOKEN environment variable not set!")
        return
    # Fetch record ID
    headers = {"Authorization": f"Bearer {PB_TOKEN}"}
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


def run_local_training(company):
    dataset_path = os.path.join(DATA_DIR, f"{company}.csv")
    model_path = os.path.join(MODEL_DIR, f"local_{company}.pkl")

    # Load and preprocess
    print(f"[📁] Loading dataset: {dataset_path}")
    X, y = load_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor(X_train)
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    weight_dict = dict(enumerate(class_weights))

    input_shape = X_train.shape[1]
    model = build_model(input_shape)

    print(f"[🧠] Training model for {company}...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=weight_dict)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump((model, preprocessor), model_path)
    print(f"[💾] Model saved to: {model_path}")

    # Update PocketBase
    update_pocketbase_status(company, model_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python client_runner.py <company>")
    else:
        run_local_training(sys.argv[1])
