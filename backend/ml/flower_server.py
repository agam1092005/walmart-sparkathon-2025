import flwr as fl
import tensorflow as tf
import os
import pickle
import requests
from flwr.server import ServerConfig
from encryption import decrypt_bytes, get_shared_key
import numpy as np
import glob
from tensorflow import keras
import base64
import io
import hashlib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import pandas as pd
from client_helper import load_global_preprocessor_from_pocketbase
import shutil

POCKETBASE_URL = "http://127.0.0.1:8090"
MODEL_DIR = "../models"

PB_TOKEN = "PB_ADMIN_TOKEN"
if not PB_TOKEN:
    raise RuntimeError("PB_TOKEN environment variable not set. Please set PB_TOKEN to a valid PocketBase user token.")
headers = {"Authorization": f"Bearer {PB_TOKEN}"}

def decrypt_weights(ciphertext_str: str) -> list:
    if isinstance(ciphertext_str, str):
        ciphertext_bytes = base64.b64decode(ciphertext_str)
    else:
        ciphertext_bytes = ciphertext_str
    decrypted_bytes = decrypt_bytes(ciphertext_bytes, get_shared_key())
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
        if not isinstance(clients_str, str):
            clients_str = ""
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

    def download_client_weights(self):
        try:
            res = requests.get(f"{POCKETBASE_URL}/api/collections/global/records", headers=headers)
            res.raise_for_status()
            items = res.json().get("items", [])
            if not items or not items[0].get("client_weights"):
                return {}
            record = items[0]
            collection_id = record.get("collectionId", "global")
            record_id = record["id"]
            filename = record["client_weights"]
            url = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{filename}"
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            return pickle.load(io.BytesIO(resp.content))
        except Exception as e:
            print(f"[WARN] Could not download client_weights: {e}")
            return {}

    def upload_client_weights(self, client_weights_dict):
        try:
            res = requests.get(f"{POCKETBASE_URL}/api/collections/global/records", headers=headers)
            res.raise_for_status()
            items = res.json().get("items", [])
            if not items:
                print("[ERROR] No global record to upload client_weights to.")
                return
            global_id = items[0]["id"]
            with open("client_weights.pkl", "wb") as f:
                pickle.dump(client_weights_dict, f)
            with open("client_weights.pkl", "rb") as f:
                files = {"client_weights": f}
                data = {}
                update_res = requests.patch(
                    f"{POCKETBASE_URL}/api/collections/global/records/{global_id}",
                    data=data,
                    files=files,
                    headers=headers
                )
            os.remove("client_weights.pkl")
            if not update_res.ok:
                print(f"[ERROR] Failed to upload client_weights: {update_res.text}")
            else:
                print(f"[INFO] Uploaded updated client_weights to PocketBase.")
        except Exception as e:
            print(f"[ERROR] Exception during upload_client_weights: {e}")

    def aggregate_fit(self, server_round, results, failures):
        print(f"[DEBUG] aggregate_fit called for round {server_round} with {len(results)} results and {len(failures)} failures.")
        client_weights_dict = self.download_client_weights()
        decrypted_results = []
        contributing_clients = set()

        for idx, (client_proxy, fit_res) in enumerate(results):
            print(f"[DEBUG] Processing result {idx+1}/{len(results)}")
            try:
                encrypted_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)[0].item()
                print(f"[DEBUG] Encrypted weights received from client.")
                weights = decrypt_weights(encrypted_weights)
                print(f"[DEBUG] Weights decrypted successfully.")
                client_id = fit_res.metrics.get("client_id", f"client_{idx}")
                print(f"[DEBUG] Client ID: {client_id}, num_examples: {fit_res.num_examples}")
                client_weights_dict[client_id] = weights
                decrypted_results.append((weights, fit_res.num_examples))
                contributing_clients.add(client_id)
            except Exception as e:
                print(f"[ERROR] Failed to process client result: {e}")

        all_results = [(w, 1) for w in client_weights_dict.values()]  # Use equal weighting
        print(f"[DEBUG] Aggregating weights from {len(all_results)} clients (all stored).")
        def weighted_average(results):
            total_examples = sum(num_examples for _, num_examples in results)
            if total_examples == 0:
                return None
            avg = [np.zeros_like(w) for w in results[0][0]]
            for weights, num_examples in results:
                for i, w in enumerate(weights):
                    avg[i] += w * (num_examples / total_examples)
            return avg
        if all_results:
            aggregated_weights = weighted_average(all_results)
            if aggregated_weights is not None:
                print(f"[DEBUG] Aggregation successful. Setting model weights.")
                aggregated_weights_params = fl.common.ndarrays_to_parameters(aggregated_weights)
                self.model.set_weights(aggregated_weights)
                model_path = os.path.join(MODEL_DIR, f"global_model_round_{server_round}_clients_{len(client_weights_dict)}.h5")
                self.model.save(model_path)
                print(f"Global model saved: {model_path}")
                self.upload_client_weights(client_weights_dict)
                try:
                    with open(model_path, "rb") as f:
                        files = {"global_model": f}
                        data = {"round": str(server_round), "clients": str(len(client_weights_dict))}
                        res = requests.get(f"{POCKETBASE_URL}/api/collections/global/records", headers=headers)
                        items = res.json().get("items", [])
                        if not items:
                            raise RuntimeError("No global record exists in PocketBase to update with global_model.")
                        global_id = items[0]["id"]
                        patch_res = requests.patch(
                            f"{POCKETBASE_URL}/api/collections/global/records/{global_id}",
                            data=data,
                            files=files,
                            headers=headers
                        )
                    if patch_res.ok:
                        print(f"Global model uploaded to PocketBase.")
                    else:
                        print(f"Failed to upload global model: {patch_res.text}")
                except Exception as e:
                    print(f"[ERROR] Exception during upload to PocketBase: {e}")
                try:
                    for client_id in contributing_clients:
                        data_url = f"{POCKETBASE_URL}/api/collections/data/records?filter=company='{client_id}'"
                        data_res = requests.get(data_url, headers=headers)
                        data_items = data_res.json().get("items", [])
                        if data_items:
                            record_id = data_items[0]["id"]
                            patch_url = f"{POCKETBASE_URL}/api/collections/data/records/{record_id}"
                            patch_res = requests.patch(
                                patch_url,
                                data={"encrypted": "true"},
                                headers=headers
                            )
                            if patch_res.ok:
                                print(f"[INFO] Set encrypted=True for company {client_id} in data collection.")
                            else:
                                print(f"[WARN] Failed to set encrypted=True for company {client_id}: {patch_res.text}")
                        else:
                            print(f"[WARN] No data record found for company {client_id} to set encrypted=True.")
                except Exception as e:
                    print(f"[ERROR] Exception while setting encrypted=True: {e}")
                cleanup_models_dir()
                return aggregated_weights_params, {"num_clients": len(client_weights_dict)}
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

def cleanup_models_dir(models_dir=MODEL_DIR):
    for fname in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
                print(f"[CLEANUP] Deleted file: {fpath}")
            except Exception as e:
                print(f"[CLEANUP] Failed to delete file {fpath}: {e}")

def build_and_upload_global_preprocessor(pb_token, dataset_paths):
    """
    Build a global preprocessor from all datasets, save and upload to PocketBase.
    Ensures only one record is ever created in the global collection.
    """
    dfs = []
    for path in dataset_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    X = df_all.drop('label_is_fraud', axis=1)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    preprocessor.fit(X)
    preproc_path = os.path.join(MODEL_DIR, 'global_preprocessor.pkl')
    joblib.dump(preprocessor, preproc_path)
    with open(preproc_path, 'rb') as f:
        preproc_bytes = f.read()
        preproc_hash = hashlib.sha256(preproc_bytes).hexdigest()
    headers = {"Authorization": f"Bearer {pb_token}"}
    res = requests.get(f"{POCKETBASE_URL}/api/collections/global/records", headers=headers)
    items = res.json().get("items", [])
    if not items:
        with open(preproc_path, 'rb') as f:
            files = {"global_preprocessor": f}
            data = {"preprocessor_version": preproc_hash}
            create_res = requests.post(
                f"{POCKETBASE_URL}/api/collections/global/records",
                data=data,
                files=files,
                headers=headers
            )
        if not create_res.ok:
            raise RuntimeError(f"Failed to create global record with preprocessor: {create_res.text}")
        global_id = create_res.json()["id"]
        print(f"[INFO] Created new global record and uploaded preprocessor to PocketBase with version/hash: {preproc_hash}")
    else:
        global_id = items[0]["id"]
        with open(preproc_path, 'rb') as f:
            files = {"global_preprocessor": f}
            data = {"preprocessor_version": preproc_hash}
            update_res = requests.patch(
                f"{POCKETBASE_URL}/api/collections/global/records/{global_id}",
                data=data,
                files=files,
                headers=headers
            )
        if not update_res.ok:
            raise RuntimeError(f"Failed to upload global preprocessor: {update_res.text}")
        print(f"[INFO] Uploaded global preprocessor to PocketBase with version/hash: {preproc_hash}")
    cleanup_models_dir()
    return preproc_hash

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        preprocessor = None
        try:
            preprocessor = load_global_preprocessor_from_pocketbase(PB_TOKEN)
        except Exception as e:
            print(f"[WARN] Could not load global preprocessor: {e}")
        if preprocessor is not None:
            dummy = pd.DataFrame([{col: 0 for col in preprocessor.feature_names_in_}])
            input_shape = preprocessor.transform(dummy).shape[1]
            print(f"[INFO] Inferred input_shape from global preprocessor: {input_shape}")
        else:
            dataset_files = sorted(glob.glob(os.path.join("../datasets", "*.csv")))
            if not dataset_files:
                raise RuntimeError("No dataset found to infer input_shape. Cannot proceed.")
            df = pd.read_csv(dataset_files[0])
            X = df.drop('label_is_fraud', axis=1)
            from client_helper import create_preprocessor
            temp_preproc = create_preprocessor(X)
            temp_preproc.fit(X)
            input_shape = temp_preproc.transform(X[:1]).shape[1]
            print(f"[INFO] Inferred input_shape from dataset: {input_shape}")
    except Exception as e:
        print(f"[ERROR] Could not infer input_shape: {e}. Aborting.")
        raise

    NUM_CLIENTS = get_num_clients_from_pocketbase()
    print(f"Found {NUM_CLIENTS} unique clients in PocketBase global record.")

    model = build_global_model(input_shape)
    strategy = SecureFedAvg(
        model=model,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        on_fit_config_fn=lambda rnd: {"rnd": rnd}
    )

    print("Starting Flower server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=1)
    )

