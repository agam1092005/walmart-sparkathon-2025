import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from keras import layers, models
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('label_is_fraud', axis=1)
    y = data['label_is_fraud']
    return X, y


def create_preprocessor(X):
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

    return preprocessor


def build_model(input_shape, learning_rate=0.001, num_microbatches=32):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(1, activation='sigmoid'),
    ])

    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=1.1,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate
    )

    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    )

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


def load_global_preprocessor_from_pocketbase(pb_token, X_train=None, company=None):
    """
    Downloads and loads the global preprocessor from PocketBase 'global' collection.
    If no global record exists, builds it locally (requires X_train and company), uploads it, and then uses it.
    Returns the loaded preprocessor object.
    """
    import requests
    import joblib
    import io
    import hashlib
    import os
    POCKETBASE_URL = "http://127.0.0.1:8090"
    headers = {"Authorization": f"Bearer {pb_token}"}
    url = f"{POCKETBASE_URL}/api/collections/global/records"
    resp = requests.get(url, headers=headers)
    if not resp.ok:
        raise RuntimeError(f"Failed to fetch global record: {resp.text}")
    items = resp.json().get("items", [])
    if not items:
        if X_train is None or company is None:
            raise RuntimeError("No global record found in PocketBase and insufficient data to build one. Please provide X_train and company.")
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
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
        preprocessor.fit(X_train)
        preproc_path = "global_preprocessor.pkl"
        joblib.dump(preprocessor, preproc_path)
        with open(preproc_path, 'rb') as f:
            preproc_bytes = f.read()
            preproc_hash = hashlib.sha256(preproc_bytes).hexdigest()
        with open(preproc_path, 'rb') as f:
            files = {"global_preprocessor": f}
            data = {
                "preprocessor_version": preproc_hash,
                "source": company,
                "createdBy": "system"
            }
            create_res = requests.post(
                f"{POCKETBASE_URL}/api/collections/global/records",
                data=data,
                files=files,
                headers=headers
            )
        os.remove(preproc_path)
        if not create_res.ok:
            raise RuntimeError(f"Failed to create global record with preprocessor: {create_res.text}")
        print(f"[INFO] Created new global record and uploaded preprocessor to PocketBase with version/hash: {preproc_hash}")
        return preprocessor
    record = items[0]
    if "global_preprocessor" not in record or not record["global_preprocessor"]:
        raise RuntimeError("No global preprocessor found in PocketBase.")
    collection_id = record.get("collectionId", "global")
    record_id = record["id"]
    filename = record["global_preprocessor"]
    file_url = f"{POCKETBASE_URL}/api/files/{collection_id}/{record_id}/{filename}"
    file_resp = requests.get(file_url, headers=headers)
    if not file_resp.ok:
        raise RuntimeError(f"Failed to download global preprocessor: {file_resp.text}")
    preprocessor = joblib.load(io.BytesIO(file_resp.content))
    return preprocessor
