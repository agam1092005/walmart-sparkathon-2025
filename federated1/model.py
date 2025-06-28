import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
import tensorflow as tf
from keras import layers, regularizers, models
import random
import pickle
import os

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer


# ------------------ Load Data ------------------
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['label_is_fraud'])
    X = data.drop('label_is_fraud', axis=1)
    y = data['label_is_fraud'].astype(int)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y


# ------------------ Global Categorical Mapping ------------------
def create_global_categorical_mappings(datasets_info):
    print("🔧 Creating global categorical mappings...")
    global_categorical_values = {}

    for file_path, client_name in datasets_info:
        print(f"   📊 Processing {client_name} data from {file_path}")
        try:
            X, _ = load_data(file_path)
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                if col not in global_categorical_values:
                    global_categorical_values[col] = set()
                unique_vals = X[col].dropna().unique()
                global_categorical_values[col].update(unique_vals)
        except Exception as e:
            print(f"   ⚠️ Warning: Could not process {file_path}: {e}")

    for col in global_categorical_values:
        global_categorical_values[col] = sorted(list(global_categorical_values[col]))
        print(f"    {col}: {len(global_categorical_values[col])} unique values")

    with open('global_categorical_mappings.pkl', 'wb') as f:
        pickle.dump(global_categorical_values, f)

    print(" Global categorical mappings created and saved!")
    return global_categorical_values


def load_global_categorical_mappings():
    try:
        with open('global_categorical_mappings.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("⚠️ Global categorical mappings file not found!")
        return None


# ------------------ Custom Encoder ------------------
class ConsistentOrdinalEncoder:
    def __init__(self, global_mappings=None):
        self.global_mappings = global_mappings
        self.encoders = {}
        self.is_fitted = False

    def fit(self, X, y=None):  
        if self.global_mappings is None:
            raise ValueError("Global mappings not provided!")

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            if col in self.global_mappings:
                categories = ['missing'] + self.global_mappings[col]
                self.encoders[col] = {cat: idx for idx, cat in enumerate(categories)}
                print(f"    {col}: {len(categories)} categories (0-{len(categories)-1})")
            else:
                print(f"   ⚠️ Warning: {col} not found in global mappings")

        self.is_fitted = True
        return self

    def transform(self, X, y=None):  
        if not self.is_fitted:
            raise ValueError("Encoder not fitted yet!")

        X_encoded = X.copy()
        for col in self.encoders:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].fillna('missing')
                X_encoded[col] = X_encoded[col].map(self.encoders[col]).fillna(0)
                X_encoded[col] = X_encoded[col].astype(int)

        return X_encoded

    def fit_transform(self, X, y=None):  
        return self.fit(X).transform(X)


# ------------------ Preprocessor ------------------
def create_preprocessor(X, use_global_mappings=True):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f" Creating preprocessor:")
    print(f"    Numerical columns: {len(numerical_cols)}")
    print(f"    Categorical columns: {len(categorical_cols)}")

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    if use_global_mappings and categorical_cols:
        global_mappings = load_global_categorical_mappings()
        if global_mappings is not None:
            print("    Using global categorical mappings for consistency")
            categorical_pipeline = Pipeline([
                ('consistent_encoder', ConsistentOrdinalEncoder(global_mappings)),
                ('scaler', StandardScaler())
            ])
        else:
            print("   ⚠️ Global mappings not found, using simple ordinal encoding")
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ('scaler', StandardScaler())
            ])
    else:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ('scaler', StandardScaler())
        ])

    transformers = [('num', numerical_pipeline, numerical_cols)]
    if categorical_cols:
        transformers.append(('cat', categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


# ------------------ Model ------------------
def create_enhanced_model(input_shape, use_batch_norm=True, dropout_rate=0.3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
        tf.keras.layers.BatchNormalization(name='batch_norm_1') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_1'),

        tf.keras.layers.Dense(128, activation='relu', name='dense_2'),
        tf.keras.layers.BatchNormalization(name='batch_norm_2') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_2'),

        tf.keras.layers.Dense(64, activation='relu', name='dense_3'),
        tf.keras.layers.BatchNormalization(name='batch_norm_3') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_3'),

        tf.keras.layers.Dense(32, activation='relu', name='dense_4'),
        tf.keras.layers.BatchNormalization(name='batch_norm_4') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Dropout(dropout_rate / 2, name='dropout_4'),

        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    print(f" Enhanced model created:")
    print(f"    Input shape: {input_shape}")
    print(f"    Total parameters: {model.count_params()}")
    print(f"    Batch normalization: {use_batch_norm}")
    print(f"    Dropout rate: {dropout_rate}")

    return model


# ------------------ Federated Setup ------------------
def setup_federated_preprocessing(datasets_info):
    print(" Setting up federated learning preprocessing...")
    global_mappings = create_global_categorical_mappings(datasets_info)

    print("\n Testing preprocessing consistency...")
    input_shapes = []

    for file_path, client_name in datasets_info:
        try:
            print(f"    Testing {client_name}...")
            X, y = load_data(file_path)

            preprocessor = create_preprocessor(X, use_global_mappings=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_processed = preprocessor.fit_transform(X_train)
            input_shape = X_processed.shape[1]
            input_shapes.append(input_shape)

            print(f"    {client_name}: Input shape = {input_shape}")

        except Exception as e:
            print(f"   ❌ {client_name}: Error = {e}")

    if len(set(input_shapes)) == 1:
        print(f"✅ All clients have consistent input shape: {input_shapes[0]}")
        return input_shapes[0]
    else:
        print(f"❌ Inconsistent input shapes: {input_shapes}")
        return None


def get_fixed_input_shape():
    try:
        global_mappings = load_global_categorical_mappings()
        if global_mappings is None:
            return None

        total_categorical_features = sum(len(values) for values in global_mappings.values())
        estimated_numerical_features = 10  # Adjust if needed
        estimated_shape = total_categorical_features + estimated_numerical_features

        print(f"📏 Estimated input shape: {estimated_shape}")
        return estimated_shape

    except Exception as e:
        print(f"❌ Error calculating input shape: {e}")
        return None


# ------------------ Main ------------------
if __name__ == "__main__":
    print("🔧 Testing federated learning preprocessing setup...")

    datasets_info = [
        ('cleaned_dataset1.txt', 'Client A'),
        ('cleaned_dataset2.txt', 'Client B'),  # Add more datasets if needed
    ]

    try:
        fixed_input_shape = setup_federated_preprocessing(datasets_info)

        if fixed_input_shape:
            print(f"\n Federated learning setup complete!")
            print(f"    Fixed input shape: {fixed_input_shape}")
            print(f"    Global mappings saved to: global_categorical_mappings.pkl")

            model = create_enhanced_model(fixed_input_shape)
            print(f"    Model created successfully with {model.count_params()} parameters")

        else:
            print(" Setup failed - inconsistent input shapes")

    except Exception as e:
        print(f" Setup error: {e}")
