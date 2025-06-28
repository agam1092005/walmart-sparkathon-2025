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
