import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from keras import layers, models
import random

# Import DP optimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

# ------------------ Load Data ------------------
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('label_is_fraud', axis=1)
    y = data['label_is_fraud']
    return X, y

# ------------------ Preprocessing ------------------
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

# ------------------ Build Model ------------------
def build_model(input_shape, num_microbatches=32, learning_rate=0.001):
    num_layers = random.randint(1, 3)
    units_options = [32, 64, 128, 256]
    dropout_min, dropout_max = 0.1, 0.5
    use_batchnorm = [True, False]

    units_list = [random.choice(units_options) for _ in range(num_layers)]
    dropout_list = [round(random.uniform(dropout_min, dropout_max), 2) for _ in range(num_layers)]
    batchnorm_flags = [random.choice(use_batchnorm) for _ in range(num_layers)]
    learning_rate = 0.001  # fixed for DP optimizer

    print("🔧 Building model with parameters:")
    print(f"  Layers: {num_layers}")
    print(f"  Units per layer: {units_list}")
    print(f"  Dropout: {dropout_list}")
    print(f"  BatchNorm: {batchnorm_flags}")
    print(f"  Learning rate (DP): {learning_rate:.5f}")

    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    for i in range(num_layers):
        # Remove L1 regularization to avoid shape mismatch with DP optimizer
        model.add(layers.Dense(
            units_list[i],
            activation='relu'
            # kernel_regularizer removed for DP compatibility
        ))
        if batchnorm_flags[i]:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_list[i]))

    model.add(layers.Dense(1, activation='sigmoid'))

    # Define DP optimizer with corrected parameters
    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=1.1,
        num_microbatches=num_microbatches,  # Use the parameter passed to function
        learning_rate=learning_rate
    )

    # Set loss reduction to NONE for per-sample losses (required for DP)
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model