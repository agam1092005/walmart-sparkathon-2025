# client_a.py - Fixed client with robust training

import flwr as fl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from model import load_data, create_preprocessor, create_enhanced_model
import tensorflow as tf
import time
import socket
import os

# Configure TensorFlow for better stability
tf.config.run_functions_eagerly(False)
tf.get_logger().setLevel('ERROR')  # Reduce TF logging

def build_enhanced_deterministic_model(input_shape, use_batch_norm=True, dropout_rate=0.3):
    """Build the exact same enhanced deterministic model as the server"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # First dense block
        tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
        tf.keras.layers.BatchNormalization(name='batch_norm_1') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_1'),
        
        # Second dense block
        tf.keras.layers.Dense(128, activation='relu', name='dense_2'),
        tf.keras.layers.BatchNormalization(name='batch_norm_2') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Third dense block
        tf.keras.layers.Dense(64, activation='relu', name='dense_3'),
        tf.keras.layers.BatchNormalization(name='batch_norm_3') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_3'),
        
        # Fourth dense block
        tf.keras.layers.Dense(32, activation='relu', name='dense_4'),
        tf.keras.layers.BatchNormalization(name='batch_norm_4') if use_batch_norm else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.Dropout(dropout_rate / 2, name='dropout_4'),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ],
        run_eagerly=False  # Ensure graph mode
    )
    
    print(f"🏗️ Built enhanced deterministic model:")
    print(f"   📏 Input shape: {input_shape}")
    print(f"   🔢 Total parameters: {model.count_params()}")
    print(f"   📊 Weight arrays: {len(model.get_weights())}")
    print(f"   🧱 Batch normalization: {use_batch_norm}")
    
    return model

class EnhancedRetailerClient(fl.client.NumPyClient):
    def __init__(self, data_path, client_name="Enhanced Client"):
        self.client_name = client_name
        self.data_path = data_path
        self.model_initialized = False
        self.input_shape = None
        self.preprocessor = None
        self.class_weights = None
        self._setup_data()
    
    def _setup_data(self):
        """Setup data and preprocessing with consistent encoding"""
        print(f"🔧 {self.client_name}: Setting up data with enhanced preprocessing...")

        try:
            # Load data
            X, y = load_data(self.data_path)
            print(f"📊 {self.client_name}: Loaded data - Shape: {X.shape}, Labels: {len(np.unique(y))} classes")

            # FIXED: Check for empty data
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty dataset loaded")

            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            print(f"📈 {self.client_name}: Class distribution: {dict(zip(unique, counts))}")

            #  Ensure minimum class sizes for stratification
            min_class_size = min(counts)
            if min_class_size < 2:
                print(f"⚠️ {self.client_name}: Very small class size detected, using random split")
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

            #  preprocessing with global mappings
            print(f"🔄 {self.client_name}: Applying enhanced preprocessing...")
            self.preprocessor = create_preprocessor(self.X_train, use_global_mappings=True)

            # Fit and transform data with error handling
            try:
                self.X_train_processed = self.preprocessor.fit_transform(self.X_train).astype(np.float32)
                self.X_test_processed = self.preprocessor.transform(self.X_test).astype(np.float32)
            except Exception as e:
                print(f"⚠️ {self.client_name}: Preprocessing error: {e}")
                print("🔄 Falling back to simple preprocessing...")
                # Fallback to simple preprocessing
                from sklearn.preprocessing import StandardScaler
                from sklearn.impute import SimpleImputer
                
                # Select only numeric columns as fallback
                numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
                self.X_train_simple = self.X_train[numeric_cols]
                self.X_test_simple = self.X_test[numeric_cols]
                
                imputer = SimpleImputer(strategy='mean')
                scaler = StandardScaler()
                
                self.X_train_processed = scaler.fit_transform(
                    imputer.fit_transform(self.X_train_simple)
                ).astype(np.float32)
                self.X_test_processed = scaler.transform(
                    imputer.transform(self.X_test_simple)
                ).astype(np.float32)

            # FIXED: Validate processed data
            if np.any(np.isnan(self.X_train_processed)) or np.any(np.isinf(self.X_train_processed)):
                print(f"⚠️ {self.client_name}: Invalid values detected in training data")
                # Replace NaN/inf with 0
                self.X_train_processed = np.where(
                    np.isfinite(self.X_train_processed), 
                    self.X_train_processed, 
                    0
                ).astype(np.float32)
                
            if np.any(np.isnan(self.X_test_processed)) or np.any(np.isinf(self.X_test_processed)):
                print(f"⚠️ {self.client_name}: Invalid values detected in test data")
                self.X_test_processed = np.where(
                    np.isfinite(self.X_test_processed), 
                    self.X_test_processed, 
                    0
                ).astype(np.float32)

            print(f" {self.client_name}: After preprocessing:")
            print(f"    Training shape: {self.X_train_processed.shape}")
            print(f"    Test shape: {self.X_test_processed.shape}")
            print(f"    Training data range: [{self.X_train_processed.min():.3f}, {self.X_train_processed.max():.3f}]")

            # Store input shape
            self.input_shape = self.X_train_processed.shape[1]

            #  Robust class weight computation
            try:
                # Ensure labels are in correct format
                y_train_clean = self.y_train.astype(int)
                unique_classes = np.unique(y_train_clean)
                
                if len(unique_classes) >= 2:
                    class_weights = class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=unique_classes,
                        y=y_train_clean
                    )
                    self.class_weights = dict(zip(unique_classes, class_weights))
                    
                    # Cap extreme class weights to prevent training instability
                    max_weight = 10.0
                    for k, v in self.class_weights.items():
                        if v > max_weight:
                            print(f"⚠️ {self.client_name}: Capping class weight for class {k} from {v:.2f} to {max_weight}")
                            self.class_weights[k] = max_weight

                    print(f"⚖️ {self.client_name}: Class weights: {self.class_weights}")
                else:
                    print(f"⚠️ {self.client_name}: Only one class found, not using class weights")
                    self.class_weights = None

            except Exception as e:
                print(f"⚠️ {self.client_name}: Could not compute class weights: {e}")
                self.class_weights = None

            # Build enhanced model
            self.model = build_enhanced_deterministic_model(
                input_shape=self.input_shape,
                use_batch_norm=True,
                dropout_rate=0.3
            )

            # Test model with a small batch to ensure it works
            print(f" {self.client_name}: Testing model with sample data...")
            test_batch = self.X_train_processed[:min(10, len(self.X_train_processed))]
            test_output = self.model(test_batch, training=False)
            print(f" {self.client_name}: Model test successful, output shape: {test_output.shape}")

            print(f" {self.client_name}: Enhanced setup complete!")
            print(f"    Training samples: {len(self.X_train)} (Fraud: {sum(self.y_train)}, Normal: {len(self.y_train) - sum(self.y_train)})")
            print(f"    Test samples: {len(self.X_test)} (Fraud: {sum(self.y_test)}, Normal: {len(self.y_test) - sum(self.y_test)})")
            print(f"    Final input shape: {self.input_shape}")
            print(f"    Model parameters: {self.model.count_params()}")

        except Exception as e:
            print(f"❌ {self.client_name}: Failed to setup data: {e}")
            raise

    def get_parameters(self, config):
        """Get model parameters"""
        if self.model is None:
            print(f"⚠️ {self.client_name}: Model is None, returning empty parameters")
            return []
        return self.model.get_weights()

    def set_parameters(self, parameters):
        """Set model parameters with enhanced validation"""
        if not parameters:
            print(f"⚠️ {self.client_name}: Received empty parameters")
            return
        
        try:
            # Ensure model exists
            if self.model is None:
                print(f" {self.client_name}: Creating new enhanced model...")
                self.model = build_enhanced_deterministic_model(
                    input_shape=self.input_shape,
                    use_batch_norm=True,
                    dropout_rate=0.3
                )
            
            # Set parameters
            self.model.set_weights(parameters)
            self.model_initialized = True
            print(f" {self.client_name}: Parameters set successfully")
            
        except Exception as e:
            print(f"❌ {self.client_name}: Failed to set parameters: {e}")
            self.model_initialized = False

    def fit(self, parameters, config):
        """Enhanced training with robust error handling"""
        server_round = config.get('server_round', '?')
        print(f"\n🏋️ {self.client_name}: Starting enhanced training (Round {server_round})...")
        
        try:
            # Set global model parameters
            self.set_parameters(parameters)
            
            if not self.model_initialized:
                return self.get_parameters(config), len(self.X_train), {
                    "train_loss": 0.0, 
                    "train_accuracy": 0.0,
                    "error": "Model not initialized"
                }
            
            # Get training configuration
            epochs = config.get("epochs", 2)
            learning_rate = config.get("learning_rate", 0.001)
            batch_size = min(config.get("batch_size", 32), len(self.X_train) // 4)  # FIXED: Ensure reasonable batch size
            use_class_weights = config.get("use_class_weights", True)
            
            print(f" {self.client_name}: Training configuration:")
            print(f"    Epochs: {epochs}")
            print(f"    Learning Rate: {learning_rate}")
            print(f"    Batch Size: {batch_size}")
            print(f"    Class Weights: {use_class_weights}")
            
            # FIXED: Create fresh optimizer to avoid state issues
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ],
                run_eagerly=False
            )
            
            # FIXED: Prepare training data with proper validation
            X_train = self.X_train_processed.copy()
            #y_train = self.y_train.astype(np.float32)
            y_train = np.asarray(self.y_train).astype(np.float32)
            
            # Ensure data is valid
            if len(X_train) == 0 or len(y_train) == 0:
                raise ValueError("Training data is empty")
            
            # FIXED: Simple train/validation split to avoid validation_split issues
            val_split = 0.15
            val_size = max(1, int(len(X_train) * val_split))
            
            # Split manually to have more control
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            X_train_split = X_train[train_indices]
            y_train_split = y_train[train_indices]
            X_val_split = X_train[val_indices]
            y_val_split = y_train[val_indices]
            
            print(f"📊 {self.client_name}: Split - Train: {len(X_train_split)}, Val: {len(X_val_split)}")
            
            # Prepare training arguments
            fit_kwargs = {
                "epochs": epochs,
                "batch_size": batch_size,
                "verbose": 1,
                "validation_data": (X_val_split, y_val_split),
                "shuffle": True
            }
            
            # Add class weights if available and requested
            if use_class_weights and self.class_weights is not None:
                fit_kwargs["class_weight"] = self.class_weights
                print(f"⚖️ {self.client_name}: Using class weights: {self.class_weights}")
            
            # FIXED: Simplified callbacks to avoid issues
            callbacks = []
            
            # Only add callbacks if we have enough data
            if len(X_train_split) > 100:
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True,
                        verbose=0
                    )
                ]
            
            if callbacks:
                fit_kwargs["callbacks"] = callbacks
            
            # FIXED: Train the model with better error handling
            print(f" {self.client_name}: Starting training...")
            
            # Test model before training
            test_pred = self.model.predict(X_train_split[:min(5, len(X_train_split))], verbose=0)
            print(f" {self.client_name}: Pre-training test successful")
            
            history = self.model.fit(
                X_train_split,
                y_train_split,
                **fit_kwargs
            )
            
            # FIXED: Safe metric extraction
            def safe_get_metric(hist_dict, key, default=0.0):
                try:
                    values = hist_dict.get(key, [default])
                    if values and len(values) > 0:
                        val = float(values[-1])
                        return val if np.isfinite(val) else default
                    return default
                except:
                    return default
            
            train_loss = safe_get_metric(history.history, "loss")
            train_accuracy = safe_get_metric(history.history, "accuracy")
            train_precision = safe_get_metric(history.history, "precision")
            train_recall = safe_get_metric(history.history, "recall")
            
            val_loss = safe_get_metric(history.history, "val_loss")
            val_accuracy = safe_get_metric(history.history, "val_accuracy")
            val_precision = safe_get_metric(history.history, "val_precision")
            val_recall = safe_get_metric(history.history, "val_recall")
            
            # Calculate F1 scores safely
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
            
            print(f" {self.client_name}: Training complete!")
            print(f"    Training   - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}")
            print(f"    Validation - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}")
            
            return self.get_parameters(config), len(self.X_train), {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            }
            
        except Exception as e:
            print(f"❌ {self.client_name}: Training failed: {e}")
            print(f" Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            return self.get_parameters(config), len(self.X_train), {
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "error": str(e)
            }

    def evaluate(self, parameters, config):
        """Enhanced evaluation with detailed metrics"""
        print(f"🔍 {self.client_name}: Enhanced model evaluation...")
        
        try:
            # Set global model parameters
            self.set_parameters(parameters)
            
            if not self.model_initialized:
                return 0.0, len(self.X_test), {"accuracy": 0.0, "error": "Model not initialized"}
            
            # Get evaluation configuration
            batch_size = min(config.get("batch_size", 64), len(self.X_test))
            
            # FIXED: Evaluate with proper error handling
            results = self.model.evaluate(
                self.X_test_processed, 
                self.y_test.astype(np.float32), 
                batch_size=batch_size,
                verbose=0,
                return_dict=True
            )
            
            # FIXED: Safe metric extraction
            def safe_extract(results_dict, key, default=0.0):
                try:
                    val = results_dict.get(key, default)
                    return float(val) if np.isfinite(float(val)) else default
                except:
                    return default
            
            loss = safe_extract(results, 'loss')
            accuracy = safe_extract(results, 'accuracy')
            precision = safe_extract(results, 'precision')
            recall = safe_extract(results, 'recall')
            
            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"📊 {self.client_name}: Evaluation Results:")
            print(f"   📉 Loss: {loss:.4f}")
            print(f"   📈 Accuracy: {accuracy:.4f}")
            print(f"   🎯 Precision: {precision:.4f}")
            print(f"   🔍 Recall: {recall:.4f}")
            print(f"   🏆 F1 Score: {f1_score:.4f}")
            
            return float(loss), len(self.X_test), {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score)
            }
            
        except Exception as e:
            print(f"❌ {self.client_name}: Evaluation failed: {e}")
            return 0.0, len(self.X_test), {"accuracy": 0.0, "error": str(e)}

def check_prerequisites(data_path):
    """Check if all prerequisites are met"""
    print(" Checking prerequisites...")
    
    issues = []
    
    # Check data file
    if not os.path.exists(data_path):
        issues.append(f"Data file not found: {data_path}")
    else:
        # Check if data file is readable and not empty
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            if len(df) == 0:
                issues.append(f"Data file is empty: {data_path}")
            else:
                print(f"    Data file: {data_path} ({len(df)} rows)")
        except Exception as e:
            issues.append(f"Cannot read data file: {e}")
    
    # Check global mappings (optional)
    if not os.path.exists('global_categorical_mappings.pkl'):
        print("   ⚠️ Global categorical mappings not found (will use fallback)")
    else:
        print("    Global categorical mappings found")
    
    # Check TensorFlow
    try:
        print(f"    TensorFlow: {tf.__version__}")
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"    GPU Available: {len(gpus)} device(s)")
        else:
            print("   ℹ️ No GPU detected, using CPU")
    except Exception as e:
        issues.append(f"TensorFlow issue: {e}")
    
    if issues:
        print("❌ Prerequisites not met:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(" All prerequisites met!")
        return True

def check_server_availability(server_address="localhost:8080", timeout=5):
    """Check if server is available"""
    try:
        host, port = server_address.split(":")
        port = int(port)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        return result == 0
    except Exception:
        return False

def start_enhanced_client_with_retry(client, server_address="localhost:8080", max_retries=3):
    """Start enhanced client with retry logic"""
    
    # Check server availability
    if not check_server_availability(server_address):
        print(f"❌ Server not available at {server_address}")
        print(" Make sure the server is running: python enhanced_deterministic_server.py")
        return
    
    for attempt in range(max_retries):
        try:
            print(f" Starting enhanced client (attempt {attempt + 1}/{max_retries})...")
            print(f"    Client: {client.client_name}")
            print(f"    Input shape: {client.input_shape}")
            print(f"    Model parameters: {client.model.count_params()}")
            
            # Start the client
            fl.client.start_client(
                server_address=server_address,
                client=client.to_client()
            )
            
            print(" Enhanced client completed successfully!")
            break
            
        except Exception as e:
            print(f"❌ Connection failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting 10 seconds before retry...")
                time.sleep(10)
            else:
                print("❌ Max retries reached. Check server status.")

# Enhanced client startup
if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'cleaned_dataset2.txt'  # Change this for different clients
    CLIENT_NAME = "Enhanced Retailer B"  # Change this for different clients
    SERVER_ADDRESS = "localhost:8080"
    
    print("🔧 Initializing Fixed Enhanced Federated Learning Client...")
    print(f"    Data: {DATA_PATH}")
    print(f"    Name: {CLIENT_NAME}")
    print(f"    Server: {SERVER_ADDRESS}")
    
    try:
        # Check prerequisites
        if not check_prerequisites(DATA_PATH):
            print("\n Setup Instructions:")
            print("   1. Ensure data file exists and is readable")
            print("   2. Optional: Run python model1.py (for global mappings)")
            print("   3. Start server: python enhanced_deterministic_server.py")
            exit(1)
        
        # Create enhanced client
        client = EnhancedRetailerClient(
            data_path=DATA_PATH,
            client_name=CLIENT_NAME
        )
        
        print(" Connecting to enhanced federated learning server...")
        start_enhanced_client_with_retry(client, server_address=SERVER_ADDRESS)
        
    except KeyboardInterrupt:
        print("🛑 Enhanced client stopped by user")
    except Exception as e:
        print(f"❌ Enhanced client error: {e}")
        import traceback
        traceback.print_exc()
        print("\n Troubleshooting:")
        print("   1. Check data file exists and is readable")
        print("   2. Start server: python enhanced_deterministic_server.py")
        print("   3. Verify network connectivity")