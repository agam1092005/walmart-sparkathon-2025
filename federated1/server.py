                                          # enhanced_server.py
import flwr as fl
import tensorflow as tf
from model import load_data, create_preprocessor, create_enhanced_model, load_global_categorical_mappings, get_fixed_input_shape
import numpy as np
from sklearn.model_selection import train_test_split
import os

def build_enhanced_deterministic_model(input_shape, use_batch_norm=True, dropout_rate=0.3):
    """Build an enhanced deterministic model with batch normalization"""
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f" Built enhanced deterministic model:")
    print(f"    Input shape: {input_shape}")
    print(f"    Total parameters: {model.count_params()}")
    print(f"    Weight arrays: {len(model.get_weights())}")
    print(f"    Batch normalization: {use_batch_norm}")
    
    return model

class EnhancedFederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, expected_clients=2, **kwargs):
        super().__init__(**kwargs)
        self.global_model = None
        self.input_shape = None
        self.round_counter = 0
        self.expected_clients = expected_clients
        self.client_data_info = {}
        self.shape_determined = False
    
    def _determine_input_shape_from_client(self, parameters):
        """Dynamically determine input shape from first client's model"""
        try:
            weights = fl.common.parameters_to_ndarrays(parameters)
            if len(weights) > 0:
                # First weight matrix shape: (input_features, first_layer_neurons)
                first_layer_weights = weights[0]
                input_features = first_layer_weights.shape[0]
                print(f"🔍 Detected input shape from client: {input_features}")
                return input_features
        except Exception as e:
            print(f"❌ Error detecting input shape from client: {e}")
        return None
    
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters - defer until first client connects"""
        print(" Server: Deferring model initialization until first client connects...")
        # Return None to defer initialization
        return None
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results from multiple clients"""
        print(f"\n🔄 Server Round {server_round}")
        print(f"📊 Received results from {len(results)} clients")
        
        if failures:
            print(f"❌ Failures: {len(failures)}")
            for failure in failures:
                print(f"   - {failure}")
        
        # Initialize global model on first round using client's shape
        if not self.shape_determined and len(results) > 0:
            print("🔍 Determining input shape from first client...")
            first_client_params = results[0][1].parameters
            detected_shape = self._determine_input_shape_from_client(first_client_params)
            
            if detected_shape:
                self.input_shape = detected_shape
                print(f" Input shape determined: {self.input_shape}")
                
                # Create global model with correct shape
                self.global_model = build_enhanced_deterministic_model(
                    input_shape=self.input_shape,
                    use_batch_norm=True,
                    dropout_rate=0.3
                )
                self.shape_determined = True
                
                # Set initial weights from first client (optional)
                try:
                    initial_weights = fl.common.parameters_to_ndarrays(first_client_params)
                    self.global_model.set_weights(initial_weights)
                    print("✅ Global model initialized with first client's weights")
                except Exception as e:
                    print(f"⚠️ Could not set initial weights: {e}")
            else:
                print("❌ Could not determine input shape from client")
                return None
        
        # Verify all clients have consistent shapes
        for i, (client, fit_res) in enumerate(results):
            try:
                client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
                if len(client_weights) > 0:
                    client_input_shape = client_weights[0].shape[0]
                    if client_input_shape != self.input_shape:
                        print(f"❌ Client {i+1} shape mismatch: expected {self.input_shape}, got {client_input_shape}")
                        return None
            except Exception as e:
                print(f"⚠️ Error checking client {i+1} shape: {e}")
        
        # Print detailed client metrics
        total_samples = 0
        weighted_metrics = {'train_loss': 0, 'train_accuracy': 0, 'val_loss': 0, 'val_accuracy': 0}
        
        for i, (client, fit_res) in enumerate(results):
            metrics = fit_res.metrics
            samples = fit_res.num_examples
            total_samples += samples
            
            print(f" Client {i+1} metrics:")
            print(f"   - Samples: {samples}")
            
            # Safe metric extraction and weighting
            try:
                for metric_name in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']:
                    value = metrics.get(metric_name, 0)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        weighted_metrics[metric_name] += value * samples
                        print(f"   - {metric_name.replace('_', ' ').title()}: {value:.4f}")
                    else:
                        print(f"   - {metric_name.replace('_', ' ').title()}: N/A")
                        
            except Exception as e:
                print(f"   - Error processing metrics: {e}")
                print(f"   - Raw metrics: {metrics}")
        
        # Calculate weighted averages
        if total_samples > 0:
            print(f"\n Weighted Average Metrics (across {len(results)} clients):")
            for metric_name, total_weighted_value in weighted_metrics.items():
                avg_value = total_weighted_value / total_samples
                if avg_value > 0:
                    print(f"   - {metric_name.replace('_', ' ').title()}: {avg_value:.4f}")
        
        # Perform aggregation
        try:
            aggregated_result = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_result is not None:
                aggregated_weights, aggregated_metrics = aggregated_result
                print(" Model aggregation successful")
                
                # Update global model
                if self.global_model is not None:
                    try:
                        weights_as_ndarrays = fl.common.parameters_to_ndarrays(aggregated_weights)
                        
                        # Verify shapes before setting weights
                        current_weights = self.global_model.get_weights()
                        if len(weights_as_ndarrays) == len(current_weights):
                            shape_match = all(
                                w1.shape == w2.shape 
                                for w1, w2 in zip(weights_as_ndarrays, current_weights)
                            )
                            
                            if shape_match:
                                self.global_model.set_weights(weights_as_ndarrays)
                                print(" Global model weights updated successfully")
                                
                                # Save model
                                model_path = f"global_model_round_{server_round}_clients_{len(results)}.h5"
                                self.global_model.save(model_path)
                                print(f" Model saved: {model_path}")
                            else:
                                print("❌ Weight shape mismatch detected, skipping update")
                                for i, (w1, w2) in enumerate(zip(weights_as_ndarrays, current_weights)):
                                    if w1.shape != w2.shape:
                                        print(f"   Layer {i}: expected {w2.shape}, got {w1.shape}")
                        else:
                            print(f"❌ Weight count mismatch: expected {len(current_weights)}, got {len(weights_as_ndarrays)}")
                        
                    except Exception as e:
                        print(f"⚠️ Could not update/save model: {e}")
                
                # Store aggregated metrics
                if aggregated_metrics:
                    print(f"📊 Server aggregated metrics: {aggregated_metrics}")
            
            return aggregated_result
            
        except Exception as e:
            print(f"❌ Error during aggregation: {e}")
            print(f"📊 Results info: {[(type(r[1]), hasattr(r[1], 'parameters')) for r in results]}")
            return None
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from multiple clients"""
        print(f"\n🔍 Server: Aggregating evaluation results for round {server_round}")
        
        if failures:
            print(f"❌ Evaluation failures: {len(failures)}")
            for failure in failures:
                print(f"   - {failure}")
        
        # Print detailed client evaluation metrics
        total_examples = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        weighted_precision = 0.0
        weighted_recall = 0.0
        
        print(f"📊 Client evaluation results:")
        for i, (client, eval_res) in enumerate(results):
            metrics = eval_res.metrics
            loss = eval_res.loss
            num_examples = eval_res.num_examples
            
            # Extract metrics safely
            accuracy = metrics.get('accuracy', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            
            print(f"   Client {i+1}:")
            print(f"      📊 Samples: {num_examples}")
            print(f"      📉 Loss: {loss:.4f}")
            print(f"      📈 Accuracy: {accuracy:.4f}")
            if precision > 0:
                print(f"      🎯 Precision: {precision:.4f}")
            if recall > 0:
                print(f"      🔍 Recall: {recall:.4f}")
            
            # Accumulate weighted metrics
            total_examples += num_examples
            weighted_loss += loss * num_examples
            weighted_accuracy += accuracy * num_examples
            weighted_precision += precision * num_examples
            weighted_recall += recall * num_examples
        
        # Calculate and display weighted averages
        if total_examples > 0:
            avg_loss = weighted_loss / total_examples
            avg_accuracy = weighted_accuracy / total_examples
            avg_precision = weighted_precision / total_examples
            avg_recall = weighted_recall / total_examples
            
            print(f"\n📈 Weighted Average Performance:")
            print(f"   📉 Loss: {avg_loss:.4f}")
            print(f"   📈 Accuracy: {avg_accuracy:.4f}")
            if avg_precision > 0:
                print(f"   🎯 Precision: {avg_precision:.4f}")
            if avg_recall > 0:
                print(f"   🔍 Recall: {avg_recall:.4f}")
            
            # Calculate F1 score if we have precision and recall
            if avg_precision > 0 and avg_recall > 0:
                f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                print(f"   🏆 F1 Score: {f1_score:.4f}")
        
        return super().aggregate_evaluate(server_round, results, failures)
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training with adaptive settings"""
        print(f"\n🔧 Server: Configuring training round {server_round}")
        
        # Adaptive learning rate schedule
        if server_round <= 3:
            learning_rate = 0.001
            epochs = 3
        elif server_round <= 7:
            learning_rate = 0.0005
            epochs = 2
        else:
            learning_rate = 0.0001
            epochs = 1
        
        config = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "server_round": server_round,
            "batch_size": 32,
            "use_class_weights": True
        }
        
        print(f" Training config: {config}")
        
        return super().configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation"""
        config = {
            "server_round": server_round,
            "batch_size": 64
        }
        
        return super().configure_evaluate(server_round, parameters, client_manager)

def start_enhanced_server(expected_clients=2, num_rounds=15):
    """Start the enhanced federated learning server with dynamic shape detection"""
    print(" Starting Enhanced Federated Learning Server with Dynamic Shape Detection...")
    print(f" Server configuration:")
    print(f"    Address: localhost:8080")
    print(f"    Expected clients: {expected_clients}")
    print(f"    Training rounds: {num_rounds}")
    print(f"    Dynamic model architecture based on client data")
    print(f"    Automatic shape detection from first client")
    print(" Waiting for clients to connect...")
    
    # Create enhanced strategy with dynamic shape detection
    strategy = EnhancedFederatedStrategy(
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        expected_clients=expected_clients
    )
    
    # Start server
    try:
        print(" Starting Flower server...")
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )
        print("🏁 Federated learning completed successfully!")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    EXPECTED_CLIENTS = 2
    NUM_ROUNDS = 15
    
    start_enhanced_server(
        expected_clients=EXPECTED_CLIENTS,
        num_rounds=NUM_ROUNDS
    )
