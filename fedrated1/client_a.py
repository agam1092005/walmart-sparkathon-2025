from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import flwr as fl
from model import load_data , create_preprocessor , build_model
import matplotlib.pyplot as plt

file_path = 'cleaned_dataset1.txt'
X, y = load_data(file_path)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
preprocessor = create_preprocessor(X_train)
X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
X_test_processed = preprocessor.transform(X_test).astype(np.float32)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Build Model
input_shape = X_train_processed.shape[1]
model = build_model(input_shape)

# Train Model
history = model.fit(
    X_train_processed, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['test_accuracy'], label='test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['test_loss'], label='test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()