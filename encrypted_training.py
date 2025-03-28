import numpy as np
import time
import torch
from torch import nn
from torch.optim import SGD
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from concrete.ml.sklearn import LogisticRegression, NeuralNetClassifier
from sklearn.neural_network import MLPClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on ", device)

# --- Load and Preprocess Spambase Dataset ---
print("Loading and preprocessing Spambase dataset...")
data = fetch_openml('spambase', version=1, as_frame=False)
X, y = data.data, data.target
y = y.astype(int)  # Convert labels to integers (0 or 1)

# Normalize features to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train and Evaluate Models with Concrete ML ---

# 1. Logistic Regression
print("\nTraining Logistic Regression with Concrete ML...")
log_reg_model = LogisticRegression(n_bits=18, random_state=42)
log_reg_model.fit(X_train, y_train)

# Clear evaluation (non-FHE)
print("Evaluating Logistic Regression on clear data (non-FHE)...")
y_pred_clear_log_reg = log_reg_model.predict(X_test)
clear_accuracy_log_reg = (y_pred_clear_log_reg == y_test).mean()
print(f"Clear Test Accuracy (Logistic Regression): {100 * clear_accuracy_log_reg:.2f}%")

# Compile for FHE
print("Compiling Logistic Regression for FHE...")
compiled_log_reg_circuit = log_reg_model.compile(X_train)
print(f"Generating a key for an {compiled_log_reg_circuit.graph.maximum_integer_bit_width()}-bit circuit")
time_begin = time.time()
compiled_log_reg_circuit.client.keygen(force=False)
print(f"Key generation time: {time.time() - time_begin:.4f} seconds")
     

# FHE inference
print("Running FHE Inference for Logistic Regression...")
time_begin = time.time()
y_pred_fhe_log_reg = log_reg_model.predict(X_test, fhe="execute")
print(f"Execution time: {(time.time() - time_begin) / len(X_test):.4f} seconds per sample")
fhe_accuracy_log_reg = (y_pred_fhe_log_reg == y_test).mean()
print(f"FHE Test Accuracy (Logistic Regression): {100 * fhe_accuracy_log_reg:.2f}%")

# Compare predictions
print("Comparing clear and FHE predictions for Logistic Regression...")
mismatches_log_reg = np.sum(y_pred_clear_log_reg != y_pred_fhe_log_reg)
print(f"Number of prediction mismatches: {mismatches_log_reg}")

# 2. Multi-Layer Perceptron (MLP)
print("\nTraining MLP with Concrete ML...")
mlp_model = NeuralNetClassifier(
    # criterion = torch.nn.BCELoss,
    optimizer = SGD,
    lr = 0.01,
    max_epochs = 100,
    batch_size = 8,
    module__n_layers=3,
    module__n_w_bits=2,
    module__n_a_bits=2,
    module__n_accum_bits=7,
    module__n_hidden_neurons_multiplier=1,
    module__activation_function=torch.nn.ReLU,
    device = 'cpu',
)

# X_train = torch.tensor(X_train)
# y_train = torch.tensor(y_train).view(-1, 1)
# X_test = torch.tensor(X_test)
# y_test = torch.tensor(y_test).view(-1, 1)

mlp_model.fit(X_train, y_train)
y_pred_clear_mlp = mlp_model.predict(X_test)
clear_accuracy_mlp = (np.array(y_pred_clear_mlp) == y_test).astype(int).mean()
print(f"Clear Test Accuracy (Logistic Regression): {100 * clear_accuracy_mlp:.2f}%")

# Compile the model
print("Compiling MLP for FHE...")
compiled_mlp_circuit = mlp_model.compile(X_train)

print(f"Generating a key for an {compiled_mlp_circuit.graph.maximum_integer_bit_width()}-bit circuit")
time_begin = time.time()
compiled_mlp_circuit.client.keygen(force=False)
print(f"Key generation time: {time.time() - time_begin:.4f} seconds")

# FHE inference
print("Running FHE Inference for Multi-Layer Perceptron (MLP)...")
time_begin = time.time()
y_pred_fhe_mlp = mlp_model.predict(X_test, fhe="execute")
print(f"Execution time: {(time.time() - time_begin) / len(X_test):.4f} seconds per sample")
fhe_accuracy_mlp = (np.array(y_pred_fhe_mlp) == y_test).astype(int).mean()
print(f"FHE Test Accuracy (Multi-Layer Perceptron (MLP)): {100 * fhe_accuracy_mlp:.2f}%")

# Compare predictions
print("Comparing clear and FHE predictions for Multi-Layer Perceptron (MLP)...")
mismatches_mlp = np.sum(np.array(y_pred_clear_mlp) != np.array(y_pred_fhe_mlp))
print(f"Number of prediction mismatches: {mismatches_mlp}")





