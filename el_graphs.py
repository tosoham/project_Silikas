import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.sklearn import NeuralNetClassifier
from concrete.ml.torch.compile import compile_torch_model
import matplotlib.pyplot as plt
import time

# --- Load and Preprocess Spambase Dataset ---
print("Loading and preprocessing Spambase dataset...")
data = fetch_openml('spambase', version=1, as_frame=False)
X, y = data.data, data.target
y = y.astype(int)

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch tensors for PyTorch training
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# --- PyTorch Model Definitions ---
class PyTorchLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=30):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# --- Train PyTorch Models ---
def train_pytorch_model(model, train_loader, epochs=20):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return model

input_dim = X_train.shape[1]
print("\nTraining PyTorch Logistic Regression...")
pytorch_log_reg = PyTorchLogisticRegression(input_dim)
pytorch_log_reg = train_pytorch_model(pytorch_log_reg, train_loader)
torch.save(pytorch_log_reg.state_dict(), "pytorch_log_reg.pth")

print("Training PyTorch MLP...")
pytorch_mlp = PyTorchMLP(input_dim)
pytorch_mlp = train_pytorch_model(pytorch_mlp, train_loader)
torch.save(pytorch_mlp.state_dict(), "pytorch_mlp.pth")

# --- Load PyTorch Models for FHE ---
pytorch_log_reg.load_state_dict(torch.load("pytorch_log_reg.pth"))
pytorch_mlp.load_state_dict(torch.load("pytorch_mlp.pth"))

# --- Concrete ML Models ---
print("\nTraining Concrete ML Logistic Regression...")
concrete_log_reg = ConcreteLogisticRegression(n_bits=8, random_state=42)
concrete_log_reg.fit(X_train, y_train)

print("Training Concrete ML MLP...")
concrete_mlp = NeuralNetClassifier(
    #criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=optim.SGD,
    lr=0.01,
    max_epochs=10,
    batch_size=8,
    module__n_layers=3,
    module__n_w_bits=2,
    module__n_a_bits=2,
    module__n_accum_bits=7,
    module__n_hidden_neurons_multiplier=1,
    module__activation_function=torch.nn.ReLU,
    device='cpu'
)
concrete_mlp.fit(X_train, y_train)

# --- Inference and Metrics ---
models = {
    "PyTorch Logistic Regression": (pytorch_log_reg, compile_torch_model(pytorch_log_reg, X_train_torch, rounding_threshold_bits=8)),
    "PyTorch MLP": (pytorch_mlp, compile_torch_model(pytorch_mlp, X_train_torch, rounding_threshold_bits=8)),
    "Concrete ML Logistic Regression": (concrete_log_reg, concrete_log_reg.compile(X_train)),
    "Concrete ML MLP": (concrete_mlp, concrete_mlp.compile(X_train))
}

results = {"clear_accuracy": {}, "fhe_accuracy": {}, "fhe_time_per_sample": {}, "mismatches": {}}

for name, (model, compiled_model) in models.items():
    print(f"\nProcessing {name}...")
    
    # Clear inference
    if "PyTorch" in name:
        model.eval()
        with torch.no_grad():
            y_pred_clear = (model(X_test_torch) > 0.5).float().numpy().flatten()
    else:
        y_pred_clear = model.predict(X_test)
    clear_accuracy = (y_pred_clear == y_test).mean()
    results["clear_accuracy"][name] = clear_accuracy
    
    # FHE inference with timing
    time_begin = time.time()
    if "PyTorch" in name:
        # key = compiled_model.client.keygen()
        y_pred_fhe = compiled_model.forward(X_test_torch.numpy())
        y_pred_fhe = (y_pred_fhe > 0.5).astype(int)
    else:
        compiled_model = model.compile(X_train)
        key = compiled_model.client.keygen()
        y_pred_fhe = model.predict(X_test, fhe="execute")
    fhe_time = (time.time() - time_begin) / len(X_test)
    fhe_accuracy = (y_pred_fhe == y_test).mean()
    results["fhe_accuracy"][name] = fhe_accuracy
    results["fhe_time_per_sample"][name] = fhe_time
    results["mismatches"][name] = np.sum(y_pred_clear != y_pred_fhe)

# --- Generate Graphs ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Accuracy Comparison
model_names = list(results["clear_accuracy"].keys())
clear_accs = [results["clear_accuracy"][name] * 100 for name in model_names]
fhe_accs = [results["fhe_accuracy"][name] * 100 for name in model_names]
x = np.arange(len(model_names))
width = 0.35
ax1.bar(x - width/2, clear_accs, width, label='Clear', color='skyblue')
ax1.bar(x + width/2, fhe_accs, width, label='FHE', color='salmon')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Clear vs FHE Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha="right")
ax1.legend()

# FHE Execution Time
fhe_times = [results["fhe_time_per_sample"][name] for name in model_names]
ax2.bar(model_names, fhe_times, color='lightgreen')
ax2.set_ylabel('Time per Sample (seconds)')
ax2.set_title('FHE Inference Time per Sample')
ax2.set_xticklabels(model_names, rotation=45, ha="right")

# Prediction Mismatches
mismatches = [results["mismatches"][name] for name in model_names]
ax3.bar(model_names, mismatches, color='lightcoral')
ax3.set_ylabel('Number of Mismatches')
ax3.set_title('Clear vs FHE Prediction Mismatches')
ax3.set_xticklabels(model_names, rotation=45, ha="right")

plt.tight_layout()
plt.savefig("comparison_graphs.png")
plt.show()

# Print results
for name in model_names:
    print(f"\n{name}:")
    print(f"Clear Accuracy: {results['clear_accuracy'][name] * 100:.2f}%")
    print(f"FHE Accuracy: {results['fhe_accuracy'][name] * 100:.2f}%")
    print(f"FHE Time per Sample: {results['fhe_time_per_sample'][name]:.4f} seconds")
    print(f"Mismatches: {results['mismatches'][name]}")