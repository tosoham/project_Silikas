import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on ", device)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Create DataLoader for mini-batch training
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# --- Model Definitions ---
# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1).to(device)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# MLP Model with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=30):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# --- Training Function ---
def train_model(model, train_loader, epochs=20):
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # Track accuracy
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            running_loss += loss.item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    return model

# --- Evaluation Function ---
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Test Accuracy: {100 * accuracy:.2f}%")

# --- Main Execution ---
if __name__ == "__main__":
    input_dim = X_train.shape[1]

    # Train and evaluate Logistic Regression
    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(input_dim)
    trained_log_reg = train_model(log_reg, train_loader, epochs=100)
    evaluate_model(trained_log_reg, X_test, y_test)

    # Train and evaluate MLP
    print("\nTraining MLP...")
    mlp = MLP(input_dim, hidden_dim=30)
    trained_mlp = train_model(mlp, train_loader, epochs=100)
    evaluate_model(trained_mlp, X_test, y_test)

    # Save the trained models for use in the next program
    torch.save(trained_log_reg.state_dict(), "log_reg.pth")
    torch.save(trained_mlp.state_dict(), "mlp.pth")
    print("\nModels saved as 'log_reg.pth' and 'mlp.pth'")