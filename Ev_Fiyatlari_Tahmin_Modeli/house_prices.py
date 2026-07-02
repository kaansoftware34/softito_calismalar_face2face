import sys

try:
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
    missing_pkg = getattr(e, 'name', None) or str(e)
    raise ModuleNotFoundError(
        f"Required package '{missing_pkg}' is missing. Install dependencies with pip: pip install pandas numpy torch scikit-learn matplotlib seaborn"
    ) from e

# 1. Load Dataset
# Dosya adını doğrudan veriyoruz. Böylece Python dosyayı çalıştığı klasörde arayacak.
file_path = 'house_prices.csv'
df = pd.read_csv(file_path)


print(f"Data successfully loaded. Shape: {df.shape}")

# 2. Data Preprocessing & Feature Selection
# Selecting available numerical features for house price prediction
features = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
]
target = 'SalePrice'

# Cleaning the data from missing values (if any)
df_clean = df[features + [target]].dropna()

X = df_clean[features].values
y = df_clean[target].values.reshape(-1, 1)

# Split the data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature standardization for more stable and better PyTorch training performance
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert numpy arrays to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# 3. Building an Improved PyTorch Model Architecture
class PlayerPerformanceModel(nn.Module):
    def __init__(self, input_dim):
        super(PlayerPerformanceModel, self).__init__()
        # Using a multi-layer architecture with Dropout to prevent overfitting
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

model = PlayerPerformanceModel(input_dim=len(features))

# 4. Optimizer and Loss Function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) # Using Adam optimizer for faster convergence

# 5. Model Training Process
epochs = 100
train_losses = []

print("Starting training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass & optimization
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 6. Model Evaluation
model.eval()
with torch.no_grad():
    predictions_scaled = model(X_test_tensor)
    # Reverse the prediction scaling back to the original rating values (0-10)
    predictions = scaler_y.inverse_transform(predictions_scaled.numpy())

# Calculate a simple correlation score
correlation = np.corrcoef(y_test.flatten(), predictions.flatten())[0, 1]
print(f"\nModel Evaluation Complete. Correlation Output: {correlation:.4f}")

# 7. Result Visualization with a Blue Color Theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Loss Decay (Dark Blue)
ax[0].plot(train_losses, color='#1f77b4', linewidth=2.5, label='Train Loss')
ax[0].set_title('Training Loss Decay Over Epochs', fontsize=12, fontweight='bold', color='#0f4c81')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Mean Squared Error (MSE)')
ax[0].legend()

# Plot 2: Actual vs Predicted (Light Blue / Gradient Scatter)
ax[1].scatter(y_test, predictions, alpha=0.4, color='#5b92e5', label='Predicted Prices')
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='#0f4c81', lw=2, label='Perfect Prediction')
ax[1].set_title('Actual vs Predicted House Prices', fontsize=12, fontweight='bold', color='#0f4c81')
ax[1].set_xlabel('Actual Price')
ax[1].set_ylabel('Predicted Price')
ax[1].legend()

plt.tight_layout()
plt.savefig('output.png')
print("Plot saved as output.png")