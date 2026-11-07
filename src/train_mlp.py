# src/train_mlp.py — the hand-built PyTorch MLP.
# Data arrives through src/common.py

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
# THis loads the data split frames
from common import load_frames, FEATURES, CD_COL

# The seed goes here, before the model exists
torch.manual_seed(42)

# loading the splits
train_df, val_df, test_df = load_frames()
print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# --- Scale the features, fitting the scaler on the TRAIN rows only ---------------
scaler = StandardScaler()
scaler.fit(train_df[FEATURES])

X_train = scaler.transform(train_df[FEATURES]).astype("float32")
y_train = train_df[CD_COL].to_numpy(dtype="float32").reshape(-1, 1)
X_val = scaler.transform(val_df[FEATURES]).astype("float32")
y_val = val_df[CD_COL].to_numpy(dtype="float32").reshape(-1, 1)
X_test = scaler.transform(test_df[FEATURES]).astype("float32")
y_test = test_df[CD_COL].to_numpy(dtype="float32").reshape(-1, 1)

print("Scaled train means (expect ~0):", np.round(X_train.mean(axis=0), 3))
print("Scaled train stds  (expect ~1):", np.round(X_train.std(axis=0), 3))

# --- Chunk: the Dataset -----------------------------------------------------------
from torch.utils.data import Dataset


class CdDataset(Dataset):
    """
    Serves one design at a time:  its scaled features, and its Cd.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_ds = CdDataset(X_train, y_train)
val_ds = CdDataset(X_val, y_val)
test_ds = CdDataset(X_test, y_test)

print("Train designs:", len(train_ds))
x0, y0 = train_ds[0]
print("First design:", x0.shape, "features; its Cd:", float(y0))

# --- Chunk: the DataLoaders ---------------------------------------------------------
from torch.utils.data import DataLoader

BATCH_SIZE = 64

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# One batch, fetched and read:
xb, yb = next(iter(train_loader))
print("Batch features:", xb.shape, " Batch targets:", yb.shape)

# --- Chunk: the model ------------------------------------------------------------------
import torch.nn as nn


class CdNet(nn.Module):
        """
        The network: inputs -> 128 -> 64 -> 1, ReLU between.
        """
        def __init__(self, n_features):
             super().__init__()
             self.stack = nn.Sequential(
                  nn.Linear(n_features, 128),
                  nn.ReLU(),
                  nn.Linear(128, 64),
                  nn.ReLU(),
                  nn.Linear(64, 1), # 1 regression output
             )

        def forward(self, x):
             return self.stack(x)

model = CdNet(n_features=len(FEATURES))

# One untrained pass over the batch to check the shape
print("Untrained output shape:", model(xb).shape)

# --- Chunk: the training loop -------------------------------------------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 100

for epoch in range(1, EPOCHS + 1):
     model.train() # training mode

     total_loss = 0.0
     for xb, yb in train_loader:
          pred = model(xb)          # 1. predict
          loss = loss_fn(pred, yb)  # 2. score the prediction against the truth
          optimizer.zero_grad()      # 3. clear the old gradients
          loss.backward()           # 4. backprop for new gradients
          optimizer.step()          # 5. move the weights
          total_loss += loss.item() * len(xb)

     line = f"epoch {epoch}: train loss {total_loss / len(train_ds):.6f}"

     model.eval()  # scoring mode
     val_total = 0.0
     with torch.no_grad():
          for xb, yb in val_loader:
               val_total += loss_fn(model(xb), yb).item() * len(xb)
     line += f", val loss {val_total / len(val_ds):.6f}"

     print(line)

# --- Chunk: scoring, and the results file ---------------------------------------------------
def r2_from_definition(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def predict_array(X):
    """
    Predict a whole frame from its array, in table order.

    Scoring never goes through the train loader: that loader shuffles, and
    shuffled predictions scored against unshuffled truth would be garbage.
    """
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


train_r2 = r2_from_definition(y_train, predict_array(X_train))
val_r2 = r2_from_definition(y_val, predict_array(X_val))
test_r2 = r2_from_definition(y_test, predict_array(X_test))
print(f"Train R²: {train_r2}")
print(f"Validation R²: {val_r2}")
print(f"Test R²: {test_r2}")

# Cross-check: the library's one-liner on the same test numbers must agree.
from sklearn.metrics import r2_score

print(f"sklearn cross-check: {r2_score(y_test, predict_array(X_test))}")

lines = [
    "contender: PyTorch MLP (hand-built)",
    f"architecture: {len(FEATURES)} -> 128 -> 64 -> 1, ReLU",
    f"epochs: {EPOCHS}",
    f"train_rows: {len(train_df)}",
    f"val_rows: {len(val_df)}",
    f"test_rows: {len(test_df)}",
    f"train_r2: {train_r2}",
    f"val_r2: {val_r2}",
    f"test_r2: {test_r2}",
]
with open("results/mlp_results.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote results/mlp_results.txt")



