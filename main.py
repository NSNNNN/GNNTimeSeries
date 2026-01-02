import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from data import generate_graph_ts, create_dataset
from model import GNN_GRU
from metrics import mae, rmse, mape

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, A = generate_graph_ts()
    X, Y = create_dataset(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    num_nodes = data.shape[1]
    model = GNN_GRU(num_nodes=num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb, A_t)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

        pred = model(X_test_t, A_t).cpu().numpy()
        true = y_test_t.cpu().numpy()

        print("\nEvaluation:")
        print("MAE :", mae(true, pred))
        print("RMSE:", rmse(true, pred))
        print("MAPE:", mape(true, pred))

if __name__ == "__main__":
    main()
