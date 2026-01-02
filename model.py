import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        I = torch.eye(A.size(0), device=A.device)
        A_hat = A + I
        D_hat = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        A_norm = D_hat @ A_hat @ D_hat

        X = self.linear(X)
        return torch.relu(A_norm @ X)


class GNN_GRU(nn.Module):
    def __init__(self, num_nodes, gcn_hidden=16, gru_hidden=32, pred_len=3):
        super().__init__()
        self.gcn = GCNLayer(1, gcn_hidden)
        self.gru = nn.GRU(gcn_hidden * num_nodes, gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, pred_len * num_nodes)
        self.num_nodes = num_nodes
        self.pred_len = pred_len

    def forward(self, X, A):
        B, T, N = X.shape
        X = X.unsqueeze(-1)

        gcn_seq = []
        for t in range(T):
            x_t = self.gcn(X[:, t], A)
            gcn_seq.append(x_t)

        gcn_seq = torch.stack(gcn_seq, dim=1)
        gcn_seq = gcn_seq.reshape(B, T, -1)

        _, h = self.gru(gcn_seq)
        h = h.squeeze(0)

        out = self.fc(h)
        return out.view(B, self.pred_len, self.num_nodes)
