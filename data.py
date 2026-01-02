import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_graph_ts(num_nodes=10, seq_len=300, noise=0.1, seed=42):
    np.random.seed(seed)

    # adjacency matrix
    A = np.random.rand(num_nodes, num_nodes)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)

    # base time pattern
    t = np.linspace(0, 20, seq_len)
    base = np.sin(t) + 0.3 * np.cos(0.5 * t)

    data = []
    for i in range(num_nodes):
        offset = np.random.uniform(-0.5, 0.5)
        series = base + offset + noise * np.random.randn(seq_len)
        data.append(series)

    data = np.stack(data, axis=1)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data.astype(np.float32), A.astype(np.float32)


def create_dataset(data, input_len=12, pred_len=3):
    X, Y = [], []
    T, N = data.shape

    for i in range(T - input_len - pred_len):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+pred_len])

    return np.array(X), np.array(Y)
