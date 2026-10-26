"""Unit tests for the graph message-passing model's core invariants."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_graph_message_passing import adjacency_from_edges, message_features


def main():
    features = ["x", "y", "z"]
    edges = pd.DataFrame([
        {"source": "x", "destination": "y", "selected": True},
        {"source": "z", "destination": "y", "selected": True},
        {"source": "y", "destination": "z", "selected": False},
    ])
    A = adjacency_from_edges(edges, features)
    assert A.shape == (3, 3)
    # Both selected source nodes send a normalized message to destination y.
    assert np.isclose(A[0, 1], 0.5) and np.isclose(A[2, 1], 0.5)
    assert np.isclose(A[:, 0].sum(), 0.0)
    X = np.array([[2.0, 4.0, 6.0]])
    mapped = message_features(X, A)
    assert mapped.shape == (1, 6)
    # The y-node's aggregate is the mean of x and z, not a feature mask.
    assert np.isclose(mapped[0, 3 + 1], 4.0)
    assert np.array_equal(mapped[0, :3], X[0])
    assert not np.array_equal(message_features(X, A), message_features(X, np.zeros_like(A)))
    print("PASS: directed adjacency orientation, normalization, and graph-dependent messages verified.")


if __name__ == "__main__":
    main()
