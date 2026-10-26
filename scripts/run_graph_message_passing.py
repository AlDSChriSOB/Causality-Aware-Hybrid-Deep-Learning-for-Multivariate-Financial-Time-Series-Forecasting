"""Graph message-passing ablation for Paper 2.

The feature graph is a directed lagged-predictive graph estimated on the initial
training partition only. It is not a structural causal graph. Each sample is
represented by raw node values plus graph-aggregated incoming messages; this
makes adjacency affect the model's inputs beyond selecting a subset of features.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_robust_experiments import build, bh


def feature_graph(train, features, maxlag=3, alpha=0.05):
    """Estimate source -> destination Granger edges using training rows only."""
    from statsmodels.tsa.stattools import grangercausalitytests
    import contextlib
    import io
    import warnings

    rows = []
    for source in features:
        for destination in features:
            if source == destination:
                continue
            pair = train[[destination, source]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            pair = pair[np.isfinite(pair).all(axis=1)]
            err = ""
            try:
                if len(pair) <= 3 * maxlag + 10:
                    raise ValueError("insufficient observations")
                if np.std(pair[:, 0]) < 1e-12 or np.std(pair[:, 1]) < 1e-12:
                    raise ValueError("constant series")
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    result = grangercausalitytests(pair, maxlag=maxlag)
                ps = [float(result[k][0]["ssr_ftest"][1]) for k in range(1, maxlag + 1)]
                if not np.isfinite(ps).all():
                    raise ValueError("non-finite p-values")
                p_min = min(ps)
                # Bonferroni across tested lags for each ordered feature pair.
                p_lag_adjusted = min(1.0, p_min * maxlag)
            except Exception as exc:
                p_min, p_lag_adjusted = 1.0, 1.0
                err = f"{type(exc).__name__}: {exc}"
            rows.append({"source": source, "destination": destination,
                         "min_p": p_min, "lag_adjusted_p": p_lag_adjusted,
                         "test_error": err})
    qvals = bh([r["lag_adjusted_p"] for r in rows])
    for row, q in zip(rows, qvals):
        row["fdr_q"] = float(q)
        row["selected"] = bool(not row["test_error"] and q < alpha)
    if rows and all(r["test_error"] for r in rows):
        raise RuntimeError("All pairwise Granger tests failed; refusing to create an empty graph.")
    return pd.DataFrame(rows)


def adjacency_from_edges(edge_table, features):
    """Return row-normalised A where A[source, destination] encodes source messages."""
    index = {name: i for i, name in enumerate(features)}
    A = np.zeros((len(features), len(features)), dtype=float)
    for row in edge_table.loc[edge_table["selected"]].itertuples():
        A[index[row.source], index[row.destination]] = 1.0
    degree = A.sum(axis=0)
    degree[degree == 0] = 1.0
    return A / degree[None, :]


def message_features(X, A):
    """Append incoming graph messages to raw feature values."""
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float)
    return np.concatenate([X, X @ A], axis=1)


def metrics(y, pred):
    return {"n": int(len(y)), "MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2_vs_test_mean": float(r2_score(y, pred))}


def make_mlp(hidden, alpha):
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=hidden, alpha=alpha, max_iter=350,
        early_stopping=False, random_state=42, learning_rate_init=0.001))


def tune_and_fit(X, y, V, vy):
    """Validation-only model selection; return selected configuration and validation scores."""
    grid = [((16,), 0.001), ((32,), 0.001), ((32, 16), 0.001), ((32, 16), 0.01)]
    records = []
    for hidden, alpha in grid:
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        model = make_mlp(hidden, alpha)
        model.fit(X, scaler_y.transform(y.reshape(-1, 1)).ravel())
        pv = scaler_y.inverse_transform(model.predict(V).reshape(-1, 1)).ravel()
        records.append({"hidden": list(hidden), "alpha": alpha,
                        "validation_MSE": float(mean_squared_error(vy, pv))})
    best = min(records, key=lambda r: r["validation_MSE"])
    return tuple(best["hidden"]), best["alpha"], records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="SP500.csv")
    parser.add_argument("--out", default="results/advanced/graph_message_passing")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame, raw = build(pd.read_csv(args.data))
    n = len(frame)
    i, j = int(0.6 * n), int(0.8 * n)
    train, val, test = frame.iloc[:i], frame.iloc[i:j], frame.iloc[j:]
    features = [c for c in frame.columns if c not in ("Date", "target")]
    graph_tests = feature_graph(train, features)
    A = adjacency_from_edges(graph_tests, features)
    edge_count = int(graph_tests["selected"].sum())

    # Matched controls preserve the same 2d input width and MLP family.
    rng = np.random.default_rng(20261026)
    random_A = np.zeros_like(A)
    possible = [(a, b) for a in range(len(features)) for b in range(len(features)) if a != b]
    if edge_count:
        chosen = rng.choice(len(possible), size=min(edge_count, len(possible)), replace=False)
        for k in chosen:
            random_A[possible[int(k)]] = 1.0
        deg = random_A.sum(axis=0)
        deg[deg == 0] = 1.0
        random_A /= deg[None, :]

    y = train.target.to_numpy()
    vy = val.target.to_numpy()
    yt = test.target.to_numpy()
    X, V, T = train[features].to_numpy(), val[features].to_numpy(), test[features].to_numpy()
    architectures = {
        "graph_message_passing": A,
        "no_graph_zero_message": np.zeros_like(A),
        "random_graph_ablation": random_A,
    }
    results, tuning = [], {}
    final_history = pd.concat([train, val])
    for name, matrix in architectures.items():
        x_train, x_val = message_features(X, matrix), message_features(V, matrix)
        hidden, alpha, records = tune_and_fit(x_train, y, x_val, vy)
        tuning[name] = {"selected_hidden": list(hidden), "selected_alpha": alpha,
                        "validation_grid": records}
        # Final fit uses train+validation; graph topology remains frozen from train.
        x_final = message_features(final_history[features].to_numpy(), matrix)
        x_test = message_features(T, matrix)
        y_final = final_history.target.to_numpy()
        scaler_y = StandardScaler().fit(y_final.reshape(-1, 1))
        model = make_mlp(hidden, alpha).fit(
            x_final, scaler_y.transform(y_final.reshape(-1, 1)).ravel())
        pred = scaler_y.inverse_transform(model.predict(x_test).reshape(-1, 1)).ravel()
        results.append({"model": name, "selected_graph_edges": edge_count,
                        "input_feature_nodes": len(features), "input_width": 2 * len(features),
                        **metrics(yt, pred)})
    graph_tests.to_csv(out / "feature_graph_edges.csv", index=False)
    pd.DataFrame(results).to_csv(out / "test_metrics.csv", index=False)
    (out / "tuning.json").write_text(json.dumps(tuning, indent=2))
    (out / "metadata.json").write_text(json.dumps({
        "target": "log(Close[t+1])-log(Close[t])",
        "graph_estimation": "ordered pairwise Granger tests on initial training partition only",
        "edge_semantics": "lagged predictive dependence, not structural causation",
        "multiple_testing": "Bonferroni across lags per ordered pair, then Benjamini-Hochberg across ordered pairs",
        "test_used_for_tuning": False,
        "graph_frozen_after_initial_training": True,
        "selected_edge_count": edge_count,
        "edge_fallback_used": False,
        "input_sha256": __import__("hashlib").sha256(Path(args.data).read_bytes()).hexdigest()
    }, indent=2))
    print(pd.DataFrame(results).sort_values("RMSE").to_string(index=False))
    print(f"Selected directed feature edges: {edge_count}; no exploratory edge fallback was used.")


if __name__ == "__main__":
    main()
