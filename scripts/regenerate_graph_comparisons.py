"""Regenerate graph-model test forecasts and paired HAC comparisons.

This deliberately reproduces the graph-message-passing experiment with the same
split, graph estimation, validation grid, random seed and MLP settings, then
writes test_forecasts.csv and forecast_comparison_hac.json from those identical
predictions. No test outcomes are used for fitting or tuning.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from run_graph_message_passing import feature_graph, adjacency_from_edges, message_features
from run_robust_experiments import build, hac


def make_mlp(hidden, alpha):
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=hidden, alpha=alpha, max_iter=350,
        early_stopping=False, random_state=42, learning_rate_init=0.001))


def tune(X, y, V, vy):
    grid = [((16,), 0.001), ((32,), 0.001), ((32, 16), 0.001), ((32, 16), 0.01)]
    best = None
    for hidden, alpha in grid:
        sy = StandardScaler().fit(y.reshape(-1, 1))
        model = make_mlp(hidden, alpha).fit(X, sy.transform(y.reshape(-1, 1)).ravel())
        pred = sy.inverse_transform(model.predict(V).reshape(-1, 1)).ravel()
        mse = mean_squared_error(vy, pred)
        row = (mse, hidden, alpha)
        if best is None or row[0] < best[0]:
            best = row
    return best[1], best[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="SP500.csv")
    p.add_argument("--out", default="results/advanced/graph_message_passing")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frame, _ = build(pd.read_csv(args.data))
    n = len(frame); i, j = int(0.6*n), int(0.8*n)
    train, val, test = frame.iloc[:i], frame.iloc[i:j], frame.iloc[j:]
    features = [c for c in frame.columns if c not in ("Date", "target")]
    edges = feature_graph(train, features)
    A = adjacency_from_edges(edges, features)
    edge_count = int(edges["selected"].sum())
    rng = np.random.default_rng(20261026)
    random_A = np.zeros_like(A)
    possible = [(a,b) for a in range(len(features)) for b in range(len(features)) if a != b]
    if edge_count:
        chosen = rng.choice(len(possible), size=min(edge_count, len(possible)), replace=False)
        for k in chosen: random_A[possible[int(k)]] = 1.0
        deg = random_A.sum(axis=0); deg[deg == 0] = 1.0; random_A /= deg[None, :]
    mats = {"graph_message_passing": A, "no_graph_zero_message": np.zeros_like(A), "random_graph_ablation": random_A}
    y = train.target.to_numpy(); vy = val.target.to_numpy(); yt = test.target.to_numpy()
    X = train[features].to_numpy(); V = val[features].to_numpy(); T = test[features].to_numpy()
    history = pd.concat([train, val]); forecasts = {}
    for name, matrix in mats.items():
        hidden, alpha = tune(message_features(X,matrix), y, message_features(V,matrix), vy)
        sy = StandardScaler().fit(history.target.to_numpy().reshape(-1,1))
        model = make_mlp(hidden, alpha).fit(
            message_features(history[features].to_numpy(), matrix),
            sy.transform(history.target.to_numpy().reshape(-1,1)).ravel())
        forecasts[name] = sy.inverse_transform(
            model.predict(message_features(T,matrix)).reshape(-1,1)).ravel()
    forecasts["training_mean_baseline"] = np.full(len(yt), float(history.target.mean()))
    forecasts["zero_return_baseline"] = np.zeros(len(yt))
    pd.DataFrame({"Date": test.Date.to_numpy(), "actual": yt, **forecasts}).to_csv(out/"test_forecasts.csv", index=False)

    pairs = [
        ("graph_message_passing", "no_graph_zero_message"),
        ("graph_message_passing", "random_graph_ablation"),
        ("graph_message_passing", "training_mean_baseline"),
        ("graph_message_passing", "zero_return_baseline"),
    ]
    comparisons = {}
    for a,b in pairs:
        comparisons[f"{a}_vs_{b}"] = hac((yt-forecasts[a])**2, (yt-forecasts[b])**2, L=10)
    valid = sorted([(k,v["p_value"]) for k,v in comparisons.items()], key=lambda x:x[1])
    running=0.0; m=len(valid)
    for rank,(name,pv) in enumerate(valid):
        running=max(running,min(1.0,(m-rank)*pv)); comparisons[name]["p_value_holm"]=running
    (out/"forecast_comparison_hac.json").write_text(json.dumps({
        "primary_bandwidth":10,
        "comparison_family":"four predeclared graph-model comparisons",
        "loss":"squared error",
        "interpretation":"Positive mean_loss_difference_A_minus_B means model B has lower squared-error loss than model A.",
        "comparisons":comparisons,
        "source_predictions":"test_forecasts.csv; all comparisons use the identical saved test predictions",
        "graph_selected_edges":edge_count,
        "inference_note":"Approximate HAC inference; bandwidth sensitivity should be reported alongside the primary result."
    },indent=2))
    print(f"Regenerated {len(yt)} identical test predictions and four HAC comparisons; selected edges={edge_count}.")

if __name__ == "__main__": main()
