"""Leakage-aware baseline experiments for Paper 2.

Run from the repository root:
    python scripts/run_baseline_experiments.py --data SP500.csv --out results/baseline

This first-stage script deliberately does not claim to implement the graph-constrained
neural model. It produces a reproducible benchmark table and aligned out-of-sample
forecasts; the causal-graph model and neural ablations are subsequent stages.
"""
from __future__ import annotations
import argparse, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    required = {'Date','Open','High','Low','Close'}
    missing = required - set(raw.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        if c in raw: raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw.dropna(subset=['Date','Open','High','Low','Close']).sort_values('Date')
    raw = raw.drop_duplicates('Date', keep='last').reset_index(drop=True)
    if (raw[['Open','High','Low','Close']] <= 0).any().any():
        raise ValueError('OHLC prices must be positive for log-return features.')
    close = raw['Close'].astype(float)
    log_close = np.log(close)
    ret = log_close.diff()
    f = pd.DataFrame({'Date': raw['Date'], 'Close': close})
    f['target_next_return'] = ret.shift(-1)
    for lag in [1,2,3,5,10,20]: f[f'ret_lag_{lag}'] = ret.shift(lag-1)
    f['range_scaled'] = (raw['High'] - raw['Low']) / close
    f['open_close_return'] = np.log(raw['Close'] / raw['Open'])
    f['overnight_gap'] = np.log(raw['Open'] / raw['Close'].shift(1))
    for w in [5,20,60]: f[f'volatility_{w}'] = ret.rolling(w, min_periods=w).std()
    if 'Volume' in raw and (raw['Volume'] > 0).any():
        vol = raw['Volume'].where(raw['Volume'] > 0)
        f['log_volume_change'] = np.log(vol).diff()
    else:
        f['log_volume_change'] = np.nan
    # RSI computed from information available at t; no centred windows.
    delta = close.diff(); gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    f['rsi14'] = 100 - 100 / (1 + rs)
    f = f.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return f

def hac_dm(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int | None = None) -> dict:
    """Two-sided normal DM-style test with Newey-West long-run variance.
    Positive mean(loss_a-loss_b) means model B has lower loss than model A.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n < 10: return {'mean_loss_difference': float(np.mean(d)), 'statistic': None, 'p_value': None, 'n': n}
    d = d - d.mean()
    L = int(max_lag if max_lag is not None else max(1, round(4*(n/100)**(2/9))))
    gamma0 = np.dot(d,d)/n
    lrv = gamma0
    for lag in range(1, min(L,n-1)+1):
        gamma = np.dot(d[lag:], d[:-lag])/n
        weight = 1 - lag/(L+1)
        lrv += 2*weight*gamma
    if not np.isfinite(lrv) or lrv <= 0:
        return {'mean_loss_difference': float(np.mean(loss_a-loss_b)), 'statistic': None, 'p_value': None, 'n': n, 'hac_lags': L}
    stat = float(np.mean(loss_a-loss_b) / np.sqrt(lrv/n))
    # normal approximation; report as HAC loss-differential test, not exact finite-sample DM
    from scipy.stats import norm
    return {'mean_loss_difference': float(np.mean(loss_a-loss_b)), 'statistic': stat, 'p_value': float(2*norm.sf(abs(stat))), 'n': n, 'hac_lags': L}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='SP500.csv')
    ap.add_argument('--out', default='results/baseline')
    ap.add_argument('--train-frac', type=float, default=.60)
    ap.add_argument('--validation-frac', type=float, default=.20)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data)
    frame = build_frame(raw)
    n = len(frame); tr_end = int(n*args.train_frac); val_end = int(n*(args.train_frac+args.validation_frac))
    if not (0 < tr_end < val_end < n): raise ValueError('Invalid chronological split fractions.')
    feature_cols = [c for c in frame.columns if c not in ['Date','Close','target_next_return']]
    train, val, test = frame.iloc[:tr_end], frame.iloc[tr_end:val_end], frame.iloc[val_end:]
    # Fixed, chronological split; no test-driven tuning. Preprocessing fitted on training only.
    Xtr, ytr = train[feature_cols], train['target_next_return']
    Xte, yte = test[feature_cols], test['target_next_return'].to_numpy()
    models = {
      'zero_return': None,
      'training_mean': float(ytr.mean()),
      'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
      'random_forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.8, random_state=42, n_jobs=-1)
    }
    preds = {}; rows = []
    for name, model in models.items():
        if name == 'zero_return': p = np.zeros(len(test))
        elif name == 'training_mean': p = np.full(len(test), model)
        else:
            model.fit(Xtr, ytr)
            p = model.predict(Xte)
        preds[name] = np.asarray(p)
        rows.append({'model':name, 'n_test':len(yte), 'MAE':mean_absolute_error(yte,p), 'RMSE':float(np.sqrt(mean_squared_error(yte,p))), 'R2_oos_vs_test_mean':r2_score(yte,p), 'directional_accuracy':float(np.mean(np.sign(yte)==np.sign(p))), 'mean_prediction':float(np.mean(p)), 'prediction_std':float(np.std(p))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'metrics.csv', index=False)
    forecast = test[['Date','target_next_return']].rename(columns={'target_next_return':'actual_return'}).copy()
    for name,p in preds.items(): forecast[f'pred_{name}'] = p
    forecast.to_csv(out/'test_forecasts.csv', index=False)
    # Compare each learned model with zero-return using aligned squared-error losses.
    dm = {}
    for name,p in preds.items():
        if name == 'zero_return': continue
        dm[f'{name}_vs_zero_return'] = hac_dm((yte-p)**2, yte**2)
    (out/'statistical_tests.json').write_text(json.dumps(dm, indent=2))
    metadata = {'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,'raw_rows':len(raw),'model_rows_after_features':n,'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'first_date':str(frame.Date.min().date()),'last_date':str(frame.Date.max().date()),'train_end':str(train.Date.max().date()),'validation_end':str(val.Date.max().date()),'test_start':str(test.Date.min().date()),'test_end':str(test.Date.max().date()),'features':feature_cols,'random_seed':42,'note':'No hyperparameter tuning; validation segment reserved for later model selection. HAC normal approximation is a preliminary loss-differential test, not a definitive finite-sample DM test.'}
    (out/'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(metrics.to_string(index=False))
    print(f'Wrote results to {out.resolve()}')

if __name__ == '__main__': main()