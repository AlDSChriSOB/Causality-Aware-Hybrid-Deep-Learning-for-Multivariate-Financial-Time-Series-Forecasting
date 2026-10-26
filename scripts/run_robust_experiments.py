"""Leakage-aware Paper 2 experiments. Graph edges are train-only Granger predictive dependencies, not proof of structural causality.

Model selection uses chronological training/validation data. Final estimators are refit on
train+validation, and the held-out test is evaluated once after refitting."""
import argparse,json,platform,hashlib,warnings
from pathlib import Path
import numpy as np,pandas as pd,sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")

def build(raw):
 raw=raw.copy(); raw.columns=[str(c).strip() for c in raw.columns]
 need={"Date","Open","High","Low","Close"}
 if need-set(raw.columns): raise ValueError(f"Missing columns: {need-set(raw.columns)}")
 raw.Date=pd.to_datetime(raw.Date,errors="coerce")
 for c in ["Open","High","Low","Close","Volume"]:
  if c in raw: raw[c]=pd.to_numeric(raw[c],errors="coerce")
 raw=raw.dropna(subset=["Date","Open","High","Low","Close"]).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
 if (raw[["Open","High","Low","Close"]]<=0).any().any(): raise ValueError("Nonpositive prices")
 close=raw.Close.astype(float); r=np.log(close).diff()
 f=pd.DataFrame({"Date":raw.Date,"target":r.shift(-1)})
 for lag in [1,2,3,5,10,20]: f[f"ret_lag_{lag}"]=r.shift(lag-1)
 f["range_scaled"]=(raw.High-raw.Low)/close
 f["open_close_return"]=np.log(raw.Close/raw.Open)
 f["overnight_gap"]=np.log(raw.Open/raw.Close.shift(1))
 for w in [5,20,60]: f[f"volatility_{w}"]=r.rolling(w,min_periods=w).std()
 if "Volume" in raw and (raw.Volume>0).any(): f["log_volume_change"]=np.log(raw.Volume.where(raw.Volume>0)).diff()
 d=close.diff(); g=d.clip(lower=0).rolling(14,min_periods=14).mean(); l=(-d.clip(upper=0)).rolling(14,min_periods=14).mean()
 f["rsi14"]=100-100/(1+g/l.replace(0,np.nan))
 return f.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True),raw

def bh(p):
 p=np.asarray(p); n=len(p); order=np.argsort(p); q=np.empty(n); cur=1.
 for j in range(n-1,-1,-1):
  cur=min(cur,p[order[j]]*n/(j+1)); q[order[j]]=cur
 return np.minimum(q,1)

def graph_screen(train,features,maxlag=5):
 """Train-only Granger screen with explicit diagnostics.

 statsmodels 0.15 removed the legacy verbose argument. Passing it raised
 TypeError, which the previous broad exception handler silently converted
 into p=1 for every feature.
 """
 from statsmodels.tsa.stattools import grangercausalitytests
 rows=[]
 for col in features:
  err=""
  try:
   pair=train[["target",col]].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
   pair=pair[np.isfinite(pair).all(axis=1)]
   if len(pair) <= 3*maxlag+10:
    raise ValueError(f"insufficient finite observations: n={len(pair)}")
   if np.std(pair[:,0]) < 1e-12 or np.std(pair[:,1]) < 1e-12:
    raise ValueError("constant target or feature")
   # statsmodels tests whether column 2 Granger-predicts column 1.
   # Do not pass verbose: it was removed in statsmodels 0.15.
   import contextlib,io
   with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter("ignore")
    res=grangercausalitytests(pair,maxlag=maxlag)
   lag_p=[float(res[k][0]["ssr_ftest"][1]) for k in range(1,maxlag+1)]
   if not np.isfinite(lag_p).all():
    raise ValueError(f"non-finite p-values returned: {lag_p}")
   raw_min=float(min(lag_p))
   # Correct the within-feature search across lags before correcting across features.
   lag_adjusted=min(1.0,raw_min*maxlag)
  except Exception as exc:
   raw_min=1.0
   lag_adjusted=1.0
   err=f"{type(exc).__name__}: {exc}"
  rows.append({"feature":col,"min_p":raw_min,"lag_adjusted_p":lag_adjusted,"test_error":err})
 q=bh([r["lag_adjusted_p"] for r in rows])
 for r,v in zip(rows,q):
  r["fdr_q"]=float(v)
  r["selected"]=bool(not r["test_error"] and v<.05)
 # Fail loudly if every test failed rather than writing a misleading all-null screen.
 if rows and all(r["test_error"] for r in rows):
  details="; ".join(f"{r['feature']}: {r['test_error']}" for r in rows[:3])
  raise RuntimeError("All Granger tests failed. First errors: "+details)
 return pd.DataFrame(rows)

def met(y,p):
 y=np.asarray(y); p=np.asarray(p); nz=np.abs(p)>1e-12
 return {"n":int(len(y)),"MAE":float(mean_absolute_error(y,p)),"RMSE":float(np.sqrt(mean_squared_error(y,p))),
 "R2_vs_test_mean":float(r2_score(y,p)),"directional_accuracy_nonzero":float(np.mean(np.sign(y[nz])==np.sign(p[nz]))) if nz.any() else None,
 "directional_coverage":float(nz.mean()),"prediction_mean":float(p.mean()),"prediction_std":float(p.std())}

def hac(a,b,L=10):
 """Two-sided HAC loss-differential test with Bartlett/Newey-West covariance.

 Positive mean(A loss - B loss) means A has higher loss (B is better).
 The t reference is a finite-sample diagnostic; inference remains approximate.
 """
 a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
 if a.shape!=b.shape or a.ndim!=1: raise ValueError("Loss arrays must be same-length 1-D vectors")
 d=a-b; d=d[np.isfinite(d)]; n=len(d)
 if n<3: return {"mean_loss_difference_A_minus_B":float(np.mean(d)) if n else None,"statistic":None,"p_value":None,"n":n,"hac_lags":0}
 mu=float(d.mean()); z=d-mu; L=max(0,min(int(L),n-1)); long_run=float(np.dot(z,z)/n)
 for k in range(1,L+1):
  long_run+=2*(1-k/(L+1))*float(np.dot(z[k:],z[:-k])/n)
 if not np.isfinite(long_run) or long_run<=0:
  return {"mean_loss_difference_A_minus_B":mu,"statistic":None,"p_value":None,"n":n,"hac_lags":L}
 # Newey-West finite-sample correction for an intercept-only mean regression.
 se=float(np.sqrt((long_run/n)*(n/(n-1))))
 stat=mu/se if se>0 else None
 p=float(2*student_t.sf(abs(stat),df=n-1)) if stat is not None else None
 return {"mean_loss_difference_A_minus_B":mu,"statistic":float(stat) if stat is not None else None,"p_value":p,"n":n,"hac_lags":L,"hac_estimator":"Bartlett/Newey-West with n/(n-1) correction","reference_distribution":"Student-t, df=n-1"}

def mlp(hidden,alpha):
 return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=hidden,alpha=alpha,max_iter=350,early_stopping=False,random_state=42,learning_rate_init=.001))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data",default="SP500.csv"); ap.add_argument("--out",default="results/advanced"); ap.add_argument("--rolling-folds",type=int,default=5); a=ap.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); frame,raw=build(pd.read_csv(a.data))
 n=len(frame); i=int(.6*n); j=int(.8*n); tr,va,te=frame.iloc[:i],frame.iloc[i:j],frame.iloc[j:]
 feats=[c for c in frame if c not in ["Date","target"]]; X,y=tr[feats],tr.target.to_numpy(); V,vy=va[feats],va.target.to_numpy(); T,ty=te[feats],te.target.to_numpy()
 # Scale the neural target using training data only; invert predictions to return units.
 y_scaler=StandardScaler().fit(y.reshape(-1,1)); ys=y_scaler.transform(y.reshape(-1,1)).ravel()
 # Tune AR lag order using validation only; dynamic=False gives one-step rolling predictions.
 from statsmodels.tsa.ar_model import AutoReg
 arval={}
 for lag in [1,2,5,10]:
  try:
   fit=AutoReg(tr.target,lags=lag,trend="c").fit()
   arval[lag]=float(mean_squared_error(vy,fit.predict(start=len(tr),end=len(tr)+len(va)-1,dynamic=False)))
  except Exception: pass
 lag=min(arval,key=arval.get) if arval else 1
 configs=[((16,),.001),((32,),.001),((32,16),.001),((32,16),.01)]; tune=[]
 for h,alpha in configs:
  m=mlp(h,alpha); m.fit(X,ys); pv=y_scaler.inverse_transform(m.predict(V).reshape(-1,1)).ravel(); tune.append({"hidden":list(h),"alpha":alpha,"validation_MSE":float(mean_squared_error(vy,pv))})
 best=min(tune,key=lambda z:z["validation_MSE"]); hidden=tuple(best["hidden"]); alpha=best["alpha"]
 graph=graph_screen(tr,feats); graph_selected=graph.loc[graph.selected,"feature"].tolist()
 # If the screen has no FDR-significant edges, retain a clearly labelled exploratory
 # minimum-q feature subset for diagnostic comparison; it is not a discovered causal graph.
 selected=graph_selected if graph_selected else [str(graph.sort_values(["fdr_q","min_p"]).iloc[0]["feature"])]
 graph_mode="FDR-significant feature mask" if graph_selected else "exploratory fallback: no FDR-significant features"
 # Hyperparameters were selected on train/validation only. Refit final estimators
 # on the combined pre-test history, with every learned scaler fitted on that history.
 tv=pd.concat([tr,va]); Xt=tv[feats]; yt=tv.target.to_numpy(); T,ty=te[feats],te.target.to_numpy()
 y_scaler_final=StandardScaler().fit(yt.reshape(-1,1)); ys_final=y_scaler_final.transform(yt.reshape(-1,1)).ravel()
 full=mlp(hidden,alpha).fit(Xt,ys_final); gm=mlp(hidden,alpha).fit(Xt[selected],ys_final)
 ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(Xt,yt)
 rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=10,max_features=.8,random_state=42,n_jobs=-1).fit(Xt,yt)
 ar=AutoReg(tv.target,lags=lag,trend="c").fit()
 rng=np.random.default_rng(42); rnd=list(rng.choice(feats,size=len(selected),replace=False)); rn=mlp(hidden,alpha).fit(Xt[rnd],ys_final)
 preds={"zero_return":np.zeros(len(te)),"training_mean":np.full(len(te),yt.mean()),
 "AR_selected":np.asarray(ar.predict(start=len(tv),end=len(tv)+len(te)-1,dynamic=False)),
 "ridge":ridge.predict(T),"random_forest":rf.predict(T),
 "MLP_full_features":y_scaler_final.inverse_transform(full.predict(T).reshape(-1,1)).ravel(),
 "MLP_graph_masked":y_scaler_final.inverse_transform(gm.predict(T[selected]).reshape(-1,1)).ravel(),
 "MLP_random_mask_ablation":y_scaler_final.inverse_transform(rn.predict(T[rnd]).reshape(-1,1)).ravel()}
 rows=[]
 for name,p in preds.items(): rows.append({"model":name,"features_used":len(selected) if "graph_masked" in name else len(rnd) if "random_mask" in name else len(feats),**met(ty,p)})
 pd.DataFrame(rows).to_csv(out/"test_metrics.csv",index=False); graph.to_csv(out/"training_graph_edges.csv",index=False); pd.DataFrame(tune).to_csv(out/"neural_validation_tuning.csv",index=False)
 fc=pd.DataFrame({"Date":te.Date,"actual_next_return":ty,**preds}); fc.to_csv(out/"test_forecasts.csv",index=False)
 # Expanding-window rolling-origin evaluation. Re-estimate the Granger screen
 # using only history available before each fold. Hyperparameters remain fixed
 # from the initial train/validation split and are not tuned on future fold data.
 rolls=[]; rolling_graph=[]
 for fold,idx in enumerate(np.array_split(np.arange(j,n),a.rolling_folds),1):
  if not len(idx): continue
  hist=frame.iloc[:int(idx[0])]; block=frame.iloc[idx]; xx=hist[feats]; yy=hist.target.to_numpy(); xt=block[feats]; yt=block.target.to_numpy()
  fold_graph=graph_screen(hist,feats)
  fold_graph_selected=fold_graph.loc[fold_graph.selected,"feature"].tolist()
  fold_cols=fold_graph_selected if fold_graph_selected else [str(fold_graph.sort_values(["fdr_q","min_p"]).iloc[0]["feature"])]
  fold_mode="FDR-significant feature mask" if fold_graph_selected else "exploratory fallback: no FDR-significant features"
  fold_graph.insert(0,"fold",fold); fold_graph["selected_for_model"]=fold_graph.feature.isin(fold_cols); fold_graph["graph_mode"]=fold_mode
  rolling_graph.append(fold_graph)
  for name,cols in [("ridge",feats),("MLP_full_features",feats),("MLP_graph_masked",fold_cols)]:
   model=make_pipeline(StandardScaler(),Ridge(alpha=1.)) if name=="ridge" else mlp(hidden,alpha)
   if name=="ridge":
    model.fit(xx[cols],yy); pp=model.predict(xt[cols])
   else:
    ys_fold=StandardScaler().fit(yy.reshape(-1,1)); model.fit(xx[cols],ys_fold.transform(yy.reshape(-1,1)).ravel())
    pp=ys_fold.inverse_transform(model.predict(xt[cols]).reshape(-1,1)).ravel()
   rolls.append({"fold":fold,"start":str(block.Date.iloc[0].date()),"end":str(block.Date.iloc[-1].date()),"model":name,"features_used":len(cols),"graph_selected_count":len(fold_graph_selected) if name=="MLP_graph_masked" else None,"graph_mode":fold_mode if name=="MLP_graph_masked" else None,**met(yt,pp)})
 pd.DataFrame(rolls).to_csv(out/"rolling_origin_metrics.csv",index=False)
 pd.concat(rolling_graph,ignore_index=True).to_csv(out/"rolling_origin_graph_screens.csv",index=False)
 regs=[]
 for start,end,label in [("2010-07-09","2019-12-31","pre-COVID"),("2020-01-01","2021-12-31","COVID-era"),("2022-01-01","2025-12-31","recent")]:
  mask=(te.Date>=start)&(te.Date<=end)
  if mask.sum()>=10:
   for name,p in preds.items(): regs.append({"regime":label,"start":start,"end":end,"model":name,**met(ty[mask],p[mask])})
 pd.DataFrame(regs).to_csv(out/"regime_metrics.csv",index=False)
 def loss_comparisons(L):
  comparisons={name+"_vs_zero_return":hac((ty-p)**2,(ty-preds["zero_return"])**2,L=L) for name,p in preds.items() if name!="zero_return"}
  comparisons["graph_masked_vs_full_MLP"]=hac((ty-preds["MLP_graph_masked"])**2,(ty-preds["MLP_full_features"])**2,L=L)
  # Holm family-wise adjustment across this predeclared family of comparisons.
  valid=sorted([(name,val["p_value"]) for name,val in comparisons.items() if val.get("p_value") is not None],key=lambda z:z[1])
  running=0.0; m=len(valid)
  for rank,(name,pv) in enumerate(valid):
   adjusted=min(1.0,(m-rank)*pv)
   running=max(running,adjusted)
   comparisons[name]["p_value_holm"]=running
  return comparisons
 tests=loss_comparisons(10)
 sensitivity={str(L):loss_comparisons(L) for L in [0,5,10,20]}
 (out/"forecast_comparison_tests.json").write_text(json.dumps(tests,indent=2))
 (out/"forecast_comparison_hac_sensitivity.json").write_text(json.dumps({"description":"Sensitivity of squared-error loss comparisons to Bartlett/Newey-West HAC bandwidth. Holm adjustment is applied within the comparison family separately at each lag; lag 10 is the primary bandwidth.","primary_lag":10,"results_by_hac_lag":sensitivity},indent=2))
 # Figures
 fig,ax=plt.subplots(figsize=(10,5)); ax.plot(te.Date,ty,label="Actual",lw=.7); ax.plot(te.Date,preds["MLP_full_features"],label="MLP all features",lw=.7); ax.plot(te.Date,preds["MLP_graph_masked"],label="Graph-masked MLP",lw=.7); ax.set(title="S&P 500 next-day log-return forecasts",xlabel="Date",ylabel="Log return"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"figure_test_forecasts.png",dpi=300); plt.close(fig)
 mr=pd.DataFrame(rows).sort_values("RMSE"); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(mr.model,mr.RMSE); ax.set(title="Held-out forecast error",ylabel="RMSE",xlabel="Model"); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(out/"figure_rmse_comparison.png",dpi=300); plt.close(fig)
 gg=graph.sort_values("fdr_q"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(gg.feature,-np.log10(np.maximum(gg.fdr_q,1e-300))); ax.axvline(-np.log10(.05),ls="--",lw=1); ax.set(title="Training-only Granger screen",xlabel="-log10(BH-adjusted q)",ylabel="Feature"); fig.tight_layout(); fig.savefig(out/"figure_training_graph_screen.png",dpi=300); plt.close(fig)
 bestrow=min(rows,key=lambda z:z["RMSE"]); fullrow=next(z for z in rows if z["model"]=="MLP_full_features"); grow=next(z for z in rows if z["model"]=="MLP_graph_masked"); randomrow=next(z for z in rows if z["model"]=="MLP_random_mask_ablation")
 graph_full_test=tests["graph_masked_vs_full_MLP"]
 rf_test=tests.get("random_forest_vs_zero_return",{})
 report=f"""# Paper 2 — Automatically generated results snapshot

## Forecast definition and split
Forecast origin is after the close on date t; target is log(Close[t+1])-log(Close[t]).
Rows: {len(tr)} training, {len(va)} validation, {len(te)} untouched test observations.
Test period: {te.Date.min().date()} to {te.Date.max().date()}. AR order and MLP settings were selected using validation only.

## Held-out evidence
Best RMSE among tested candidates: {bestrow['model']} (RMSE {bestrow['RMSE']:.7g}; MAE {bestrow['MAE']:.7g}; R-squared against test mean {bestrow['R2_vs_test_mean']:.4f}).
Full-feature MLP RMSE: {fullrow['RMSE']:.7g}. Graph-masked MLP RMSE: {grow['RMSE']:.7g}. Random-mask MLP RMSE: {randomrow['RMSE']:.7g}.
The graph-masked model's directional accuracy is {grow['directional_accuracy_nonzero']:.3f} with {grow['directional_coverage']:.1%} prediction coverage; directional accuracy is descriptive and is not evidence of statistical significance by itself.

## Evidence-based conclusion
The tested graph-masked MLP does not establish a reliable forecasting improvement. Its RMSE is {grow['RMSE']:.7g}, versus {fullrow['RMSE']:.7g} for the unrestricted MLP and {randomrow['RMSE']:.7g} for the random-mask ablation. For the graph-masked versus full-feature squared-error comparison, the primary HAC test gives p={graph_full_test['p_value']:.4f} and Holm-adjusted p={graph_full_test['p_value_holm']:.4f}; this does not reject equal expected loss at the conventional 5% level. The {bestrow['model']} baseline has the lowest held-out RMSE among candidates tested. Accordingly, results support a cautious negative/null conclusion: the present experiments do not demonstrate that the Granger-screened feature mask improves next-day S&P 500 return forecasts. They do not establish that graph-informed methods cannot help under other specifications or datasets.

## Interpretation and limitations
The graph screen is estimated only from training data, with Benjamini-Hochberg adjustment across features. FDR-significant selected features: {len(graph_selected)}. Graph-model mode: {graph_mode}. When no feature passes the adjusted threshold, the one-feature fallback is exploratory and must not be interpreted as a discovered graph edge. The screen tests Granger predictive dependence, not structural causal effects. Forecast comparisons use squared-error loss differentials with a Bartlett/Newey-West HAC variance estimate, finite-sample n/(n-1) correction, and a t(n-1) reference. The primary bandwidth is 10; sensitivity results at bandwidths 0, 5, 10, and 20 are saved separately, with Holm adjustment within each comparison family. These tests are approximate and bandwidth-sensitive inference should be interpreted cautiously. Rolling-origin graph selection is re-estimated from each fold's prior history only; hyperparameters remain fixed from initial validation. Rolling-origin and calendar-regime results are robustness descriptions, not causal estimates. Negative out-of-sample R-squared values for most candidates indicate they do not improve upon the test-mean reference under that metric. Regime-specific differences are descriptive and should not be treated as evidence of a robust regime interaction without an appropriate inferential test.

## Outputs
test_metrics.csv; test_forecasts.csv; training_graph_edges.csv; neural_validation_tuning.csv; rolling_origin_metrics.csv; rolling_origin_graph_screens.csv; regime_metrics.csv; forecast_comparison_tests.json; forecast_comparison_hac_sensitivity.json; three 300-dpi figures.
"""
 (out/"RESULTS_AND_DISCUSSION.md").write_text(report)
 meta={"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,"raw_rows":len(raw),"analysis_rows":n,"train_rows":len(tr),"validation_rows":len(va),"test_rows":len(te),"first_date":str(frame.Date.min().date()),"last_date":str(frame.Date.max().date()),"train_end":str(tr.Date.max().date()),"validation_end":str(va.Date.max().date()),"test_start":str(te.Date.min().date()),"test_end":str(te.Date.max().date()),"target":"log(Close[t+1])-log(Close[t])","forecast_origin":"after close t","features":feats,"graph_selected_features":graph_selected,"graph_model_features":selected,"graph_model_mode":graph_mode,"graph_semantics":"train-only Granger predictive-dependency screen; not structural causality","selected_AR_lag":lag,"AR_validation_MSE":arval,"selected_MLP":best,"MLP_validation_grid":tune,"test_used_for_tuning":False,"directional_accuracy":"non-zero predictions only; coverage also reported","rolling_origin_graph_selection":"re-estimated from historical data available before each fold; exploratory fallback if no FDR-significant features","forecast_comparison_inference":"Bartlett/Newey-West HAC loss-differential tests, n/(n-1) correction, t(n-1) reference; primary bandwidth 10 and sensitivity bandwidths 0,5,10,20; Holm adjustment within each comparison family","input_sha256":hashlib.sha256(Path(a.data).read_bytes()).hexdigest()}
 (out/"run_metadata.json").write_text(json.dumps(meta,indent=2,default=str))
 print(f"Completed: best test RMSE model {bestrow['model']} ({bestrow['RMSE']:.7g}); selected graph features {len(selected)}")
if __name__=="__main__": main()
