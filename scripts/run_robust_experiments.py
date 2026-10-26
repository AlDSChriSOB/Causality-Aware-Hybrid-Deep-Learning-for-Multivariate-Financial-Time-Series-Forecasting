"""Leakage-aware Paper 2 experiments. Graph edges are train-only Granger predictive dependencies, not proof of structural causality."""
import argparse,json,platform,hashlib,warnings
from pathlib import Path
import numpy as np,pandas as pd,sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
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

def graph_screen(train,features):
 from statsmodels.tsa.stattools import grangercausalitytests
 rows=[]
 for col in features:
  try:
   with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res=grangercausalitytests(train[["target",col]],maxlag=5,verbose=False)
   pv=min(float(res[k][0]["ssr_ftest"][1]) for k in range(1,6))
  except Exception: pv=1.
  rows.append({"feature":col,"min_p":pv})
 q=bh([r["min_p"] for r in rows])
 for r,v in zip(rows,q): r["fdr_q"]=float(v); r["selected"]=bool(v<.05)
 if not any(r["selected"] for r in rows): rows[int(np.argmin(q))]["selected"]=True
 return pd.DataFrame(rows)

def met(y,p):
 y=np.asarray(y); p=np.asarray(p); nz=np.abs(p)>1e-12
 return {"n":int(len(y)),"MAE":float(mean_absolute_error(y,p)),"RMSE":float(np.sqrt(mean_squared_error(y,p))),
 "R2_vs_test_mean":float(r2_score(y,p)),"directional_accuracy_nonzero":float(np.mean(np.sign(y[nz])==np.sign(p[nz]))) if nz.any() else None,
 "directional_coverage":float(nz.mean()),"prediction_mean":float(p.mean()),"prediction_std":float(p.std())}

def hac(a,b,L=10):
 d=np.asarray(a)-np.asarray(b); n=len(d); mu=float(d.mean()); z=d-mu; L=min(L,n-1); v=np.dot(z,z)/n
 for k in range(1,L+1): v+=2*(1-k/(L+1))*np.dot(z[k:],z[:-k])/n
 if v<=0: return {"mean_loss_difference_A_minus_B":mu,"p_value":None,"n":n}
 t=mu/np.sqrt(v/n)
 return {"mean_loss_difference_A_minus_B":mu,"statistic":float(t),"p_value":float(2*norm.sf(abs(t))),"n":n,"hac_lags":L}

def mlp(hidden,alpha):
 return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=hidden,alpha=alpha,max_iter=350,early_stopping=False,random_state=42,learning_rate_init=.001))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data",default="SP500.csv"); ap.add_argument("--out",default="results/advanced"); ap.add_argument("--rolling-folds",type=int,default=5); a=ap.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); frame,raw=build(pd.read_csv(a.data))
 n=len(frame); i=int(.6*n); j=int(.8*n); tr,va,te=frame.iloc[:i],frame.iloc[i:j],frame.iloc[j:]
 feats=[c for c in frame if c not in ["Date","target"]]; X,y=tr[feats],tr.target.to_numpy(); V,vy=va[feats],va.target.to_numpy(); T,ty=te[feats],te.target.to_numpy()
 # Tune AR lag order using validation only; dynamic=False gives one-step rolling predictions.
 from statsmodels.tsa.ar_model import AutoReg
 arval={}
 for lag in [1,2,5,10]:
  try:
   fit=AutoReg(tr.target,lags=lag,trend="c",old_names=False).fit()
   arval[lag]=float(mean_squared_error(vy,fit.predict(start=len(tr),end=len(tr)+len(va)-1,dynamic=False)))
  except Exception: pass
 lag=min(arval,key=arval.get) if arval else 1
 configs=[((16,),.001),((32,),.001),((32,16),.001),((32,16),.01)]; tune=[]
 for h,alpha in configs:
  m=mlp(h,alpha); m.fit(X,y); tune.append({"hidden":list(h),"alpha":alpha,"validation_MSE":float(mean_squared_error(vy,m.predict(V)))})
 best=min(tune,key=lambda z:z["validation_MSE"]); hidden=tuple(best["hidden"]); alpha=best["alpha"]
 graph=graph_screen(tr,feats); selected=graph.loc[graph.selected,"feature"].tolist()
 full=mlp(hidden,alpha).fit(X,y); gm=mlp(hidden,alpha).fit(X[selected],y)
 ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(X,y)
 rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=10,max_features=.8,random_state=42,n_jobs=-1).fit(X,y)
 tv=pd.concat([tr,va]); ar=AutoReg(tv.target,lags=lag,trend="c",old_names=False).fit()
 preds={"zero_return":np.zeros(len(te)),"training_mean":np.full(len(te),y.mean()),
 "AR_selected":np.asarray(ar.predict(start=len(tv),end=len(tv)+len(te)-1,dynamic=False)),
 "ridge":ridge.predict(T),"random_forest":rf.predict(T),"MLP_full_features":full.predict(T),"MLP_graph_masked":gm.predict(T[selected])}
 rng=np.random.default_rng(42); rnd=list(rng.choice(feats,size=len(selected),replace=False)); rn=mlp(hidden,alpha).fit(X[rnd],y); preds["MLP_random_mask_ablation"]=rn.predict(T[rnd])
 rows=[]
 for name,p in preds.items(): rows.append({"model":name,"features_used":len(selected) if "graph_masked" in name else len(rnd) if "random_mask" in name else len(feats),**met(ty,p)})
 pd.DataFrame(rows).to_csv(out/"test_metrics.csv",index=False); graph.to_csv(out/"training_graph_edges.csv",index=False); pd.DataFrame(tune).to_csv(out/"neural_validation_tuning.csv",index=False)
 fc=pd.DataFrame({"Date":te.Date,"actual_next_return":ty,**preds}); fc.to_csv(out/"test_forecasts.csv",index=False)
 # Expanding-window rolling-origin evaluation, fixed hyperparameters selected on initial validation.
 rolls=[]
 for fold,idx in enumerate(np.array_split(np.arange(j,n),a.rolling_folds),1):
  if not len(idx): continue
  hist=frame.iloc[:int(idx[0])]; block=frame.iloc[idx]; xx=hist[feats]; yy=hist.target.to_numpy(); xt=block[feats]; yt=block.target.to_numpy()
  for name,cols in [("ridge",feats),("MLP_full_features",feats),("MLP_graph_masked",selected)]:
   model=make_pipeline(StandardScaler(),Ridge(alpha=1.)) if name=="ridge" else mlp(hidden,alpha)
   model.fit(xx[cols],yy); rolls.append({"fold":fold,"start":str(block.Date.iloc[0].date()),"end":str(block.Date.iloc[-1].date()),"model":name,**met(yt,model.predict(xt[cols]))})
 pd.DataFrame(rolls).to_csv(out/"rolling_origin_metrics.csv",index=False)
 regs=[]
 for start,end,label in [("2010-07-09","2019-12-31","pre-COVID"),("2020-01-01","2021-12-31","COVID-era"),("2022-01-01","2025-12-31","recent")]:
  mask=(te.Date>=start)&(te.Date<=end)
  if mask.sum()>=10:
   for name,p in preds.items(): regs.append({"regime":label,"start":start,"end":end,"model":name,**met(ty[mask],p[mask])})
 pd.DataFrame(regs).to_csv(out/"regime_metrics.csv",index=False)
 tests={name+"_vs_zero_return":hac((ty-p)**2,(ty-preds["zero_return"])**2) for name,p in preds.items() if name!="zero_return"}
 tests["graph_masked_vs_full_MLP"]=hac((ty-preds["MLP_graph_masked"])**2,(ty-preds["MLP_full_features"])**2)
 (out/"forecast_comparison_tests.json").write_text(json.dumps(tests,indent=2))
 # Figures
 fig,ax=plt.subplots(figsize=(10,5)); ax.plot(te.Date,ty,label="Actual",lw=.7); ax.plot(te.Date,preds["MLP_full_features"],label="MLP all features",lw=.7); ax.plot(te.Date,preds["MLP_graph_masked"],label="Graph-masked MLP",lw=.7); ax.set(title="S&P 500 next-day log-return forecasts",xlabel="Date",ylabel="Log return"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"figure_test_forecasts.png",dpi=300); plt.close(fig)
 mr=pd.DataFrame(rows).sort_values("RMSE"); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(mr.model,mr.RMSE); ax.set(title="Held-out forecast error",ylabel="RMSE",xlabel="Model"); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(out/"figure_rmse_comparison.png",dpi=300); plt.close(fig)
 gg=graph.sort_values("fdr_q"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(gg.feature,-np.log10(np.maximum(gg.fdr_q,1e-300))); ax.axvline(-np.log10(.05),ls="--",lw=1); ax.set(title="Training-only Granger screen",xlabel="-log10(BH-adjusted q)",ylabel="Feature"); fig.tight_layout(); fig.savefig(out/"figure_training_graph_screen.png",dpi=300); plt.close(fig)
 bestrow=min(rows,key=lambda z:z["RMSE"]); fullrow=next(z for z in rows if z["model"]=="MLP_full_features"); grow=next(z for z in rows if z["model"]=="MLP_graph_masked")
 report=f"""# Paper 2 — Automatically generated results snapshot

## Forecast definition and split
Forecast origin is after the close on date t; target is log(Close[t+1])-log(Close[t]).
Rows: {len(tr)} training, {len(va)} validation, {len(te)} untouched test observations.
Test period: {te.Date.min().date()} to {te.Date.max().date()}. AR order and MLP settings were selected using validation only.

## Held-out evidence
Best RMSE among tested candidates: {bestrow['model']} (RMSE {bestrow['RMSE']:.7g}; MAE {bestrow['MAE']:.7g}; R-squared against test mean {bestrow['R2_vs_test_mean']:.4f}).
Full-feature MLP RMSE: {fullrow['RMSE']:.7g}. Graph-masked MLP RMSE: {grow['RMSE']:.7g}.
Consult test_metrics.csv, test_forecasts.csv and forecast_comparison_tests.json for the complete measured results.

## Interpretation
The graph is estimated only from training data, with Benjamini-Hochberg adjustment across features, and masks inputs to the MLP; therefore it changes model computation. The screen detects Granger predictive dependence, not structural causal effects. Forecast comparison p-values use a HAC-normal approximation and are preliminary; multiple comparisons are not adjusted. Rolling-origin and calendar-regime results are robustness descriptions, not causal estimates. Directional accuracy excludes exactly zero predictions and reports coverage separately.

## Outputs
test_metrics.csv; test_forecasts.csv; training_graph_edges.csv; neural_validation_tuning.csv; rolling_origin_metrics.csv; regime_metrics.csv; forecast_comparison_tests.json; three 300-dpi figures.
"""
 (out/"RESULTS_AND_DISCUSSION.md").write_text(report)
 meta={"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,"raw_rows":len(raw),"analysis_rows":n,"train_rows":len(tr),"validation_rows":len(va),"test_rows":len(te),"first_date":str(frame.Date.min().date()),"last_date":str(frame.Date.max().date()),"train_end":str(tr.Date.max().date()),"validation_end":str(va.Date.max().date()),"test_start":str(te.Date.min().date()),"test_end":str(te.Date.max().date()),"target":"log(Close[t+1])-log(Close[t])","forecast_origin":"after close t","features":feats,"graph_selected_features":selected,"graph_semantics":"train-only Granger predictive-dependency screen; not structural causality","selected_AR_lag":lag,"AR_validation_MSE":arval,"selected_MLP":best,"MLP_validation_grid":tune,"test_used_for_tuning":False,"directional_accuracy":"non-zero predictions only; coverage also reported","input_sha256":hashlib.sha256(Path(a.data).read_bytes()).hexdigest()}
 (out/"run_metadata.json").write_text(json.dumps(meta,indent=2,default=str))
 print(f"Completed: best test RMSE model {bestrow['model']} ({bestrow['RMSE']:.7g}); selected graph features {len(selected)}")
if __name__=="__main__": main()
