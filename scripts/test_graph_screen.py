"""Synthetic regression test for the train-only Granger feature screen."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_robust_experiments import graph_screen

def main():
    rng=np.random.default_rng(20261026)
    n=1800
    x=rng.normal(size=n)
    noise=rng.normal(scale=0.5,size=n)
    # y[t] depends on x[t-1], so x should Granger-predict y.
    y=np.zeros(n)
    for t in range(1,n):
        y[t]=0.85*x[t-1]+noise[t]
    z=rng.normal(size=n)
    df=pd.DataFrame({"target":y,"known_driver":x,"independent_noise":z})
    result=graph_screen(df,["known_driver","independent_noise"],maxlag=3).set_index("feature")
    assert result.loc["known_driver","test_error"]=="", result.loc["known_driver","test_error"]
    assert result.loc["known_driver","min_p"]<1e-8, result.to_string()
    assert result.loc["known_driver","fdr_q"]<0.05, result.to_string()
    assert result.loc["known_driver","selected"], result.to_string()
    assert result.loc["independent_noise","test_error"]=="", result.loc["independent_noise","test_error"]
    assert not result.loc["independent_noise","selected"], result.to_string()
    print("PASS: known lagged driver selected; independent noise not selected.")
    print(result[["min_p","lag_adjusted_p","fdr_q","selected","test_error"]].to_string())

if __name__=="__main__":
    main()
