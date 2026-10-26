"""Synthetic regression test for the train-only Granger feature screen."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_robust_experiments import graph_screen, hac

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

    # Validate HAC statistic against statsmodels' intercept-only Newey-West covariance.
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    from scipy.stats import t as student_t
    rng=np.random.default_rng(9917)
    eps=rng.normal(size=1200)
    d=np.zeros_like(eps)
    for k in range(1,len(d)):
        d[k]=0.35*d[k-1]+eps[k]+0.12
    got=hac(d,np.zeros_like(d),L=5)
    fit=sm.OLS(d,np.ones((len(d),1))).fit()
    variance=float(cov_hac(fit,nlags=5,use_correction=True)[0,0])
    expected_stat=float(d.mean()/np.sqrt(variance))
    assert np.isclose(got["statistic"],expected_stat,rtol=1e-10,atol=1e-10), (got,expected_stat)
    expected_p=float(2*student_t.sf(abs(expected_stat),df=len(d)-1))
    assert np.isclose(got["p_value"],expected_p,rtol=1e-10,atol=1e-12), (got,expected_p)
    assert got["mean_loss_difference_A_minus_B"]>0, "Positive A-B loss difference must mean model A has higher loss."
    # Bandwidth sensitivity should be computable and report the actual lag count.
    for bandwidth in [0,5,10,20]:
        result_hac=hac(d,np.zeros_like(d),L=bandwidth)
        assert result_hac["n"]==len(d) and result_hac["hac_lags"]==bandwidth
        assert np.isfinite(result_hac["statistic"]) and 0<=result_hac["p_value"]<=1
    print("PASS: HAC statistic and t-reference p-value match statsmodels Newey-West covariance; lag sensitivity checks passed.")

if __name__=="__main__":
    main()
