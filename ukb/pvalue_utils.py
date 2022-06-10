"""
This file describes the Rubin's rules we have used in our downstream analysis
"""


import numpy as np
import pandas as pd
import scipy.stats as stats

def pool_analysis(parameters, covariances, return_pvalues=True):
    """
    parameters: ndarray of size (d, M) of estimators computed from the inputed data, where M is the number
                    of imputations
    covariances: list of covariance estimators associated with the parameters (same shape)
    """
    index = parameters.index
    # The mean estimator
    param_mean = parameters.mean(axis=1)

    # The average of the within-imputation covariances
    m = parameters.shape[1]
    cov_within = np.mean(covariances, axis=1)

    # The between-imputation covariance
    cov_between = np.var(parameters, axis=1)

    ## compute the p-value from the multi-dimensional Wald test
    Q_m = param_mean
    U_m = cov_within
    B_m = cov_between

    # compute the variance estimator
    T_m_pos = (1 + 1/m) * B_m + U_m # always positive, conservative
    T_m = (1 + 1/m) * B_m - U_m # can be negative
    T_m = np.where(T_m > 0, T_m, T_m_pos) # pick the conservative estimate for negative variances
    assert(np.all(T_m > 0))

    # compute pooled p-values
    v = (m - 1.)*(1. - m * U_m / ((m+1)*B_m))**2 # degree of freedom
    #pvalues = stats.f(1, v).sf(Q_m**2/T_m) # p-values from f-distribution
    pvalues = stats.chi2(1, v).sf(Q_m**2/T_m) # p-values from chi-squared.  Double check which one statsmodels uses

    #return pd.Series(pvalues, index=index), Q_m, T_m
    if return_pvalues:
        return (Q_m, pd.Series(T_m, index=index)), pd.Series(pvalues, index=index) 
    else: return Q_m, pd.Series(T_m, index=index)
