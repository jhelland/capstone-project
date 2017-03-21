"""
Package of functions for MATH 482
"""

##################################################
# libraries
##################################################

import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.linalg as la
import pandas as pd
import timeit


##################################################
# functions
##################################################

def get_AUC(x, y):
    AUC = 0
    for i in range(1, x.size):
        AUC += np.abs(y[i] + y[i-1])*(x[i] - x[i-1])/2.0

    return AUC


def cov(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)

    return sum((x - xbar)*(y - ybar))/(x.size - 1)


def get_k(x, y):
    return cov(np.log(y), x) / np.var(x)


def get_C(x, y):
    return np.mean(np.log(y)) - get_k(x, y)*(np.mean(x) - x[0])


def apply(_x, fn, axis=0):
    if axis:
        x = _x.T
    else:
        x = _x

    if len(x.shape) == 1:
        return fn(x)
    else:
        output = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        output[i] = fn(x[i])

    return output.T if axis else output


def get_estimators(x, y, clearance=3):
    return [
        y.max(axis=1),
        apply(y, lambda z: get_AUC(x, z)),
        apply(y[:, clearance:], lambda z: get_AUC(x[clearance:], z)),
        apply(y[:, clearance:], lambda z: get_k(x[clearance:], z)),
        apply(y[:, clearance:], lambda z: get_C(x[clearance:], z))
    ]


def get_stats(x):
    return [
        min(x), np.median(x), max(x), np.mean(x)
    ]


def bias_correct(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    var_y = np.var(y)
    cxy = cov(x, y)
    return xbar/ybar - (1/49)*(xbar*var_y/ybar**3 - cxy/ybar**2)


def lognormal_mom(x, y):
    u = np.log(x)
    v = np.log(y)
    return np.exp(np.mean(u) - np.mean(v) + np.var(u)/2 - np.var(v)/2)


def mean_logratio(x, y):
    return np.mean(np.log(x / y))


def median_ci(data, alpha):
    r = np.sort(data)
    k = np.floor(len(r)**2 - stats.norm.ppf(1-alpha)*len(r)*alpha*(1 - alpha))


def logmean_ci(data, alpha=0.1):
    d = np.mean(data)
    s = np.std(data)
    return [
        d - stats.t.ppf(1-alpha/2, len(data)-1)*s / np.sqrt(len(data)),
        d + stats.t.ppf(1-alpha/2, len(data)-1)*s / np.sqrt(len(data))
    ]


def get_ci_mean_ratio(x, y, alpha=0.1):
    xbar = np.mean(x)
    ybar = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cxy = cov(x, y)

    est = xbar / ybar
    temp = (
        (stats.norm.ppf(1 - alpha/2) / np.sqrt(x.shape[0]))
        * np.sqrt(vx/ybar**2 + (xbar**2)*vy/ybar**4 - 2*xbar*cxy/ybar**3)
    )

    return [
        est - temp,
        est + temp
    ]


def get_ci_bias_corrected(x, y, alpha=0.1):
    xbar = np.mean(x)
    ybar = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cxy = cov(x, y)

    est = xbar/ybar - (xbar*vy/ybar**3 - cxy/ybar**2)/x.shape[0]
    temp = (
        stats.norm.ppf(1 - alpha/2)/np.sqrt(x.shape[0])
        * np.sqrt(vx/ybar**2 + (xbar**2)*vy/ybar**4 - 2*xbar*cxy/ybar**3)
    )

    return [
        est - temp,
        est + temp
    ]


def get_ci_lognormal(x, y, alpha=0.1):
    u = np.log(x)
    v = np.log(y)

    ubar = np.mean(u)
    vbar = np.mean(v)
    vu = np.var(u)
    vv = np.var(v)
    cuv = cov(u, v)

    est = ubar - vbar + vu/2 - vv/2
    bound = stats.norm.ppf(1 - alpha/2) * np.sqrt(vu + vv - 2*cuv + vu**2/2 + vv**2/2)/np.sqrt(x.shape[0])

    return [
        np.exp(est - bound),
        np.exp(est + bound)
    ]


def est_bias_corrected(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cxy = cov(x, y)

    return xbar/ybar - (xbar*vy/ybar**3 - cxy/ybar**2)/x.shape[0]


def est_lognormal(x, y):
    u = np.log(x)
    v = np.log(y)

    ubar = np.mean(u)
    vbar = np.mean(v)
    vu = np.var(u)
    vv = np.var(v)

    return np.exp(ubar - vbar + vu/2 - vv/2)


def bootci_method1(data):
    d = np.sort(data)
    return [
        d[249],
        d[9749]
    ]


def bootci_method2(orig, data):
    d = np.sort(data)
    return [
        2*orig - d[9749],
        2*orig - d[249]
    ]


def get_bootstrap(data, statistic, n_samples=10000):

    t0 = statistic(data)
    sample_size = data.shape[0]
    boot_sample = np.zeros((n_samples, t0.shape[0]))
    for i in range(n_samples):
        indices = np.random.randint(0, sample_size, sample_size)
        sample = np.array([data[j] for j in indices])
        boot_sample[i] = statistic(sample)

    return statistic(data), boot_sample


def get_cis(sample):

    cis = np.zeros((5, 4))

    sampleA_t = sample[1][:, :49]
    sampleB_t = sample[1][:, 49:]

    sampleA_t0 = sample[0][:49]
    sampleB_t0 = sample[0][49:]

    # median ratio
    Rmed = apply(sampleA_t / sampleB_t, np.median)
    cis[0] = np.concatenate([
        bootci_method1(Rmed),
        bootci_method2(np.median(sampleA_t0 / sampleB_t0), data=Rmed)
    ])

    # mean ratio
    Rmean = apply(sampleA_t, np.mean) / apply(sampleB_t, np.mean)
    cis[1] = np.concatenate([
        bootci_method1(Rmean),
        bootci_method2(np.mean(sampleA_t0) / np.mean(sampleB_t0), data=Rmean)
    ])

    # bias corrected
    n = sampleA_t.shape[0]
    Rmean_bc = np.zeros(n)
    for i in range(n):
        Rmean_bc[i] = est_bias_corrected(sampleA_t[i], sampleB_t[i])

    cis[2] = np.concatenate([
        bootci_method1(Rmean_bc),
        bootci_method2(est_bias_corrected(sampleA_t0, sampleB_t0), Rmean_bc)
    ])

    # lognormal
    Rlognormal = np.zeros(n)
    for i in range(n):
        Rlognormal[i] = est_lognormal(sampleA_t[i], sampleB_t[i])

    cis[3] = np.concatenate([
        bootci_method1(Rlognormal),
        bootci_method2(est_lognormal(sampleA_t0, sampleB_t0), Rlognormal)
    ])

    # mean of log ratios
    Rmeanlog = apply(np.log(sampleA_t) - np.log(sampleB_t), np.mean)
    cis[4] = np.concatenate([
        bootci_method1(Rmeanlog),
        bootci_method2(np.mean(np.log(sampleA_t0) - np.log(sampleB_t0)), Rmeanlog)
    ])

    return cis


def get_permutation(_dataA, _dataB, statistic, nsamples=1000):
    from copy import deepcopy

    dataA = deepcopy(_dataA)
    dataB = deepcopy(_dataB)

    sample = np.zeros((nsamples, dataA.shape[0]+dataB.shape[0]))
    for i in range(nsamples):
        tempA = dataA
        tempB = dataB
        for j in range(dataA.shape[0]):
            b = np.floor(np.random.randint(low=0, high=2, size=dataA.shape[1]))
            for k in range(b.shape[0]):
                if b[k]:
                    a = tempA[j, k]
                    tempA[j, k] = tempB[j, k]
                    tempB[j, k] = a

        sample[i] = np.concatenate([statistic(tempA), statistic(tempB)])

    return sample


def get_quantiles(data):
    d = np.sort(data)
    return np.array([d[9499], d[499]])


def get_cis2(sampleA, sampleB, origA, origB):

    cis = np.zeros((5, 2))
    print(origA)

    Rmed = apply(sampleA / sampleB, np.median)
    q = get_quantiles(Rmed)
    Rmed_orig = np.median(origA / origB)
    cis[0] = np.array([
        Rmed_orig / q[0],
        Rmed_orig / q[1]
    ])

    Rmean = apply(sampleA, np.mean) / apply(sampleB, np.mean)
    Rmean_orig = np.mean(origA) / np.mean(origB)
    q = get_quantiles(Rmean)
    cis[1] = np.array([
        Rmean_orig / q[0],
        Rmean_orig / q[1]
    ])

    n = sampleA.shape[0]
    Rmean_bc = np.zeros(n)
    for i in range(n):
        Rmean_bc[i] = est_bias_corrected(sampleA[i], sampleB[i])
    Rmean_bc_orig = est_bias_corrected(origA, origB)
    q = get_quantiles(Rmean_bc)
    cis[2] = np.array([
        Rmean_bc_orig / q[0],
        Rmean_bc_orig / q[1]
    ])

    Rlognormal = np.zeros(n)
    for i in range(n):
        Rlognormal[i] = est_lognormal(sampleA[i], sampleB[i])
    Rlognormal_orig = est_lognormal(origA, origB)
    q = get_quantiles(Rlognormal)
    cis[3] = np.array([
        Rlognormal_orig / q[0],
        Rlognormal_orig / q[1]
    ])

    Rmean_log = np.exp(apply(np.log(sampleA) - np.log(sampleB), np.mean))
    Rmean_log_orig = np.exp(np.mean(np.log(origA) - np.log(origB)))
    q = get_quantiles(Rmean_log)
    cis[4] = np.array([
        np.log(Rmean_log_orig / q[0]),
        np.log(Rmean_log_orig / q[1])
    ])

    return cis

# Generate Normal Data
def generateNormal(meansA, covAB, r1, r2, n):
    rmat = np.random.multivariate_normal(np.zeros(16), covAB, n)

    for k in range(8):
        rmat[:, k] = rmat[:, k] + meansA[k]

    for k in range(9, 12):
        rmat[:, k] = rmat[:, k] + meansA[k - 8] + np.log(r1)
    for k in range(13, 16):
        rmat[:, k] = rmat[:, k] + meansA[4] - r2 * (meansA[4] - meansA[k - 8]) + np.log(r1)

    rmat = np.exp(rmat)
    return rmat

# Generate Skewed Data
def generateSkewed(meansA, covAB, r1, r2, n):
    rmat = np.random.multivariate_normal(np.zeros(16), np.identity(16), size=n)

    rmat = rmat**2
    rmat = 0.5 * rmat - 0.5
    rmat = rmat @ la.sqrtm(covAB)

    for k in range(8):
        rmat[:, k] = rmat[:, k] + meansA[k]
    for k in range(9, 12):
        rmat[:, k] = rmat[:, k] + meansA[k - 8] + np.log(r1)
    for k in range(13, 16):
        rmat[:, k] = rmat[:, k] + meansA[4] - r2 * (meansA[4] - meansA[k - 8]) + np.log(r1)

    rmat = np.exp(rmat)
    return rmat

"""
if __name__ == '__main__':

tretA = pd.read_csv('http://www.mines.edu/~wnavidi/math482/tretinoinA',
                    delim_whitespace=True)
tretB = pd.read_csv('http://inside.mines.edu/~wnavidi/math482/tretinoinB',
                    delim_whitespace=True)
dataA = tretA.as_matrix()
dataB = tretB.as_matrix()
dataA[dataA == 0] = 3.0
dataB[dataB == 0] = 3.0
times = np.array([.25, .5, 1.0, 1.5, 4.5, 7.5, 10.5, 13.5])

data = np.concatenate([dataA[:, 3:], dataB[:, 3:]], axis=1)

n = 10000
perm_cmax = get_permutation(_dataA=dataA,
                            _dataB=dataB,
                            statistic=lambda x: apply(x, np.max),
                            nsamples=n)

cis_cmax = get_cis2(sampleA=perm_cmax[:, :49],
                    sampleB=perm_cmax[:, 49:],
                    origA=apply(dataA, np.max),
                    origB=apply(dataB, np.max))
print(cis_cmax)
"""
