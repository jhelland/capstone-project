import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import timeit
from capstone_functions import *

tretA = pd.read_csv('http://www.mines.edu/~wnavidi/math482/tretinoinA',
                        delim_whitespace=True)
tretB = pd.read_csv('http://inside.mines.edu/~wnavidi/math482/tretinoinB',
                    delim_whitespace=True)
data = tretA.as_matrix()
data2 = tretB.as_matrix()
data[data == 0] = 3.0
data2[data2 == 0] = 3.0
times = np.array([.25, .5, 1.0, 1.5, 4.5, 7.5, 10.5, 13.5])

data = np.concatenate([data[:, 3:], data2[:, 3:]], axis=1)

#for z in range(100):

Tdata = data
Tdata2 = data2

meansA = apply(np.log(Tdata), np.mean, 1)

varsA = apply(np.log(Tdata), np.var, 1)
varsB = apply(np.log(Tdata2), np.var, 1)

varsAB = np.concatenate([varsA, varsB])

p = 0.4
r1 = 1
r2 = 1
n = 50

covAB = np.zeros((16, 16))

for i in range(16):
    covAB[i, i] = varsAB[i]
    for j in range((i + 1), 16):
        if j > 16:
            break
        covAB[i, j] = p * np.sqrt(varsAB[i]) * np.sqrt(varsAB[j])
        covAB[j, i] = covAB[i, j]

normData1 = generateNormal(meansA, covAB, r1, r2, n)
normData2 = normData1[:, 8:16]
normData1 = normData1[:, 0:8]

skewData1 = generateSkewed(meansA, covAB, r1, r2, n)
skewData2 = skewData1[:, 8:16]
skewData1 = skewData1[:, 0:8]


#########################################################
estimatorsA = get_estimators(times, normData1)
estimatorsA = np.vstack(estimatorsA).T
estimatorsB = np.vstack(get_estimators(times, normData2)).T

estimatorsA[:, 3:5] = np.exp(estimatorsA[:, 3:5])
estimatorsB[:, 3:5] = np.exp(estimatorsB[:, 3:5])

mydata = np.zeros((25, 8))

# median ratio CI
alpha = 1 - 0.9  # 90% CI

estA = estimatorsA
estB = estimatorsB


for i in range(len(estA[0, :])):
    mydata[i * 5 - 4, 0:2] = (median_ci(estA[:, i] / estB[:, i], alpha))

# ratio of means
for i in range(5):
    mydata[i * 5 - 3, 0:2] = (get_ci_mean_ratio(estA[:, i], estB[:, i]))

# bias corrected ratio of means
for i in range(5):
    mydata[i * 5 - 2, 0:2] = (get_ci_bias_corrected(estA[:, i], estB[:, i]))

# lognormal ratio of means
for i in range(5):
    mydata[i * 5 - 1, 0:2] = (get_ci_lognormal(estA[:, i], estB[:, i]))

# mean of log ratios
for i in range(len(estA[0, :])):
    mydata[i * 5, 0:2] = np.exp(logmean_ci(np.log(estA[:, i]) - np.log(estB[:, i]), alpha))






