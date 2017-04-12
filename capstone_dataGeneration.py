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

meansA = apply(np.log(Tdata), np.mean, 0)

varsA = apply(np.log(Tdata), np.var, 0)
varsB = apply(np.log(Tdata2), np.var, 0)

varsAB = np.concatenate([varsA, varsB])

p = 0.4
r1 = 1
r2 = 1
num = 50

covAB = np.zeros((16, 16))

for i in range(16):
    covAB[i, i] = varsAB[i]
    for j in range((i + 1), 16):
        if j > 16:
            break
        covAB[i, j] = p * np.sqrt(varsAB[i]) * np.sqrt(varsAB[j])
        covAB[j, i] = covAB[i, j]

normData1 = generateNormal(meansA, covAB, r1, r2, num)
normData2 = normData1[:, 8:16]
normData1 = normData1[:, 0:8]

skewData1 = generateSkewed(meansA, covAB, r1, r2, num)
skewData2 = skewData1[:, 8:16]
skewData1 = skewData1[:, 0:8]

#Change as needed
dataA_mat = normData1
dataB_mat = normData2

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
    if i == 0:
        mydata[0, 0:2] = (median_ci(estA[:, i] / estB[:, i], alpha))
    else:
        mydata[i * 5, 0:2] = (median_ci(estA[:, i] / estB[:, i], alpha))

# ratio of means
for i in range(5):
    if i == 0:
        mydata[1, 0:2] = (get_ci_mean_ratio(estA[:, i], estB[:, i]))
    else:
        mydata[i * 5 + 1, 0:2] = (get_ci_mean_ratio(estA[:, i], estB[:, i]))

# bias corrected ratio of means
for i in range(5):
    if i == 0:
        mydata[2, 0:2] = (get_ci_bias_corrected(estA[:, i], estB[:, i]))
    else:
        mydata[i * 5 + 2, 0:2] = (get_ci_bias_corrected(estA[:, i], estB[:, i]))

# lognormal ratio of means
for i in range(5):
    if i == 0:
        mydata[3, 0:2] = (get_ci_lognormal(estA[:, i], estB[:, i]))
    else:
        mydata[i * 5 + 3, 0:2] = (get_ci_lognormal(estA[:, i], estB[:, i]))

# mean of log ratios
for i in range(len(estA[0, :])):
    if i == 0:
        mydata[4, 0:2] = np.exp(logmean_ci(np.log(estA[:, i]) - np.log(estB[:, i]), alpha))
    else:
        mydata[i * 5 + 4, 0:2] = np.exp(logmean_ci(np.log(estA[:, i]) - np.log(estB[:, i]), alpha))

###############################################

n = 10000
'''

bootA_cmax = get_bootstrap(normData1, lambda x: apply(x, np.max), n)
bootB_cmax = get_bootstrap(normData2, lambda x: apply(x, np.max), n)

# median ratio
R_med = apply(bootA_cmax[1] / bootB_cmax[1], np.median, 0)
mydata[0, 2:4] = bootci_method1(R_med)
mydata[0, 4:6] = bootci_method2(np.median(bootA_cmax[0] / bootB_cmax[0]), R_med)

# mean ratio
R_mean = apply(bootA_cmax[1], np.mean, 0) / apply(bootB_cmax[1], np.mean, 0)
mydata[1, 2:4] = bootci_method1(R_mean)
mydata[1, 4:6] = bootci_method2(np.mean(bootA_cmax[0]) / np.mean(bootB_cmax[0]), R_mean)

# bias corrected
R_mean_bc = np.zeros(n)
for i in range(n):
    R_mean_bc[i] = est_bias_corrected(bootA_cmax[1][i], bootB_cmax[1][i])
mydata[2, 2:4] = bootci_method1(R_mean_bc)
mydata[2, 4:6] = bootci_method2(est_bias_corrected(bootA_cmax[0], bootB_cmax[0]), R_mean_bc)

# lognormal
R_lognormal = np.zeros(n)
for i in range(n):
    R_lognormal[i] = est_lognormal(bootA_cmax[1][i], bootB_cmax[1][i])
mydata[3, 2:4] = bootci_method1(R_lognormal)
mydata[3, 4:6] = bootci_method2(est_lognormal(bootA_cmax[0], bootB_cmax[0]), R_lognormal)

# mean of log ratios
R_mean_log = apply(np.log(bootA_cmax[1]) - np.log(bootB_cmax[1]), np.mean, 0)
mydata[4, 2:4] = np.exp(bootci_method1(R_mean_log))
mydata[4, 4:6] = np.exp(bootci_method2(np.mean(np.log(bootA_cmax[0]) - np.log(bootB_cmax[0])), R_mean_log))

#################
## AUC
dataA_auc = apply(normData1, lambda x: get_AUC(times, x), 0)
dataB_auc = get_AUC(times, normData2)



bootA_auc = get_bootstrap(normData1, lambda x: get_AUC(times, x), n)
bootB_auc = get_bootstrap(normData2, lambda x: get_AUC(times, x), n)

# median ratio
R_med = apply(bootA_auc[1] / bootB_auc[1], np.median, 0)
mydata[5, 2:4] = bootci_method1(R_med)
mydata[5, 4:6] = bootci_method2(np.median(bootA_auc[0] / bootB_auc[0]), R_med)

# ratio of means
R_mean = apply(bootA_auc[1], np.mean, 0) / apply(bootB_auc[1], np.mean, 0)
mydata[6, 2:4] = bootci_method1(R_mean)
mydata[6, 4:6] = bootci_method2(np.mean(bootA_auc[0]) / np.mean(bootB_auc[0]), R_mean)

# bias corrected
R_mean_bc = np.zeros(n)
for i in range(n):
    R_mean_bc[i] = est_bias_corrected(bootA_auc[1][i], bootB_auc[1][i])
mydata[7, 2:4] = bootci_method1(R_mean_bc)
mydata[7, 4:6] = bootci_method2(est_bias_corrected(bootA_auc[0], bootB_auc[0]), R_mean_bc)

# lognormal
R_lognormal = np.zeros(n)
for i in range(n):
    R_lognormal[i] = est_lognormal(bootA_auc[1][i], bootB_auc[1][i])
mydata[8, 2:4] = bootci_method1(R_lognormal)
mydata[8, 4:6] = bootci_method2(est_lognormal(bootA_auc[0], bootB_auc[0]), R_lognormal)

# mean of log ratios
R_mean_log = apply(np.log(bootA_auc[1]) - np.log(bootB_auc[1]), np.mean, 0)
mydata[9, 2:4] = np.exp(bootci_method1(R_mean_log))
mydata[9, 4:6] = np.exp(bootci_method2(np.mean(np.log(bootA_auc[0]) - np.log(bootB_auc[0])), R_mean_log))

###############

# AUC*

normData1 = np.concatenate((normData1[:, 3:8], normData2[:, 3:8]), 1)
boot_auc_starA = get_bootstrap(normData1[:, :5].T, lambda x: get_AUC(times[3:8], x), n)
boot_auc_starB = get_bootstrap(normData1[:, 5:].T, lambda x: get_AUC(times[3:8], x), n)

boot_auc_star = [np.concatenate([boot_auc_starA[0], boot_auc_starB[0]]),
                 np.concatenate([boot_auc_starA[1], boot_auc_starB[1]], axis=1)]

mydata[10:15, 2:6] = get_cis(boot_auc_star)
mydata[14, 2:6] = np.exp(mydata[14, 2:6])
'''
#################

#e^k
'''
normData1 = np.concatenate((normData1[:, 3:8], normData2[:, 3:8]), 1)

fn = lambda x: get_k(times[3:8], x)
boot_kA = get_bootstrap(normData1[:, :5].T, lambda x: apply(x, fn, axis=1), n)
boot_kB = get_bootstrap(normData1[:, 5:].T, lambda x: apply(x, fn, axis=1), n)

boot_k = [np.concatenate([boot_kA[0], boot_kB[0]]),
          np.concatenate([boot_kA[1], boot_kB[1]], axis=1)]

mydata[15:20, 2:6] = get_cis(boot_k)
mydata[19, 2:6] = np.exp(mydata[19, 2:6])

###############
'''
# C
normData1 = np.concatenate((normData1[:, 3:8], normData2[:, 3:8]), 1)

fn2 = lambda x: get_C(times[3:8], x)

boot_CA = get_bootstrap(normData1[:, :5].T, lambda x: apply(x, fn2, axis=1), n)
boot_CB = get_bootstrap(normData1[:, 5:].T, lambda x: apply(x, fn2, axis=1), n)

boot_C = [np.concatenate([boot_CA[0], boot_CB[0]]),
          np.concatenate([boot_CA[1], boot_CB[1]], axis=1)]

mydata[20:25, 2:6] = get_cis(boot_C)
mydata[24, 2:6] = np.exp(mydata[24, 2:6])

###################################################################################
'''
data_mat = np.concatenate((dataA_mat, dataB_mat), 1)

n = 1000

# cmax
perm_cmax = get_permutation(dataA_mat, dataB_mat, lambda x: apply(x, np.max, 0), n)

cis_cmax = get_cis2(perm_cmax[:, 0:50], perm_cmax[:, 50:100], apply(dataA_mat, np.max, 0), apply(dataB_mat, np.max, 0))
mydata[0:5, 6:8] = cis_cmax
mydata[4, 6:8] = np.exp(mydata[4, 6:8])

#######################

# AUC
perm_auc = get_permutation(dataA_mat, dataB_mat, lambda x: apply(x, lambda y: get_AUC(times, y), 0), n)

cis_auc = get_cis2(perm_auc[:, 0:50], perm_auc[:, 50:100], apply(dataA_mat, lambda x: get_AUC(times, x), 0), apply(dataB_mat, lambda x: get_AUC(times, x), 0))
mydata[5:10, 6:8] = cis_auc
mydata[9, 6:8] = np.exp(mydata[9, 6:8])

#############################

# AUC*
perm_auc_ = get_permutation(dataA_mat[:, 3:8], dataB_mat[:, 3:8], lambda x: apply(x, lambda y: get_AUC(times[3:8], y), 0), n)

cis_auc_ = get_cis2(perm_auc_[:, 0:50], perm_auc_[:, 50:100], apply(dataA_mat[:, 3:8], lambda x: get_AUC(times[3:8], x), 0, apply(dataB_mat[:, 3:8], lambda x: get_AUC(times[3:8], x), 0)))
mydata[10:15, 6:8] = cis_auc_
mydata[14, 6:8] = np.exp(mydata[14, 6:8])

####################

# e^k

perm_k = get_permutation(dataA_mat[:, 3:8], dataB_mat[:, 3:8], lambda x: np.exp(-apply(x, lambda y: get_k(times[3:8], y), 0)), n)

cis_k = get_cis2(perm_k[:, 0:50], perm_k[:, 50:100], np.exp(-apply(dataA_mat[:, 3:8], lambda x: get_k(times[3:8], x), 0)), np.exp(-apply(dataB_mat[:, 3:8], lambda x: get_k(times[3:8], x), 0)))
mydata[15:20, 6:8] = cis_k
mydata[19, 6:8] = np.exp(mydata[19, 6:8])

#####################

# C(1.5)
perm_C = get_permutation(dataA_mat[:, 3:8], dataB_mat[:, 3:8], lambda x: np.exp(apply(x, lambda y: get_C(times[3:8], y), 0)), n)

cis_C = get_cis2(perm_C[:, 0:50], perm_C[:, 50:100], np.exp(apply(dataA_mat[:, 3:8], lambda x: get_C(times[3:8], x), 0)), np.exp(apply(dataB_mat[:, 3:8], lambda x: get_C(times[3:8], x), 0)))
mydata[20:25, 6:8] = cis_C
mydata[24, 6:8] = np.exp(mydata[24, 6:8])

with open('Intervals.csv', 'a') as fout:
    np.savetxt(fout, mydata, delimiter=",")

'''

















