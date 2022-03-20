import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# get covariance matrix of cross spectra
cov = np.load('cov_matrix.npy')
cov_inv = np.linalg.inv(cov)
# calculate chi2 with the covmat and the noise_cl with chi2 = (noise_cl - cl_true)^T * covinv * (noise_cl - cl_true)
# For each chi2, calculate the PTE
pte_list = []
chi2_list = []
for i, cl in enumerate(cls_binned):
    chi2 = np.dot(np.dot(cl - cl_mean, cov_inv), cl - cl_mean)
    chi2_list.append(chi2)
    # Calculate the PTE
    pte = 1 - stats.chi2.cdf(chi2, len(cl_mean))
    pte_list.append(pte)

# do the Kolmogorov-Smirnov test
D, p = stats.kstest(chi2_list, 'chi2', args=(len(cl_mean),))
print(D, p)

