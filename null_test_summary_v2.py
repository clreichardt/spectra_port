
import os

import glob
import sys

#sys.path.insert(1,'/home/creichardt/.local/lib/python3.7')
import numpy as np
#import healpy as hp

#from spt3g import core,maps, calibration

import pickle as pkl

import matplotlib.pyplot as plt
import time
import scipy
from spectra_port import unbiased_multispec, utils, covariance_functions
import pdb

'''
Modified version 
to use actual kernels
'''





    


if __name__ == '__main__':

    freqs=['90','150','220']
    dir = '/big_scratch/cr/xspec_2022/'
    null12s = [dir + 'data_v5/null_spectrum_90.pkl', dir + 'data_v5/null_spectrum_150.pkl', dir + 'data_v5/null_spectrum_220.pkl']
    nulllrs = [dir + 'data_v5_lr/spectrum90_lrnull.pkl', dir + 'data_v5_lr/spectrum150_lrnull.pkl', dir + 'data_v5_lr/spectrum220_lrnull.pkl']
    binned = [ dir + 'spectrum500_90_small.pkl', dir + 'spectrum500_150_small.pkl', dir + 'spectrum500_220_small.pkl']

    allowed_SV = 0.04
    print('Allowed SV is ',allowed_SV)

    
    calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    calibration_factors *= 1e3 

    '''
    Going to do a simplified transfer function correction here since only used in plotting. 

    Assume each ell has Nmodes propto ell. (not quite right since drop low m)

    Scale Binned nulls by Tf

    Get NV from null, scale it with quick Tf

    Will grab SV from covariance_function object. Again rebin diagonals according to the weights with ell./ 
    May do this in fractional space since Ndof should change slower than the power. Then multiply it by the nullspectrum binned values
    Multiply by 1%? maybe

    Quote Chisq's for each Null
    '''

    dlnull=500
    dlspec=50
    lmin = 2000
    lmax = 11000
    imin_dl500 = lmin // dlnull
    imax_dl500 = lmax // dlnull
    imin_dl50  = lmin // dlspec
    imax_dl50  = lmax // dlspec






    print('loading nulls')

    for i in range(3):
        with open(binned[i],'rb') as fp:
            spec= pkl.load(fp)
        with open(null12s[i],'rb') as fp:
            n12= pkl.load(fp)
        with open(nulllrs[i],'rb') as fp:
            nlr= pkl.load(fp)
        cal = calibration_factors[i]**2

        pseudo_scov = spec['mc_spectrum'].cov
        pseudo_dcov12 = n12.est1_cov * cal**2
        pseudo_dcovlr = nlr.est1_cov * cal**2

        pseudo12 = n12.spectrum * cal
        pseudolr = nlr.spectrum * cal

        #choose ranges


        #apply inverse kernel

        comb_err12 = derr12 + serr
        comb_errlr = derrlr + serr

        print(freqs[i])
        print('Chisq 12:',np.sum((cnull12/comb_err12)**2), ' dof: ', imax_dl500-imin_dl500)
        print('Chisq LR:',np.sum((cnulllr/comb_errlr)**2), ' dof: ', imax_dl500-imin_dl500)

    pdb.set_trace()
    

    
