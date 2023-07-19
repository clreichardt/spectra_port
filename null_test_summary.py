
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




if __name__ == '__main__':

    dir = '/big_scratch/cr/xspec_2022/data_v5/'
    nulls = [dir + 'null_spectrum_90.pkl', dir + 'null_spectrum_150.pkl', dir + 'null_spectrum_220.pkl']
    binned = [ dir + 'spectrum90_nullbins.pkl', dir + 'spectrum150_nullbins.pkl', dir + 'spectrum220_nullbins.pkl']


    endfile = '/big_scratch/cr/xspec_2022/spectrum_small.pkl'
    covfile = '/big_scratch/cr/xspec_2022/covariance.pkl'
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

    with open(endfile,'rb') as fp:
        endend = pkl.load(fp)
        
    with open(covfile,'rb') as fp:
        covcov = pkl.load(fp)
        
    ellcov = utils.band_centers(endend['banddef'])


    cmbfile = '/home/creichardt/cmb_models/plik_plus_r0p01_highell_lensedtotCls_l25000.txt'
    dls = np.loadtxt(cmbfile)
    ells = dls[:,0]
    cmb_dls = dls[:,1]
    #print('first 4 ells:',ells[:5])
    #   cmb_dls_interp = utils.fill_in_theory(cmbfile,ellkern)
    norgfgtheoryfiles = ['/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_90x90.txt',
               '/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_90x150.txt',
               '/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_90x220.txt',
               '/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_150x150.txt',
               '/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_150x220.txt',
               '/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_220x220.txt']
    norgfgtheory_dls = utils.fill_in_theory(norgfgtheoryfiles,ells)

    pois = 1.2* (ells/3000.)**2

    rg_dls_interp = np.zeros([6,ells.shape[0]])
    facs  = [2.86, 1.06, 0.61]
    rg_dls_interp[0,:] = pois * facs[0]* facs[0]
    rg_dls_interp[1,:] = pois * facs[0]* facs[1]
    rg_dls_interp[2,:] = pois * facs[0]* facs[2]
    rg_dls_interp[3,:] = pois * facs[1]* facs[1]
    rg_dls_interp[4,:] = pois * facs[1]* facs[2]
    rg_dls_interp[5,:] = pois * facs[2]* facs[2]
    fgtheory_dls  = rg_dls_interp + norgfgtheory_dls

    nlc = ellcov.shape[0]
    theory_dls = np.zeros([6,nlc])
    for i in range(6):
        theory_dls[i,:] = covariance_functions.bin_spectra(cmb_dls + fgtheory_dls[i,:],spec['banddef'])
        
    
    theory90_dl50  = theory_dls[0,:]
    theory150_dl50 = theory_dls[3,:]
    theory220_dl50 = theory_dls[5,:]
    print('expect 2d: ',covcov.sample_covariance.shape)
    sv = np.diag(covcov.sample_covariance)
    sv90_dl50  = np.sqrt(sv[:nlc])
    sv150_dl50 = np.sqrt(sv[3*nlc:4*nlc])
    sv220_dl50 = np.sqrt(sv[5*nlc:6*nlc])

    allowed_SV = 0.01
    
    sv90_dl500  = allowed_SV*rebin_err(sv90_dl50,dlnull//dlspec)
    sv150_dl500 = allowed_SV*rebin_err(sv150_dl50,dlnull//dlspec)
    sv220_dl500 = allowed_SV*rebin_err(sv220_dl50,dlnull//dlspec)


    