
import os

import glob
import sys

#sys.path.insert(1,'/home/creichardt/.local/lib/python3.7')
import numpy as np
#import healpy as hp

#from spt3g import core,maps, calibration

import pickle as pkl
import pdb
import matplotlib.pyplot as plt
import time
import scipy
from spectra_port import unbiased_multispec, utils, covariance_functions

#from spt3g import core,maps, calibration


if __name__ == '__main__':

    print("initiating files")
    dlfile='/big_scratch/cr/xspec_2022/spectrum_blv3rc4_small.pkl' #input
    covfile = '/big_scratch/cr/xspec_2022/covariance_blv3rc4.pkl'  #output
    #dlfile='/big_scratch/cr/xspec_2022/spectrum_blv3rc4_1simpwf_small.pkl' #input                                                                           
    #covfile = '/big_scratch/cr/xspec_2022/covariance_blv3rc4_1simpwf.pkl'  #output      
    with open(dlfile,'rb') as fp:
        spec  = pkl.load(fp)
        
    ellcov = utils.band_centers(spec['banddef'])


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
    #calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    #change to below when reran with latest PWFs/Tfs on 2023 Sep 08
    #calibration_factors = np.asarray([ (0.9017)**-0.5, (0.9833)**-0.5, (0.9703)**-0.5 ])
    #change to latest field-speific PWF and v3bta6 beams:
    #calibration_factors = np.asarray([ (0.8888)**-0.5, (0.9797)**-0.5, (0.9755)**-0.5 ])
    #24/6/24: blv3b7 beams (and field pwf)
    #calibration_factors = np.asarray([ (0.8880)**-0.5, (0.9789)**-0.5, (0.97505)**-0.5 ])
    #24/9/20: Tilt from Aylor et al + rc4 beams:
    calibration_factors = np.asarray([ (0.88546)**-0.5, (0.97518)**-0.5, (0.95894)**-0.5 ])
    calibration_factors *= 1e-3  #correction for units between sims and real data. The transfer function brings it over.  This ends up being brought to the 4 power so 1e-12 effectively.
    
    print("initiating cov")

    cov_obj = covariance_functions.covariance(spec,theory_dls, calibration_factors)        
    nn = cov_obj.cov.shape[0]*cov_obj.cov.shape[1]
    eval,evec = np.linalg.eig(cov_obj.cov.reshape([nn,nn]))
    print('Returned <0 evals: {} of {}'.format(np.sum(eval<=0),nn))

    #exit()

    with open(covfile,'wb') as fp:
        pkl.dump(cov_obj, fp)

    pdb.set_trace()
