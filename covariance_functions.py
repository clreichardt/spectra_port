
import os

import glob
import sys

#sys.path.insert(1,'/home/creichardt/.local/lib/python3.7')
import numpy as np
import healpy as hp

#from spt3g import core,maps, calibration

import pickle as pkl

import matplotlib.pyplot as plt
import time
import scipy
from spectra_port import unbiased_multispec, utils

from spt3g import core,maps, calibration

import pickle as pkl

import matplotlib.pyplot as plt
import time
import scipy

def fit_poisson(cov, factors=[1.],imin_fit = 110, imax_fit = 160, imin_out = 15,dl=50):
    '''
    cv is assumed to be nc x bc cov matrix for the first frequency (90)
    factors are poisson temperature scalings, 
    '''
    nb = cov.shape[0]
    x = dl/2 + dl*np.arange(0,nb)
    xx = (x*x).reshape((1,nb))
    xxout = xx
    xxout[0,:imin_out]=0
    cutxx = xx[imin_fit:imax_fit]
    cut_cov = cov[imin_fit:imax_fit,imin_fit:imax_fit]
    avg=0.0
    n=0
    for i in range(imin_fit,imax_fit):
        for j in range(i+1,imax_fit):
           avg += cov[i,j]/xx[i]/xx[j]
           n   += 1 
    avg /= n
    template=np.matmul(xxout.T,xxout)
    for i in range(nb):
        template[i,i] = 0    
    template *= avg
    

    nf = factors.shape[0]
    ncombo = nf * (nf+1)/2  
    #how to scale template to each band (ie 90x150)
    scaling = np.zeros(ncomb)
    k=0
    for i in range(nf):
        for j in range(i,nf):
            scaling[k] = factors[i]/factors[0] * factors[j]/factors[0]
            k+= 1
     
    poisson = np.zeros([ncombo,nb,ncombo,nb])
    for i in range(combo):
        for j in range(i,combo):
            poisson[i,:,j,:] = template * scaling[i] * scaling[j]
            (factors[i]/factors[0]) * (factors[j]/factors[0])
            if i != j:
                poisson[j,:,i,:] = poisson[i,:,j,:]
    return poisson

def corr_matrix(cov):
    c = np.diag(cov)**0.5
    c[c < 1e-12*np.max(c)] = 1.0 # avoid divide by zero
    cc = np.matmul(c.T,c)
    corr = cov/cc
    return corr

def single_block_offdiagonal(cov):
    
def fit_mll_offdiagonal(sample_cov,meas_cov):
    
    
    

def bin_spectra(dl,banddef):
    #assume dl starts at 0 and has every ell:
    nb = banddef.shape[0]-1
    odl = np.zeros(nb)
    for i in range(nb):
        odl[i] = np.mean(dl[banddef[i]:banddef[i+1]])
    return odl
        

if __name__ == '__main__':
    dlfile='/big_scratch/cr/xspec_2022/spectrum_small.pkl'
    with open(dlfile,'rb') as fp:
        spec  = pkl.load(fp)
        
    ellcov = utils.band_centers(spec['banddef'])
    
    beam_arr = np.loadtxt('/home/creichardt/spt3g_software/beams/products/compiled_2020_beams.txt')
    beams_interp = utils.fill_in_beams(beam_arr,ellcovn)
    beams = utils.explode_beams(beams_interp)

    cmbfile = '/home/creichardt/cmb_models/plik_plus_r0p01_highell_lensedtotCls_l25000.txt'
    dls = np.loadtxt(cmbfile)
    ells = dls[0,:]
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
        theory_dls[i,:] = bin_spectra(cmb_dls + fgtheory_dls[i,:],spec['banddef'])
        
    #We need offidagonal structure for Poisson
    poisson_offdiagonals = fit_poisson(spec['sample_cov'][0,:,0,:],factors=[2.86,1.06,0.61])

    #We also want off-diagonal structure due to Mll
    offdiagonal_single_block = fit_mll_offdiagonal(spec['sample_cov'],spec['meas_cov'])

    #We need diagonals, there will be 21 of these for 3 freqs
    #this is supposed to be 2S**2
    diagonals_signal = fit_signal_diagonals(spec['sample_cov'],theory_dls)

    #THis is supposed to be 4SN + 2N**2
    diagonals_noise = fit_signal_diagonals(spec['meas_cov'],theory_dls,diagonals_signal)

    #blow this back up to a 4Dim Array
    simple_cov = construct_cov(diagonals_signal,diagonals_noise,offdiagonal_single_block)
    #and combine with poisson terms
    cov = simple_cov + poisson_offdiagonals
    #Cov should be my final cov estimate. 
    

