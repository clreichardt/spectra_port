
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

global_index_array = None

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


def get_2d_indices(i,n):
    '''Returns two indices corresponding to position
    ie 0 (90x90) -> 0,0
    '''
    if global_index_array is None or global_index_array.shape[0] != (n*(n+1))/2:
        # create array
        global_index_array = np.zeros([(n*(n+1))/2,2,dtype=np.int32])
        k=0
        for i in range(n):
            for j in range(i,n):
                global_index_array[k,0] = i
                global_index_array[k,1] = j
                k+=1 
    return global_index_array[i,0],global_index_array[i,1]
    
    
def get_1d_index(i,j,n):
    '''Returns the 1d index corresponding to block, ix j
    eg 0x0 -> 0
    0 x 3 -> 3
    Assumes 0 <= i <= j < n;  There is no error checking, except that i/j will be swapped if i > j
    '''
    if i <= j:
        return i*n - ((i-1)*i)/2 + j
    return j*n - ((j-1)*j)/2 + i


def get_theory_cov(dls,i,j):
    nspec = dls.shape[0] #check if this is right dim
    assert nspec == 6
    
    a,b = get_2d_indices(i,nspec)
    c,d = get_2d_indices(j,nspec)
    
    m = get_1d_index(a,c,nspec)
    n = get_1d_index(b,d,nspec)
    o = get_1d_index(a,d,nspec)
    p = get_1d_index(c,c,nspec)
    out = dls[m,:] * dls[n,:] + dls[o,:] * dls[p,:]
    
    return out
    
    
    

def signal_func(ells,const,amp):
    return const + amp * (ells)**(-1.3)

def fit_single_block_signal(cov,i_block,j_block, theory_dls,fit_range = [30,190], use_range = [40,250] ):
    theory = get_theory_cov(theory_dls,i_block,j_block)
    # theory is eg (150x150)(90x90)+ (90x150)**2
    
    observed = np.diag(cov[i_block,:,j_block,:])

    obs_prefactor = observed / theory
    inds = np.arange(0,observed.shape[0])+0.5
    
    fit_vals = obs_prefactor[fit_range[0]:fit_range[1]]
    fit_inds = inds[fit_range[0]:fit_range[1]]
    const_guess = fit_vals[-1]
    amp_guess = (fit_vals[0] - const_guess) * fit_inds[0]**1.3
    p0=[const_guess,amp_guess]
    sigma = 0.1 * (signal_func(fit_inds,const_guess,amp_guess))
    params = scipy.optimize.curve_fit(signal_func,fit_inds,fit_values,p0=p0,sigma=sigma)
    
    prefactor = obs_prefactor
    prefactor[use_range[0]:use_range[1]] = signal_func(inds[use_range[0]:use_range[1]],params[0],params[1])

    return prefactor    



def fit_signal_diagonals(sample_cov,theory_dls,fit_range = [30,190], use_range = [40,250] ):
    
    nspec = sample_cov.shape[0]
    nb = sample_cov.shape[1]
    ncross = nspec * (nspec+1)/2
    diagonals = np.zeros([ncross,nb])
    for i in range(nspec):
        for j in range(nspec):
            k = get_1d_index(i,j,nspec)
            diagonals[k,:] = fit_single_block_signal(sample_cov,i,j,theory_dls,fit_range=fit_range,use_range=use_range)
    return diagonals


def corr_matrix(cov):
    c = np.diag(cov)**0.5
    c[c < 1e-12*np.max(c)] = 1.0 # avoid divide by zero
    cc = np.matmul(c.T,c)
    corr = cov/cc
    return corr



def single_block_offdiagonal(cov):
    pass

def fit_mll_offdiagonal(sample_cov,meas_cov, max_offset = 1, use_noise = [4,5,6], use_sample = [5,6],bininds=[50,160]):
    summed = 0
    n = 0
    for i in use_noise:
        corr = corr_matrix(meas_cov[i,:,i,:])
        summed += corr[bininds[0]:bininds[1],bininds[0]:bininds[1]]
        n+=1
    for i in use_sample:
        corr = corr_matrix(sample_cov[i,:,i,:])
        summed += corr[bininds[0]:bininds[1],bininds[0]:bininds[1]]
        n+=1
    summed /= n
    values = np.zeros(max_offset)
    for i in range(max_offset):
        dd = np.diag(summed,i+1)
        values[i] = np.mean(dd)
    
    nb = corr.shape[0]
    block = np.identity(nb)
    for i in range(max_offset):
        for j in range(nb-i+1):
            block[j,j+i+1] = values[i]
            block[j+i+1,j] = values[i]

    return block
    

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
    

