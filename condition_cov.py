
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

def construct_cov(nspec,nb,diagonals,offdiagonal_single_block):
    cov = np.zeros([nspec,nb,nspec,nb])
    k=0
    for i in range(nspec):
        for j in range(i,nspec):
            sqrtdiag = np.sqrt(diagonals[k,:])

            sqrtdiag2d = np.tile(sqrtdiag,[nb,1])
            cc = sqrtdiag2d* sqrtdiag2d.T * offdiagonal_single_block
            cov[i,:,j,:] = cc
            if i != j:
                cov[j,:,i,:] = cov[i,:,j,:].T
            k += 1

    return cov


if __name__ == '__main__':

    nf = 3
    nspec = nf*(nf+1)//2  # 6 for expected numbers
    nblock = nspec * (nspec+1) //2 #should be 21 for current #s

    print("initiating files for bl v3beta6")
    print("Also using field-specific PWF")
    dlfile='/big_scratch/cr/xspec_2022/spectrum_blv3v6small.pkl'
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

    rg_dls_interp = np.zeros([nspec,ells.shape[0]])
    facs  = [2.86, 1.06, 0.61]
    rg_dls_interp[0,:] = pois * facs[0]* facs[0]
    rg_dls_interp[1,:] = pois * facs[0]* facs[1]
    rg_dls_interp[2,:] = pois * facs[0]* facs[2]
    rg_dls_interp[3,:] = pois * facs[1]* facs[1]
    rg_dls_interp[4,:] = pois * facs[1]* facs[2]
    rg_dls_interp[5,:] = pois * facs[2]* facs[2]
    fgtheory_dls  = rg_dls_interp + norgfgtheory_dls

    nlc = ellcov.shape[0]
    theory_dls = np.zeros([nspec,nlc])
    theorynorg_dls = np.zeros([nspec,nlc])
    for i in range(nspec):
        theory_dls[i,:] = covariance_functions.bin_spectra(cmb_dls + fgtheory_dls[i,:],spec['banddef'])
        theorynorg_dls[i,:] = covariance_functions.bin_spectra(cmb_dls + norgfgtheory_dls[i,:],spec['banddef'])
    #calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    #change to below when reran with latest PWFs/Tfs on 2023 Sep 08
    #calibration_factors = np.asarray([ (0.9017)**-0.5, (0.9833)**-0.5, (0.9703)**-0.5 ])
    
    calibration_factors = np.asarray([ (0.8888)**-0.5, (0.9797)**-0.5, (0.9755)**-0.5 ])
    calibration_factors *= 1e-3  #correction for units between sims and real data. The transfer function brings it over.  This ends up being brought to the 4 power so 1e-12 effectively.
    
    # get beams
    beam_arr = np.loadtxt('/home/creichardt/spt3g_software/beams/products/compiled_2020_beams.txt')
    #real data also has PWF (sims created at 8192)
    blmax=int(beam_arr[-1,0]+0.001)
    #pwf = hp.pixwin(nside,lmax = blmax)
    #07 jun 2024: Changed to use PWF from field
    out=hp.read_cl('/home/creichardt/spt3g_software/mapspectra/data/healpix_8192_pixwin_spt3g1500d.fits')
    pwf = out[:blmax+1]
    for i in range(nf):
        beam_arr[:,i+1] *= pwf
    lbeam = beam_arr[:,0]
    beams = beam_arr[:,1:1+nf]
    nlb = lbeam.shape[0]
    print('may need to adjust ells between theory and beam files')
    
    print('adjust noise levels...')
    sigma_noise = [4.,4.,10.] #need to check these, expect in uK-arcmin
    arcmin_area = (1./60.*np.pi/180.)**2
    Nl = sigma_noise**2  * arcmin_area
    Nls = np.zeros([nf,nlb])
    dl=50.
    prefact = np.sqrt(2/((2*l+1) * dl*fsky))
    print('maybe add a subtract 300 from these dof - the 2l+1 term')
    
    for i in range(nf):
        Nls[i,:] = Nl[i] / beams[:,i]**2

    print('set up freq pair indexing')
    global_index_array = np.zeros([(nspec*(nspec+1))//2,2],dtype=np.int32)
    global_freq_index_array = np.zeros([nspec,2],dtype=np.int32)
    k=0
    for i in range(nspec):
        for j in range(i,nspec):
            global_index_array[k,0] = i
            global_index_array[k,1] = j
            k+=1 
    k=0
    for i in range(nf):
        for j in range(i,nf):
            global_freq_index_array[k,0] = i
            global_freq_index_array[k,1] = j
            k+=1  

    print('also correlated noise terms')
    corrCovDiags = np.zeros([nblock,nlb])
    for k in range(nblock):
        i=global_index_array[k,0]
        j=global_index_array[k,1]
        l = global_freq_index_array[i,0]
        m = global_freq_index_array[j,1]
        n = global_freq_index_array[k,0]
        o = global_freq_index_array[k,1]
        corramp = corr_noise_level[l]*corr_noise_level[m]*corr_noise_level[n]*corr_noise_level[o]
        corrCovDiags[k,:] = corramp * corr_template

    
    print('bin everything')
    
    print('combine it all together')
    
 



    covfile = '/big_scratch/cr/xspec_2022/covariance_blv3b6.pkl'
    with open(covfile,'rb') as fp:
        cov_obj=pkl.load( fp)
    
    print('define off-diagonal anticorrelation')
    block_template = cov_obj.offdiagonal_single_block

    print('construct cov')

    
    print('also add RG Poisson terms')
    poisson = cov_obj.poisson
    cov_template += poisson

        
    transform = np.fromfile('/home/creichardt/highells_dls/bptransform.npy',dtype=np.float64)
    
    print('Now rebin cov guess to final bins')
    
    
    
    
