
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

def rebin_err(vector, mult):
    '''Rebin vector group (mult) values together 
    ie a 500-vector and mult=10 -> outputs a 50-length vector
    
    vector is assumed to be power spectrum errors
    Assume starting at 0-dl; dl-2dl, ...
    Assumes DOF proportional to ell.
    
    These are sample variance errors
    '''
    nl = vector.shape[0]
    lvec =np.arange(0,nl)+0.5
    
    cov1 = vector**2 * lvec
    sum_cov1 = lvec * cov1
    
    nout = nl // mult
    ndof = 0.
    out  = 0.
    for i in range(mult):
        out += sum_cov1[i:250:mult]
        ndof += lvec[i:250:mult]
    
    out /= (ndof**2)
    return np.sqrt(out)


def rebin_spec(vector, mult):
    '''Rebin vector group (mult) values together 
    ie a 500-vector and mult=10 -> outputs a 50-length vector
    
    vector is assumed to be power spectrum errors
    Assume starting at 0-dl; dl-2dl, ...
    Assumes DOF proportional to ell.
    
    These are spectra
    '''
    nl = vector.shape[0]
    lvec =np.arange(0,nl)+0.5
    
    modes_times_spectra = vector * lvec
    
    
    nout = nl // mult
    ndof = 0.
    out  = 0.
    for i in range(mult):
        out += modes_times_spectra[i:250:mult]
        ndof += lvec[i:250:mult]
    
    out /= (ndof)
    return out
    


if __name__ == '__main__':

    dir = '/big_scratch/cr/xspec_2022/data_v5/'
    nulls = [dir + 'null_spectrum_90.pkl', dir + 'null_spectrum_150.pkl', dir + 'null_spectrum_220.pkl']
    binned = [ dir + 'spectrum90_nullbins.pkl', dir + 'spectrum150_nullbins.pkl', dir + 'spectrum220_nullbins.pkl']


    dlfile='/big_scratch/cr/xspec_2022/spectrum_small.pkl'
    with open(dlfile,'rb') as fp:
        spec  = pkl.load(fp)

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
    
    print('doing theory')

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
        theory_dls[i,:] = covariance_functions.bin_spectra(cmb_dls + fgtheory_dls[i,:],endend['banddef'])
        
    
    theory90_dl50  = theory_dls[0,:]
    theory150_dl50 = theory_dls[3,:]
    theory220_dl50 = theory_dls[5,:]
    print('doing rebin theory')
    theory90_dl500  = rebin_spec(theory90_dl50,  dlnull//dlspec)[imin_dl500:imax_dl500] 
    theory150_dl500 = rebin_spec(theory150_dl50, dlnull//dlspec)[imin_dl500:imax_dl500] 
    theory220_dl500 = rebin_spec(theory220_dl50, dlnull//dlspec)[imin_dl500:imax_dl500] 

    
    print('expect 2d: ',covcov.sample_cov.shape)
    covcov.sample_cov=np.reshape(covcov.sample_cov,[6*259,6*259])
    sv = np.diag(covcov.sample_cov)
    sv90_dl50  = np.sqrt(sv[:nlc])
    sv150_dl50 = np.sqrt(sv[3*nlc:4*nlc])
    sv220_dl50 = np.sqrt(sv[5*nlc:6*nlc])

    allowed_SV = 0.04
    print('Allowed SV is ',allowed_SV)

    sv90_dl500  = allowed_SV*rebin_err(sv90_dl50,dlnull//dlspec)[imin_dl500:imax_dl500]
    sv150_dl500 = allowed_SV*rebin_err(sv150_dl50,dlnull//dlspec)[imin_dl500:imax_dl500]
    sv220_dl500 = allowed_SV*rebin_err(sv220_dl50,dlnull//dlspec)[imin_dl500:imax_dl500]
    
    calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    calibration_factors *= 1e3 

    print('loading nulls')

    with open(nulls[0],'rb') as fp:
        n90 = pkl.load(fp)
    null90 = n90.spectrum[imin_dl500:imax_dl500,0] * calibration_factors[0]**2
    err90 = (np.sqrt(np.diag(n90.est1_cov))* calibration_factors[0]**2)[imin_dl500:imax_dl500]
    
    with open(nulls[1],'rb') as fp:
        n150 = pkl.load(fp)
    null150 = n150.spectrum[imin_dl500:imax_dl500,0]* calibration_factors[1]**2
    err150 = (np.sqrt(np.diag(n150.est1_cov))* calibration_factors[1]**2)[imin_dl500:imax_dl500]
    
    with open(nulls[2],'rb') as fp:
        n220 = pkl.load(fp)
    null220 = n220.spectrum[imin_dl500:imax_dl500,0]* calibration_factors[2]**2
    err220 = (np.sqrt(np.diag(n220.est1_cov))* calibration_factors[2]**2)[imin_dl500:imax_dl500]

    print('loading binned for Tf')
    with open(binned[0],'rb') as fp:
        tmp = pkl.load(fp)
    s90 = tmp.spectrum[imin_dl500:imax_dl500,0] * calibration_factors[0]**2
    with open(binned[1],'rb') as fp:
        tmp = pkl.load(fp)
    s150 = tmp.spectrum[imin_dl500:imax_dl500,0] * calibration_factors[1]**2
    with open(binned[2],'rb') as fp:
        tmp = pkl.load(fp)
    s220 = tmp.spectrum[imin_dl500:imax_dl500,0] * calibration_factors[2]**2
    
    
    tf_90 = theory90_dl500 / s90
    tf_150 = theory150_dl500 / s150
    tf_220 = theory220_dl500 / s220
    #pdb.set_trace()
    cnull90  = tf_90  * null90
    cnull150 = tf_150 * null150
    cnull220 = tf_220 * null220

    cerr90  = tf_90  * err90
    cerr150 = tf_150 * err150
    cerr220 = tf_220 * err220

    comb_err90  = (cerr90  + sv90_dl500 )
    comb_err150 = (cerr150 + sv150_dl500)
    comb_err220 = (cerr220 + sv220_dl500)
    
    print('90s:',cnull90/comb_err90)
    print('Chisq 90:',np.sum((cnull90/comb_err90)**2), ' dof: ', imax_dl500-imin_dl500)
    
    print('150s:',cnull150/comb_err150)
    print('Chisq 150:',np.sum((cnull150/comb_err150)**2), ' dof: ', imax_dl500-imin_dl500)
    
    print('220s:',cnull220/comb_err220)
    print('Chisq 220:',np.sum((cnull220/comb_err220)**2), ' dof: ', imax_dl500-imin_dl500)
    pdb.set_trace()
    

    
