
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



def dtsingle(m,dt,tau):
    return np.abs(np.exp(complex(0,dt*m))/complex(1,m*tau) - np.exp(complex(0,-1*dt*m))/complex(1,-1*m*tau))

def single(m,tau):
    return np.abs(1/complex(1,m*tau) - 1/complex(1,-1*m*tau))

def single1(m,tau):
    return 1/complex(1,m*tau)
#4.015ms

def taufrac(tau,nb,dl=500):

    #tau = 0.0004
    vscan = 1.0 * np.cos(np.pi/180 * 57.5) * np.pi/180
    tau_rad = vscan * tau
    vals = np.zeros(23)
    for i in range(23):
        lmin = i*dl
        lmax = (i+1)*dl
        n=0.0
        val =0.0
        lmin = np.max([300,lmin])
        for m in range(300,lmin+1):
            n += dl 
            val += dl * single(m,tau_rad)
        for m in range(lmin+1,lmax):
            n += dl -(m - lmin)
            val += (dl -(m - lmin))* single(m,tau_rad)
        val /= n
        vals[i]=val
    return vals

def tautemplate(tau,theory,nb,dl=500):
    vscan = 1.0 * np.cos(np.pi/180 * 57.5) * np.pi/180
    tau_rad = vscan * tau
    vals = np.zeros(nb)
    for i in range(nb):
        lmin = i*dl
        lmax = (i+1)*dl
        dlbin = theory[lmin:lmax]
        n=0.0
        val =0.0
        lmin = np.max([300,lmin])
        for m in range(300,lmin+1):
            n += dl 
            val += single(m,tau_rad) * np.sum(dlbin)
        for m in range(lmin+1,lmax):
            n += dl -(m - lmin)
            val += single(m,tau_rad) * np.sum(dlbin[m-lmin:])
        val /= n
        vals[i]=val
        #pdb.set_trace()
    return vals

def dttautemplate(dt,tau,theory,nb,dl=500):
    vscan = 1.0 * np.cos(np.pi/180 * 57.5) * np.pi/180
    tau_rad = vscan * tau
    vals = np.zeros(nb)
    dt_rad = vscan * dt
    for i in range(nb):
        lmin = i*dl
        lmax = (i+1)*dl
        dlbin = theory[lmin:lmax]
        n=0.0
        val =0.0
        lmin = np.max([300,lmin])
        for m in range(300,lmin+1):
            n += dl 
            val += dtsingle(m,dt_rad,tau_rad) * np.sum(dlbin)
        for m in range(lmin+1,lmax):
            n += dl -(m - lmin)
            val += dtsingle(m,dt_rad,tau_rad) * np.sum(dlbin[m-lmin:])
        val /= n
        vals[i]=val
        #pdb.set_trace()
    return vals


if __name__ == '__main__':

    dolr=True
    freqs=['90','150','220']
    taus = [.0004, .00023, .00019]
    taus = [.00032, .0001, .00002]
    dt = .004015  #try 3.8 ms from software too
    dir = '/big_scratch/cr/xspec_2022/'
    null12s = [dir + 'data_v5/null_spectrum_90.pkl', dir + 'data_v5/null_spectrum_150.pkl', dir + 'data_v5/null_spectrum_220.pkl']
    nulllrs = [dir + 'data_v5_lr/spectrum90_lrnull.pkl', dir + 'data_v5_lr/notau/spectrum150_lrnull.pkl', dir + 'data_v5_lr/notau/spectrum220_lrnull.pkl']
    binned = [ dir + 'spectrum500_90_small.pkl', dir + 'spectrum500_150_small.pkl', dir + 'spectrum500_220_small.pkl']

    allowed_SV = 0.15
    print('Allowed SV is ',allowed_SV)

    
    calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    calibration_factors *= 1e-3 

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

    #imin_dl500 += 1

    cmbfile = '/home/creichardt/cmb_models/plik_plus_r0p01_highell_lensedtotCls_l25000.txt'
    dls = np.loadtxt(cmbfile)
    ells = dls[:,0]
    cmb_dls = dls[:,1]
    #print('first 4 ells:',ells[:5])
    #   cmb_dls_interp = utils.fill_in_theory(cmbfile,ellkern)
    norgfgtheoryfiles = ['/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_90x90.txt',
               '/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_150x150.txt',
               '/home/creichardt/lensing/data_lenspix/3gmodels/dl_fg_220x220.txt']
    norgfgtheory_dls = utils.fill_in_theory(norgfgtheoryfiles,ells)

    pois = 1.2* (ells/3000.)**2

    rg_dls_interp = np.zeros([3,ells.shape[0]])
    facs  = [2.86, 1.06, 0.61]
    rg_dls_interp[0,:] = pois * facs[0]* facs[0]
    rg_dls_interp[1,:] = pois * facs[1]* facs[1]
    rg_dls_interp[2,:] = pois * facs[2]* facs[2]
    fgtheory_dls  = rg_dls_interp + norgfgtheory_dls


    theory_dls = np.zeros([3,ells.shape[0]])
    for i in range(3):
        theory_dls[i,:] = cmb_dls + fgtheory_dls[i,:]



    print('loading nulls')

    for i in range(3):
        print('on ',freqs[i])
        with open(binned[i],'rb') as fp:
            spec= pkl.load(fp)
        with open(null12s[i],'rb') as fp:
            n12= pkl.load(fp)
        if dolr:
            with open(nulllrs[i],'rb') as fp:
                nlr= pkl.load(fp)
        cal = calibration_factors[i]**2
        lrcal = 1./4e-6
        nspectra=1
        nbands=23
        
        theory_dl = theory_dls[i,:].flatten()

        #pseudo_scov = spec['mc_spectrum'].cov #23x23 matrix
        pseudo_dcov = spec['data_spectrum'].est1_cov[:nbands,:nbands] * cal**2
        pseudo_dcov12 = n12.est1_cov[:nbands,:nbands] * cal**2
        if dolr:
            pseudo_dcovlr = nlr.est1_cov[:nbands,:nbands]  * cal**2 * lrcal**2

        Dl = spec['spectrum'].squeeze() * cal
        pseudo12 = n12.spectrum[:nbands,:] * cal
        if dolr:
            pseudolr = nlr.spectrum[:nbands,:] * cal *lrcal

        #pdb.set_trace()
        #choose ranges 


        #apply inverse kernel
        #kernel starts as 1x23x1x23 matrix - can squeeze to 23x23
        #iskips was 0; eskips was 23

        #pdb.set_trace()
        invkernmat =  spec['invkernmat']
        invkernmattr =  spec['invkernmatt']
        Dl12 = np.reshape(np.matmul(invkernmat, np.reshape(pseudo12.T,[nspectra*nbands])),[nspectra*nbands])
        if dolr:
            Dllr = np.reshape(np.matmul(invkernmat, np.reshape(pseudolr.T,[nspectra*nbands])),[nspectra*nbands])
            tautempl = dttautemplate(dt,taus[i],theory_dl,nbands,dl=500)

            Dllr2 = Dllr - tautempl

        sample_cov = spec['sample_cov']
        serr = allowed_SV * np.sqrt(np.diag(sample_cov.squeeze()))
        
        dcov = np.reshape(np.transpose(np.reshape(pseudo_dcov,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
        meas_cov = np.reshape(np.matmul(np.matmul(invkernmat , dcov), invkernmattr),[nspectra,nbands,nspectra, nbands])
        derr = np.sqrt(np.diag(meas_cov.squeeze()))

        dcov = np.reshape(np.transpose(np.reshape(pseudo_dcov12,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
        meas_cov12 = np.reshape(np.matmul(np.matmul(invkernmat , dcov), invkernmattr),[nspectra,nbands,nspectra, nbands])
        derr12 = np.sqrt(np.diag(meas_cov12.squeeze()))
        comb_err12 = derr12 + serr

        if dolr:
            dcov = np.reshape(np.transpose(np.reshape(pseudo_dcovlr,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
            meas_covlr = np.reshape(np.matmul(np.matmul(invkernmat , dcov), invkernmattr),[nspectra,nbands,nspectra, nbands])
            derrlr = np.sqrt(np.diag(meas_covlr.squeeze()))
            comb_errlr = derrlr + serr



        print(freqs[i])
        print('Chisq 12:',np.sum((Dl12[imin_dl500:imax_dl500]/comb_err12[imin_dl500:imax_dl500])**2), ' dof: ', imax_dl500-imin_dl500)
        print('Ratios: ',(Dl12[imin_dl500:imax_dl500]/comb_err12[imin_dl500:imax_dl500]))
        print('power ratios:',Dl12[imin_dl500:imax_dl500]/Dl[imin_dl500:imax_dl500])
        if dolr:
            print('Chisq LR:',np.sum((Dllr[imin_dl500:imax_dl500]/comb_errlr[imin_dl500:imax_dl500])**2), ' dof: ', imax_dl500-imin_dl500)
            print('Chisq LR subtracted:',np.sum((Dllr2[imin_dl500:imax_dl500]/comb_errlr[imin_dl500:imax_dl500])**2), ' dof: ', imax_dl500-imin_dl500)
            print('Ratios: ',(Dllr2[imin_dl500:imax_dl500]/comb_errlr[imin_dl500:imax_dl500]))
            print('power ratios:',Dllr2[imin_dl500:imax_dl500]/Dl[imin_dl500:imax_dl500])

            if False:
                for ii in range(20):
                    tau = .003 + .0001*ii
                    tautempl = dttautemplate(dt,tau,theory_dl,nbands,dl=500)
                    Dllr3 = Dllr - tautempl
                    print('Chisq LR subtracted for :',tau,np.sum((Dllr3[imin_dl500:imax_dl500]/comb_errlr[imin_dl500:imax_dl500])**2))
        pdb.set_trace()
    #pdb.set_trace()
    

    
