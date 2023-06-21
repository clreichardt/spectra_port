
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


class covariance:
    global_index_array = None
    nf = 0
    nspec = 0
    nb = 0 
    def __init__(self,spec,theorydls,calibration_factors,signal_factor=1e-12,factors=[2.86,1.06,0.61]):
        self.nf = factors.shape[0]
        self.nspec = (self.nf * (self.nf+1))/2
        self.nb = spec['sample_cov'].shape[1]
        self.global_index_array = np.zeros([(self.nspec*(self.nspec+1))/2,2,dtype=np.int32])
        self.global_freq_index_array = np.zeros([(self.nspec,2,dtype=np.int32])
        k=0
        for i in range(self.nspec):
            for j in range(i,self.nspec):
                self.global_index_array[k,0] = i
                self.global_index_array[k,1] = j
                k+=1 
        k=0
        for i in range(self.nf):
            for j in range(i,self.nf):
                self.global_freq_index_array[k,0] = i
                self.global_freq_index_array[k,1] = j
                k+=1      
        
        #correct for SV units
        spec['sample_cov'] *= signal_factor

        #We need offidagonal structure for Poisson
        poisson_offdiagonals = self.fit_poisson(spec['sample_cov'][0,:,0,:],factors=factors)

        #We also want off-diagonal structure due to Mll
        offdiagonal_single_block = self.fit_mll_offdiagonal(spec['sample_cov'],spec['meas_cov'])

        #We need diagonals, there will be 21 of these for 3 freqs
        #this is supposed to be 2S**2
        diagonals_signal = self.fit_signal_diagonals(spec['sample_cov'],theory_dls)

        #THis is supposed to be 4SN + 2N**2
        diagonals_noise = self.fit_noise_diagonals(spec['meas_cov'],diagonals_signal)

        #apply calibration factors
        cal_diag_noise = self.apply_calibration(diagonals_noise,calibration_factors)

        #blow this back up to a 4Dim Array
        simple_cov = self.construct_cov(diagonals_signal,cal_diag_noise,offdiagonal_single_block)
        #and combine with poisson terms
        self.cov = simple_cov + poisson_offdiagonals
        #Cov should be my final cov estimate. 
        
    def construct_cov(self,diagonals_signal,diagonals_noise,offdiagonal_single_block):
        cov = np.zeros([self.nspec,self.nb,self.nspec,self.nb])
        for i in range(self.nspec):
            for j in range(i,self.nspec):
                diag = diagonals_noise[self.get_1d_index(i,j),:] + diagonals_signal[self.get_1d_index(i,j),:]
                cov[i,:,j,:] = np.matmul(diag.T, np.matmul(offdiagonal_single_block,diag))
                if i != j:
                    cov[j,:,i,:] = cov[i,:,j,:].T
        return cov
    
    def fit_noise_diagonals(self,cov,signal_diags):
        ncross = (self.nspec * (self.nspec+1))/2
        odiag = np.zeros([ncross,self.nb])
        for k in range(ncross):
            i,j = self.get_2d_indices(k)
            diag = np.diag(cov[i,:,j,:])

            a,b = self.get_2d_freq_indices(i)
            c,d = self.get_2d_freq_indices(j)
            ncommon  = 1* (c == a .or. c == b)  + 1* (d == a .or. d == b)
            match ncommon:
                case 0:
                    odiag[k,:] = self.fit_no_map_in_common(diag)
                case 1:
                    odiag[k,:] = self.fit_one_map_in_common(diag)
                case 2:
                    odiag[k,:] = self.fit_two_map_in_common(diag)
                case _:
                    raise Exception("Wrong number of maps in common")
        
        pass
    
    def fit_one_map_in_common(self,diag):
        '''these are off-diagonals, ala 90x150x90x220, and so on. (ab ac)
        Expectation is aa bc + ac ab
        In the absence of correlated noise, this looks like (after subtracted S*S terms)
        (Naa Sbc) 
        
        Examining these -- they are not high S/N.
        Est1_cov is higher S/N and might work with the smoothing treatment done for fit_two_map_in_common below.
        Have to get calibration right so that I know right ells to transition... TBD
        '''
        
        
        
        return diag    
    
    def fit_two_map_in_common(self,diag,imin=60, nsmooth = 5):
        '''these are diagonals, ala 90x150x90x150, and so on. (ab ab)
        Expectation is aa bb + (ab)**2
        In the absence of correlated noise, this looks like (after subtracted S*S terms)
        Saa Nbb + Sbb Naa + Naa Nbb
        
        Visual inspection -- all look high S/N
        only doing slightly smoothing, in log space to reduce slopes across widths
        padding each edge to reduce edge effects (though not expecting either edge to survive into final bandpowers)
        
        Used imin = 60 as baseline -- this is about l=3000. used to define what points are 0/Inf
        '''
        y=np.log(diag)
        good = y > 0.1 * y[imin]
        ll = np.sum(good)
        nk = nsmooth
        i0 = nk + nk//2  # so 7 for fiducial choice of 5-width kernel
        tmp = np.zeros(ll+2*nk)
        keep = y[good]
        tmp[:nk]=keep[0]
        tmp[-nk:]=keep[-1]
        tmp[nk:ll+nk]=keep
        smtmp = np.convolve(tmp,np.ones(nk)/nk,mode='full') 
        smyy = np.zeros(y.shape[0])
        smyy[good] = smtmp[i0:ll+i0]
        return smyy

    def fit_no_map_in_common(self,diag,ibin = 20):
        #see low-l correlated power so can't just zero outs cross that don't have maps in common. 
        #This is high-S/N below ell ~ 1500 (bin30)
        #looks consistent with zero above ell ~2500-3000 (depending on cross-spectra)
        #The level is small compared to SV though.
        #While noisy inclined to just take values  up to first bin above 1000 that is zero. 
        #zero out all above that bin

        ind = np.min(np.where(diag[ibin:] < 0))
        ind = ind + ibin
        outdiag = diag *0
        outdiag[:ind] = diag[:ind]
        return outdiag


    def fit_poisson(self,cov, factors=[1.],imin_fit = 110, imax_fit = 160, imin_out = 15,dl=50):
        '''
        cv is assumed to be nc x bc cov matrix for the first frequency (90)
        factors are poisson temperature scalings, 
        '''
        nb = self.nb
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


    def get_2d_indices(self,i):
        '''Returns two indices corresponding to position of spectra
        ie 0 (90x90)x(90x90) -> 0,0
        '''
        return self.global_index_array[i,0],self,global_index_array[i,1]
        
        
    
    def get_2d_freq_indices(self,i):
        '''Returns two indices corresponding to position
        ie 0 (90x90) -> 0,0
        '''
        return self.global_freq_index_array[i,0],self,global_freq_index_array[i,1]
        
    def get_1d_index(self,i,j):
        '''Returns the 1d index corresponding to block, ix j
        eg 0x0 -> 0
        0 x 3 -> 3
        Assumes 0 <= i <= j < n;  There is no error checking, except that i/j will be swapped if i > j
        '''
        n = self.nspec
        if i <= j:
            return i*n - ((i-1)*i)/2 + j

        return j*n - ((j-1)*j)/2 + i
    
    def get_theory_cov(self,dls,i,j):
        nspec = self.spec #check if this is right dim

        a,b = self.get_2d_indices(i)
        c,d = self.get_2d_indices(j)
        
        m = self.get_1d_index(a,c)
        n = self.get_1d_index(b,d)
        o = self.get_1d_index(a,d)
        p = self.get_1d_index(c,c)
        out = dls[m,:] * dls[n,:] + dls[o,:] * dls[p,:]
        
        return out
        
        
        

    def signal_func(ells,const,amp):
        return const + amp * (ells)**(-1.3)

    def fit_single_block_signal(self,cov,i_block,j_block, theory_dls,fit_range = [30,190], use_range = [40,250] ):
        theory = self.get_theory_cov(theory_dls,i_block,j_block)
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



    def fit_signal_diagonals(self,sample_cov,theory_dls,fit_range = [30,190], use_range = [40,250] ):
        
        nspec = sample_cov.shape[0]
        nb = sample_cov.shape[1]
        ncross = nspec * (nspec+1)/2
        diagonals = np.zeros([ncross,nb])
        for i in range(nspec):
            for j in range(nspec):
                k = self.get_1d_index(i,j,nspec)
                diagonals[k,:] = self.fit_single_block_signal(sample_cov,i,j,theory_dls,fit_range=fit_range,use_range=use_range)
        return diagonals


    def corr_matrix(self,cov):
        c = np.diag(cov)**0.5
        c[c < 1e-12*np.max(c)] = 1.0 # avoid divide by zero
        cc = np.matmul(c.T,c)
        corr = cov/cc
        return corr



    def single_block_offdiagonal(self,cov):
        pass

    def fit_mll_offdiagonal(self,sample_cov,meas_cov, max_offset = 1, use_noise = [4,5,6], use_sample = [5,6],bininds=[50,160]):
        summed = 0
        n = 0
        for i in use_noise:
            corr = self.corr_matrix(meas_cov[i,:,i,:])
            summed += corr[bininds[0]:bininds[1],bininds[0]:bininds[1]]
            n+=1
        for i in use_sample:
            corr = self.corr_matrix(sample_cov[i,:,i,:])
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
        

    def bin_spectra(self,dl,banddef):
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
    calibration_factors = [1.,1.,1.]
    cov_obj = covariance(spec,theory_dls, calibration_factors,signal_factor=1e-12)        
  

