
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
from spectra_port import unbiased_multispec, utils
import pdb
#from spt3g import core,maps, calibration



class covariance:
    global_index_array = None
    nf = 0
    nspec = 0
    nb = 0 
    def __init__(self,spec,theory_dls,calibration_factors,factors=np.asarray([2.86,1.06,0.61])):
        self.nf = factors.shape[0]
        self.nspec = (self.nf * (self.nf+1))//2
        self.nb = spec['sample_cov'].shape[1]
        self.global_index_array = np.zeros([(self.nspec*(self.nspec+1))//2,2],dtype=np.int32)
        self.global_freq_index_array = np.zeros([self.nspec,2],dtype=np.int32)
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
        nbands = self.nb
        nspectra = self.nspec
        sample_cov = spec['sample_cov']
        meas_cov = spec['meas_cov']
        invkernmat=spec['invkernmat']
        invkernmattr=spec['invkernmatt']
        dcov = np.reshape(np.transpose(np.reshape(spec['data_spectrum'].est1_cov,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
        meas_cov1 = np.reshape(np.matmul(np.matmul(invkernmat , dcov), invkernmattr),[nspectra,nbands,nspectra, nbands])
        #dcov = np.reshape(np.transpose(np.reshape(spec['data_spectrum'].est2_cov,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
        #meas_cov2 = np.reshape(np.matmul(np.matmul(invkernmat , dcov), invkernmattr),[nspectra,nbands,nspectra, nbands])
        #These are estimators 1 and 2 respectively.
        self.meas_cov = meas_cov
        self.meas_cov1 = meas_cov1
        self.sample_cov = sample_cov

        
        #correct for units and calbration
        #Nothing to do for sample_cov
        
        #lots to do for meas_cov in each of three variations.
        #apply calibration
        for i in range(self.nspec):
                a,b = self.get_2d_freq_indices(i)
                factor = calibration_factors[a]*calibration_factors[b]
                meas_cov[i,:,:,:]  *= factor
                meas_cov[:,:,i,:]  *= factor
                meas_cov1[i,:,:,:] *= factor
                meas_cov1[:,:,i,:] *= factor
                #meas_cov2[i,:,:,:] *= factor
                #meas_cov2[:,:,i,:] *= factor

        print("Finished setup")
        #We need offidagonal structure for Poisson
        self.poisson_offdiagonals = self.fit_poisson(sample_cov[0,:,0,:].squeeze(),factors=factors)
        print("Fiished poisso")
        #We also want off-diagonal structure due to Mll
        self.offdiagonal_single_block = self.fit_mll_offdiagonal(sample_cov,meas_cov)
        print("Finished corr matrix")
        #We need diagonals, there will be 21 of these for 3 freqs
        #this is supposed to be 2S**2
        diagonals_signal = self.fit_signal_diagonals(sample_cov,theory_dls)
        print("Finished signal diag")
        raw_diags = self.get_diags(meas_cov)
        raw_diags1 = self.get_diags(meas_cov1)
        #raw_diags2 = self.get_diags(meas_cov2)
        
        #THis is supposed to be 4SN + 2N**2
        diagonals_noise = self.fit_noise_diagonals(meas_cov,diagonals_signal,raw_diags,raw_diags1)
        print("Finished noise diag")
        #blow this back up to a 4Dim Array
        self.simple_cov = self.construct_cov(diagonals_signal,diagonals_noise,self.offdiagonal_single_block)
        #and combine with poisson terms
        self.cov = self.simple_cov + self.poisson_offdiagonals
        self.diagonals_signal = diagonals_signal
        self.diagonals_noise = diagonals_noise
        self.raw_noise_diags = raw_diags
        self.raw_noise_diags_est1 = raw_diags1
        #Cov should be my final cov estimate. 
        print("Finished cov codition")

    def construct_cov(self,diagonals_signal,diagonals_noise,offdiagonal_single_block):
        cov = np.zeros([self.nspec,self.nb,self.nspec,self.nb])
        for i in range(self.nspec):
            for j in range(i,self.nspec):
                sqrtdiag = np.sqrt(diagonals_noise[self.get_1d_index(i,j),:] + diagonals_signal[self.get_1d_index(i,j),:])
                sqrtdiag2d = np.tile(sqrtdiag,[self.nb,1])
                cc = sqrtdiag2d* sqrtdiag2d.T * offdiagonal_single_block
                cov[i,:,j,:] = cc#np.matmul(diag.T, np.matmul(offdiagonal_single_block,diag))
                #pdb.set_trace()
                if i != j:
                    cov[j,:,i,:] = cov[i,:,j,:].T
        return cov
    
    def get_diags(self,cov):
        ncross = (self.nspec * (self.nspec+1))//2
        odiag = np.zeros([ncross,self.nb])
        for k in range(ncross):
            i,j = self.get_2d_indices(k)
            odiag[k,:] = np.diag(cov[i,:,j,:])
        return odiag

    def fit_noise_diagonals(self,cov,signal_diags,raw_diags,raw_diags_est1):
        ncross = (self.nspec * (self.nspec+1))//2
        odiag = np.zeros([ncross,self.nb])
        for k in range(ncross):
            i,j = self.get_2d_indices(k)
            diag = raw_diags[k,:]

            a,b = self.get_2d_freq_indices(i)
            c,d = self.get_2d_freq_indices(j)
            ncommon  = 1* (c == a or c == b)  + 1* (d == a or d == b) 
            if ncommon == 2 and c == d and a != b:
                ncommon -= 1
            print(';number matches:',ncommon,a,b,c,d)
            match ncommon:
                case 0:
                    odiag[k,:] = self.fit_no_map_in_common(diag)
                case 1:
                    odiag[k,:] = self.fit_one_map_in_common(k,diag,signal_diags,raw_diags_est1)
                case 2:
                    odiag[k,:] = self.fit_two_map_in_common(diag)
                case _:
                    raise Exception("Wrong number of maps in common")
                
        return odiag

    
    def fit_one_map_in_common(self,k,diag,signal_diags,raw_diags_est1,imin=60,nsmooth=5,bin_transition=120):
        '''these are off-diagonals, ala 90x150x90x220, and so on. (ab ac)
        Expectation is aa bc + ac ab
        In the absence of correlated noise, this looks like (after subtracted S*S terms)
        (Naa Sbc) 
        
        Examining these -- they are not high S/N.
        Est1_cov is higher S/N and might work with the smoothing treatment done for fit_two_map_in_common below.
        Have to get calibration right so that I know right ells to transition... TBD
        
        Assume that have all signal_diags entered
        Have some noise diags entered (not ones with 1 map in common)
        '''
        i,j = self.get_2d_indices(k)
        a,b = self.get_2d_freq_indices(i)
        c,d = self.get_2d_freq_indices(j)
        odiag = diag *0.0
        
        #first get smoothed version at low-l
        smdiag = (self.fit_two_map_in_common(diag[:bin_transition+nsmooth],imin=imin,nsmooth=nsmooth))[:bin_transition]
        odiag[:bin_transition] = smdiag
        
        #now the annoying bits at higher ell...
        # this is a cross of the form (ab)(ac); b != c, but a possibly equal to b or c. 
        #No hi-l correlated noise, so just Sbc N_aa term.
        #first let's reorder my a,b,c,d such that we know which is in common and which is not.
        #print('1mapcommon:',a,b,c,d)
        if c == a:
            common = a
            if b <= d:
                others = [b,d]
            else:
                others = [d,b]
        elif c == b:
            common = b
            if a <= d:
                others = [a,d]
            else:
                others = [d,a]
        elif d == a:
            common = a
            if b <= c:
                others = [b,c]
            else:
                others = [c,b]
        else:
            common = b
            if a <= c:
                others = [a,c]
            else:
                others = [c,a]
        #Noise auto term:
        i_auto = self.get_1d_index(common,common,n=self.nf)
        k00 = self.get_1d_index(i_auto,i_auto)
        two_n_squared = self.fit_two_map_in_common(raw_diags_est1[k00,:])
        n_auto = np.sqrt(two_n_squared/2.0)
        
        #signal cross term:
        i_auto_11 = self.get_1d_index(others[0],others[0],n=self.nf)
        i_auto_22 = self.get_1d_index(others[1],others[1],n=self.nf)
        k12 = self.get_1d_index(i_auto_11,i_auto_22)
        two_s_squared = signal_diags[k12,:]
        s_cross = np.sqrt(two_s_squared/2.0)
        
        prefactor = 1.0 + 1.0 *(a == b or c == d)
        #for something like 90x150x150x220, expect S*N
        # for somethnig like 90x150x150x150, expect 2 S*N
        hi_diag = prefactor * n_auto * s_cross
        #may want to do somethnig else to join them....
        odiag[bin_transition:] = hi_diag[bin_transition:]
        
        
        return odiag    
    
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
        #cutxx = xx[imin_fit:imax_fit]
        #cut_cov = cov[imin_fit:imax_fit,imin_fit:imax_fit]
        avg=0.0
        n=0
        for i in range(imin_fit,imax_fit):
            for j in range(i+1,imax_fit):
                avg += cov[i,j]/xx[0,i]/xx[0,j]
                n   += 1 
        avg /= n
        template=np.matmul(xxout.T,xxout)
        for i in range(nb):
            template[i,i] = 0    
        template *= avg
        

        nf = factors.shape[0]
        ncombo = (nf * (nf+1))//2  
        #how to scale template to each band (ie 90x150)
        scaling = np.zeros(ncombo)
        k=0
        for i in range(nf):
            for j in range(i,nf):
                scaling[k] = factors[i]/factors[0] * factors[j]/factors[0]
                k+= 1
        
        poisson = np.zeros([ncombo,nb,ncombo,nb])
        for i in range(ncombo):
            for j in range(i,ncombo):
                poisson[i,:,j,:] = template * scaling[i] * scaling[j]

                if i != j:
                    poisson[j,:,i,:] = poisson[i,:,j,:]
        return poisson


    def get_2d_indices(self,i):
        '''Returns two indices corresponding to position of spectra
        ie 0 (90x90)x(90x90) -> 0,0
        '''
        return self.global_index_array[i,0],self.global_index_array[i,1]
        
        
    
    def get_2d_freq_indices(self,i):
        '''Returns two indices corresponding to position
        ie 0 (90x90) -> 0,0
        '''
        return self.global_freq_index_array[i,0],self.global_freq_index_array[i,1]
        
    def get_1d_index(self,i,j,n=None):
        '''Returns the 1d index corresponding to block, ix j
        eg 0x0 -> 0
        0 x 3 -> 3
        Assumes 0 <= i <= j < n;  There is no error checking, except that i/j will be swapped if i > j
        '''
        if n is None:
            n = self.nspec

        if i <= j:
            return i*n - ((i+1)*i)//2 + j

        return j*n - ((j+1)*j)//2 + i
    
    def get_theory_cov(self,dls,i,j):
        #nspec = self.nspec #check if this is right dim

        #print(i,j)
        a,b = self.get_2d_freq_indices(i)
        c,d = self.get_2d_freq_indices(j)
        #print(a,b,c,d)
        m = self.get_1d_index(a,c,n=self.nf)
        n = self.get_1d_index(b,d,n=self.nf)
        o = self.get_1d_index(a,d,n=self.nf)
        p = self.get_1d_index(b,c,n=self.nf)
        #print(m,n,o,p)
        out = dls[m,:] * dls[n,:] + dls[o,:] * dls[p,:]
        
        return out
        
        
        

    def fit_single_block_signal(self,cov,i_block,j_block, theory_dls,fit_range = [30,190], use_range = [40,250] ):

        def signal_func(ells,const,amp):
            #print(ells.shape,const.shape,amp.shape)
            return const + amp * (ells)**(-1.3)


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
        params = scipy.optimize.curve_fit(signal_func,fit_inds,fit_vals,p0=p0,sigma=sigma)
        
        prefactor = obs_prefactor
        prefactor[use_range[0]:use_range[1]] = signal_func(inds[use_range[0]:use_range[1]],params[0][0],params[0][1])

        out = prefactor * theory
        return out



    def fit_signal_diagonals(self,sample_cov,theory_dls,fit_range = [30,190], use_range = [40,250] ):
        
        nspec = sample_cov.shape[0]
        nb = sample_cov.shape[1]
        ncross = nspec * (nspec+1)//2
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

    def fit_mll_offdiagonal(self,sample_cov,meas_cov, max_offset = 1, use_noise = [3,4,5], use_sample = [4,5],bininds=[50,160]):
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
            for j in range(nb-i-1):
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

    print("initiating files")
    dlfile='/big_scratch/cr/xspec_2022/spectrum_small.pkl'
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
        theory_dls[i,:] = bin_spectra(cmb_dls + fgtheory_dls[i,:],spec['banddef'])
    calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    calibration_factors *= 1e-3  #correction for units between sims and real data. The transfer function brings it over.  This ends up being brought to the 4 power so 1e-12 effectively.
    
    print("initiating cov")

    cov_obj = covariance(spec,theory_dls, calibration_factors)        
    covfile = '/big_scratch/cr/xspec_2022/covariance.pkl'
    with open(covfile,'wb') as fp:
        pkl.dump(cov_obj, fp)
