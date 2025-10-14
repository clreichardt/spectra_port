import numpy as np
import pdb


'''
Oct 2025 -- made beam covs with eell cuts
Aug 19, 2025 -- Added new modes from NDH

This set of code is intended to create a fractional beam covariance matrix for bandpowers.


Inputs:
1) ell_cov (Nl vector) - these are the sampling ells of #2. 
2) beam cov (NfxNl)x(NfxNl) covariance matrix. Nf is number of frequency bands (3 for 3G)

3)ell_beam (Nlb vector) - ell's where beam is reported
4) Nf x Nlb array -- vectors of Bl for each frequency band 

5) ell_win - Nlw vector of ell values for window functions
6) win_array: Nb x Nlw array -- projection from spectra to bandpower space Dl_obs = win_array * Dl_theory

7) ell_theory - Nlt vector of ell values for theory spectra
8) Dls - Ncross x Nlt array of theory spectra for each of the Ncross cross-spectra

Flowchart:

1) Decompose beam cov to eigenvals/vectors 
   a) cut to the top N e-vecs
   b) normalize by sqrt(eval)

1*) Read the new error modes from NDH -- added Aug 19, 2025
    Already normalized by magnitude

2) Loop over e-vecs that are kept:

    I) Take ratio of beam cov evector to beams
        a) interpolate to make match ells
        b) take ratio

    I*) Take ratio of new error modes from NDH (added Aug 19,2025). interpolation is not needed
    
    II) Turn this into an Ncross x Nlt array -- interpolate to ell_theory, and then take appropriate number of powers for each freq band based on the cross-spectra
    
    III) Multiply by theory (Dls) to get modified Dls
    
    IV) Multiply by BPWFs to get modified bandpowers. 
    
    V) Divide by unmodified theory bandpowers to get a bandpower ratio e-vector.
    
    VI) Add ratio^T ratio to the fractional beam cov
    
3) Write to disk the Nb x Nb fraction beam cov
    



'''

def apply_bpwf(win_arr,dls):
    ncross = dls.shape[0]
    nball = win_arr.shape[0]
    nb = nball//ncross

    out = np.zeros(nball,dtype=np.float64)
    for i in range(ncross):
        dli = dls[i,:]
        out[i*nb:(i+1)*nb] = np.matmul(win_arr[i*nb:(i+1)*nb,:],dli)
    
    return out
    

def load_beam_evecs(file,threshold=1e-3):
    with np.load(file) as data:
        ell=data['ell']
        cov=data['cov']
    eval,evec=np.linalg.eigh(cov)
    mx=np.max(eval)
    kept_val = eval[eval>threshold*mx]
    kept_evec = evec[:,eval>threshold*mx]
    nkept = kept_val.shape[0]
    print('keeping {} evalues'.format(nkept))
    for i in range(nkept):
        print(kept_val[i])
        kept_evec[:,i] *= np.sqrt(kept_val[i])
    return kept_evec, ell

def load_beam_evecs_v3beta7(file,threshold=1e-3):
    with np.load(file) as data:
        ell=data['ell']
        kept_evec = data['modes']
        #        cov=data['cov']
        #    eval,evec=np.linalg.eigh(cov)
        #    mx=np.max(eval)
        #    kept_val = eval[eval>threshold*mx]
        #    kept_evec = evec[:,eval>threshold*mx]
        #    nkept = kept_val.shape[0]
        #    for i in range(nkept):
        #        kept_evec[:,i] *= np.sqrt(kept_val[i])
    return kept_evec, ell

def load_new_beam_evecs(file):
    with np.load(file) as data:
        ell=data['ell']
        sidelobe = data['sidelobe_mode']
        mainlobe = data['mainlobe_mode']
    nl = ell.shape[0]
    mainlobe = mainlobe.reshape([3,nl]).T 
    sidelobe = sidelobe.reshape([3,nl]).T 
    return mainlobe, sidelobe, ell

if __name__ == "__main__":
    #1, and inputs 1/2
    #now using v3-beta7
    #norm_evecs, ell_cov = load_beam_evecs_v3beta7('/sptlocal/user/ndhuang/Frankenbeam_v3-beta/unblinded/v3-beta7_final/cov_eigenmodes.npz')
    #now for rc4 beam release
    norm_evecs, ell_cov = load_beam_evecs('/sptlocal/user/ndhuang/Frankenbeam_v3-beta/release_candidates/rc4/cov.npz')

    mainlobe,sidelobe,ell_lobe = load_new_beam_evecs('/home/creichardt/modeling_uncertainty_modes.npz')

    #        '/home/marius311/beamcov_ndh_oct30.npz', threshold = 1e-3)
    nc = ell_cov.shape[0]
    neval = norm_evecs.shape[1]
    #pdb.set_trace()

    #estimated 3% error on error:
    mainlobe   *= 1.03
    sidelobe   *= 1.03
    norm_evecs *= 1.03
    
    #get inputs 5/6 - BPWF
    #v3-beta7
    #dir = '/home/creichardt/highell_dls_blv3b7_fieldpwf/'
    #beam rc4
    dir = '/home/creichardt/highell_dls_blrc5p1_nomidl/'
    win_file = dir+'windowfunc.bin'
    with open(win_file, "rb") as fp:
        line = fp.readline()
        sline = line.decode()
        split = sline.split()
        lmin = int(split[0])
        lmax = int(split[1])
        bpwf = np.fromfile(fp,dtype=np.float32)
    nbpwf = bpwf.shape[0]
    nl = lmax - lmin + 1
    nl_padded = lmax - 1 # number of elements from 2 to lmax, inclusive
    nb = nbpwf // nl
    win_arr = np.zeros([nb,nl_padded], dtype=np.float32)
    win_arr[:,lmin-2:] = bpwf.reshape([nb,nl])
    #implied ell's go from 2 to lmax
    
    #theory spectra - 7/8
    with np.load('/home/creichardt/SPT3G_JAX_Likelihood/spt_dl_components.npz',allow_pickle=True) as npzfile:
        cmb_Dls = npzfile['cmb_Dls']    
        fg_Dls = npzfile['fg_Dls']
        #again implicit ell from 2 to lmax (which may be a different lmax than window functions)
        
    nf=3
    ncross = nf*(nf+1)//2
    for i in range(ncross):
        fg_Dls[i,:] += cmb_Dls 
    
    assert lmax == len(cmb_Dls)+1
        
    #preliminaries:
    # get inputs 3/4 - beam
    #beam_arr = np.loadtxt('/home/creichardt/spt3g_software/beams/products/compiled_2020_beams.txt')
    beam_arr = np.loadtxt('/home/creichardt/bl_v3_beta_rc4.txt')
    #lb = beam_arr[:,0], b90 = beam_arr[:,1], etc.
    # this does not include PWF factors, deliberately.
    #this goes from 0 to 14999 in practice 
    beam_arr = beam_arr[2:lmax+1,:] #cut to relevant ells. 
    
    #similar for lobe modes
    mainlobe = mainlobe[2:lmax+1,:] #cut to relevant ells. 
    sidelobe = sidelobe[2:lmax+1,:] #cut to relevant ells. 

    #now only evectors are on different ell-spacing

    bps0 = apply_bpwf(win_arr,fg_Dls)
    #pdb.set_trace() #check dims
    frac_beam_cov = 0.0
    #2
    spec_vec = fg_Dls * 0
    for i in range(neval):
        this_evec090 = np.interp(beam_arr[:,0],ell_cov,norm_evecs[:nc,i])
        this_evec150 = np.interp(beam_arr[:,0],ell_cov,norm_evecs[nc:2*nc,i])
        this_evec220 = np.interp(beam_arr[:,0],ell_cov,norm_evecs[2*nc:3*nc,i])


        ratio090 = 1+this_evec090/beam_arr[:,1]
        ratio150 = 1+this_evec150/beam_arr[:,2]
        ratio220 = 1+this_evec220/beam_arr[:,3]
        
        #III
        spec_vec[0,:] = fg_Dls[0,:] * ratio090**2
        spec_vec[1,:] = fg_Dls[1,:] * ratio090*ratio150
        spec_vec[2,:] = fg_Dls[2,:] * ratio090*ratio220
        spec_vec[3,:] = fg_Dls[3,:] * ratio150**2
        spec_vec[4,:] = fg_Dls[4,:] * ratio150*ratio220
        spec_vec[5,:] = fg_Dls[5,:] * ratio220**2
        
        #IV
        new_bps = apply_bpwf(win_arr,spec_vec)
        
        #V
        ratio_bps = (new_bps/bps0 - 1.0).reshape([-1,1]) #promote to right dimensions to do matrix multiply below
        
        #VI

        frac_beam_cov += np.matmul(ratio_bps, ratio_bps.T) 
        # pdb.set_trace() #check dims
    #now lobe modes:
    lobes = [mainlobe,sidelobe]
    for mode in lobes:
        ratio090 = 1+mode[:,0]/beam_arr[:,1]
        ratio150 = 1+mode[:,1]/beam_arr[:,2]
        ratio220 = 1+mode[:,2]/beam_arr[:,3]    
        
        #III
        spec_vec[0,:] = fg_Dls[0,:] * ratio090**2
        spec_vec[1,:] = fg_Dls[1,:] * ratio090*ratio150
        spec_vec[2,:] = fg_Dls[2,:] * ratio090*ratio220
        spec_vec[3,:] = fg_Dls[3,:] * ratio150**2
        spec_vec[4,:] = fg_Dls[4,:] * ratio150*ratio220
        spec_vec[5,:] = fg_Dls[5,:] * ratio220**2
        
        #IV
        new_bps = apply_bpwf(win_arr,spec_vec)
        
        #V
        ratio_bps = (new_bps/bps0 - 1.0).reshape([-1,1]) #promote to right dimensions to do matrix multiply below
        
        #VI

        frac_beam_cov += np.matmul(ratio_bps, ratio_bps.T) 
        
    #
    with open(dir+'fractional_beam_cov.bin','wb') as fp:
        frac_beam_cov.astype(np.float64).tofile(fp)

    with open(dir+'fractional_beam_covx4.bin','wb') as fp:
        (4.*frac_beam_cov).astype(np.float64).tofile(fp)
    with open(dir+'fractional_beam_covx100.bin','wb') as fp:
        (100.*frac_beam_cov).astype(np.float64).tofile(fp)

    eval,evec=np.linalg.eigh(frac_beam_cov)
    print(eval) #expect ~30 non-zero evals
    
