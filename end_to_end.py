import numpy as np


if __name__ == "__main__":
    
    end_to_end()
    
    
    
def end_to_end(do_window_func = True):    
    '''
    should add argument to this...
    '''
    real_map_dir = ''
    real_map_stub= ''

    ##################
    # 1: Make (or load) mode coupling kernel
    ##################


    #plan to use wrapper to NaMaster, to be written

    ##################
    # 2: Calculate spectra and covariances of monte-carlo sims
    #    This is done at both a fine ell-gridding for Tf, and broader binning that matches data
    ##################

    #see unbiased_multispec.py

    ##################
    # 3: Calculate spectra and covariances of data
    ##################

    #see unbiased_multispec.py


    ##################
    # 4: Calculate the Transfer functions
    ##################

    #see transfer_function.py


    ##################
    # 5: Create binned kernel, including mode-coupling, beams, pixel WFs, transfer functions
    #     Optionally allow different beams in sims (and thus kernel)
    #    Invert the kernel
    ##################

    super_kernel        = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    sim_super_kernel    = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    inv_super_kernel    = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    inv_sim_super_kernel= np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    defaultskip=1
    iskips = np.zeros(nspectra, dtype=np.int32)
    for i in range(nspectra):
        super_kernel[i,:,i,:]=rebin_coupling_kernel(kernel, ellkern, banddef, transfer=transfer[i,:], beam = beams[i,:])
        sim_super_kernel[i,:,i,:]=rebin_coupling_kernel(kernel, ellkern, banddef, transfer=transfer[i,:], beam = simbeams[i,:])
        slice = [superkern[i,j,i,j]==0 for j in range(nbands)]
        try:
            iskip = np.where(slice)[-1][-1]
        except IndexError:
            iskip=0 #end up here if np.where returned empty array -- all false
        # leave first (or more) usually bogus bin out of inversion
        #don't try to divide by zero
        inv_super_kernel[i,iskip:,i,iskip:] = np.linalg.inv(super_kernel[i,iskip:,i,iskip:])
        inv_sim_super_kernel[i,iskip:,i,iskip:] = np.linalg.inv(sim_super_kernel[i,iskip:,i,iskip:])
        
    invkernmat = np.reshape(inv_super_kernel,[nspectra*nbands, nbands*nspectra]))
    invkernmattr = np.transpose(invkernmat)
    invsimkernmat = np.reshape(inv_sim_super_kernel,[nspectra*nbands, nbands*nspectra]))
    invsimkernmattr = np.transpose(invsimkernmat)

    ##################
    # 6: Multiply data bandpowers by Inverse Kernel
    ##################
    spectrum = np.reshape(invkernmat * spectrum_data_raw,[nspectra,nbands])

    ##################
    # 7: Apply inverse kernel to the covariance matrices
    ##################
    sample_cov = np.reshape(invsimkernmat * cov_mc_raw * invsimkernmattr,[nspectra,nbands,nspectra, nbands])
    meas_cov = np.reshape(invkernmat * cov_data_raw * invkernmattr,[nspectra,nbands,nspectra, nbands])
    
    ##################
    # 8: Combine covariances yield total covariance estimate
    ##################
    cov = meas_cov + sample_cov

    ##################
    # 9: Optionally: Calculate bandpower window functions
    ##################

    if do_window_func:
        assert(win_minell < win_maxell)
        nlwin = win_maxell-win_minell+1
        windowfunc = np.zeros([nbands*nspectra,nlwin],dtype=np.float32)
        for i in range(nspectra):
            iskip = iskips[i]
            transdic = {'ell':ellkern, 
                        'kernel':kernel,
                        'transfer':transfer[i,:],
                        'bl':beams[i,:]}
            windowfunc[iskip+i*nbands:nbands+i*nbands,:] = 
                window_function_calc(banddef,transdic,nskip=iskip,ellmin = win_minell,ellmax=win_maxell)
