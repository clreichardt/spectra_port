import numpy as np
import utils
import unbiased_multispec as spec

if __name__ == "__main__":
    
    end_to_end()
    
    
    
def end_to_end(beamfiles,simbeamfiles=None,do_window_func = True):    
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

    mcdir = workdir + '/mc/'
    mc_specrum = spec.unbiased_multispec(mapfiles,window,banddef,nside,lmax=lmax,
                                         resume=resume,
                                         basedir=mcdir,
                                         setdef=setdef_mc,
                                         jackknife=False, auto=True,
                                         kmask=None,
                                         cmbweighting=True)
    mc_specrum_fine = spec.unbiased_multispec(mapfiles,window,banddef_fine,nside,lmax=lmax,
                                              resume=True, #reuse the SHTs
                                              basedir=mcdir,
                                              setdef=setdef_mc,
                                              jackknife=False, auto=True,
                                              kmask=None,
                                              skipcov=True,
                                              cmbweighting=True)

    ##################
    # 3: Calculate spectra and covariances of data
    ##################
    datadir = workdir + '/data/'
    data_specrum = spec.unbiased_multispec(mapfiles,window,banddef,nside,lmax=lmax,
                                      resume=resume,
                                      basedir=datadir,
                                      setdef=setdef,
                                      jackknife=False, auto=True,
                                      kmask=None,
                                      cmbweighting=True)

    ##################
    # 4: Calculate the Transfer functions
    ##################
    nkern = len(ellkern)
    
    
    #Get Beams
    assert(len(beamfiles) == nsets)
    beams_interp = utils.fill_in_beams(beamfiles,ellkern)
    beams = utils.explode_beams(beams)
    if simbeamfiles is None:
        simbeams_interp=beams_interp
        simbeams=beams
    else:
        assert(len(simbeamfiles) == nsets)
        simbeams_interp = utils.fill_in_beams(simbeamfiles,ellkern)
        simbeams = utils.explode_beams(simbeams_interp)
    #see transfer_function.py

    '''
    ; By default, we make one transfer function per set, and 
    ; cross spectra are scaled  by the geometric mean of the
    ; appropriate single set spectra. However it
    ; is an option to make a transfer function for cross spectrum 
    '''

    beams_for_tf=simbeam_interp
    ntfs=nsets ;  this might be changed below

'''
; This is where we need to know the theory spectrum of the monte carlo sims
; The user has the option of inputting one file per set OR
;  a list of files or "components" per set. 
; Thirdly, if the user specifies one component per spectrum (including 
; cross spectra), then a separate transfer function will be made for
; the cross spectra, (rather than using the geometric mean of the
; the single frequency spectra)
'''

    ##################
    # 5: Create binned kernel, including mode-coupling, beams, pixel WFs, transfer functions
    #     Optionally allow different beams in sims (and thus kernel)
    #    Invert the kernel
    ##################

    super_kernel        = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    sim_super_kernel    = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    inv_super_kernel    = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    inv_sim_super_kernel= np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
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
    spectrum = np.reshape(invkernmat * data_spectrum.spectrum,[nspectra,nbands])

    ##################
    # 7: Apply inverse kernel to the covariance matrices
    ##################
    sample_cov = np.reshape(invsimkernmat * mc_spectrum.cov * invsimkernmattr,[nspectra,nbands,nspectra, nbands])
    meas_cov = np.reshape(invkernmat * data_spectrum.cov * invkernmattr,[nspectra,nbands,nspectra, nbands])
    
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
