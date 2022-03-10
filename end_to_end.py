import numpy as np
import utils
import unbiased_multispec as spec

if __name__ == "__main__":
    
    banddef = np.arange(0,11000,500)
    mapfiles = np.zeros([3,4])
    setdef_mc = np.zeros(100)
    setdef = np.zeros(100)
    workdir = '/scratch/cr/xspec_2022/'

    mask = utils.load_window(mask_file)
    
    output = end_to_end( mapfiles,
                         mcmapfiles,
                         banddef,
                         beamfiles,
                         workdir,
                         simbeamfiles=None,
                         setdef=setdef,
                         setdef_mc=setdef_mc,
                         do_window_func=False, 
                         banddef = banddef,
                         lmax=13000,
                         nside=8192,
                         kmask=None,
                         mask=mask,
                         kernel_file ='Placeholder',
                         resume-True
                       )
    with open(file_out,'wb') as fp:
        pkl.save(output,fp)
    
    
    
    
def end_to_end(mapfiles,
               mcmapfiles,
               banddef,
               beamfiles,
               workdir,
               setdef=None,
               setdef_mc=None,
               simbeamfiles=None,
               do_window_func = True,
               lmax=13000,
               nside=8192,
               kmask=None,
               mask=None,
               kernel_file=None,
               resume=True):    
    '''
    should add argument to this...
    '''

    #storage dictionary for later monitoring
    output = {}
    output['mask']=mask
    output['kmask']=kmask
    output['lmax']=lmax
    output['nside']=nside
    output['resume']=resume
    output['mapfiles']=mapfiles
    output['mcmapfiles']=mcmapfiles
    output['banddef']=banddef
    output['beamfiles']=beamfiles
    output['simbeamfiles']=simbeamfiles
    output['kernel_file']=kernel_file
    output['workdir']=workdir
    output['do_window_func']=do_window_func
    
    ##################
    # 1: Make (or load) mode coupling kernel
    ##################


    
    # should have already done a binned version of the M_ll
    # also need to have move it to Dl space 
    # this code is here, but commented out
    #
    
    #plan to use wrapper to NaMaster, to be written
    info, kernel = utils.load_mll(kernel_file)
    banddef_fine = utils.bands_from_range(info)
    ellkern = utils.band_centers(banddef_fine)

    #and store
    output['info']=info
    output['kernel']=kernel
    output['ellkern']=ellkern
    output['banddef_fine']=banddef_fine

    ##################
    # 2: Calculate spectra and covariances of monte-carlo sims
    #    This is done at both a fine ell-gridding for Tf, and broader binning that matches data
    ##################

    mcdir = workdir + '/mc/'

    # will used alm's in mcdir+'shts_processed.bin'
    mc_specrum      = spec.unbiased_multispec(mcmapfiles,mask,banddef,nside,
                                              lmax=lmax,
                                              resume=resume,
                                              basedir=mcdir,
                                              setdef=setdef_mc,
                                              jackknife=False, auto=False,
                                              kmask=kmask,
                                              cmbweighting=True)
    
    mc_specrum_fine = spec.unbiased_multispec(mcmapfiles,mask,banddef_fine,nside,
                                              lmax=lmax,
                                              resume=True, #reuse the SHTs
                                              basedir=mcdir,
                                              setdef=setdef_mc,
                                              jackknife=False, auto=False,
                                              kmask=kmask,
                                              skipcov=True,
                                              cmbweighting=True)

    output['mc_spectrum']=mc_spectrum
    output['mc_spectrum_fine']=mc_spectrum_fine

    ##################
    # 3: Calculate spectra and covariances of data
    ##################
    datadir = workdir + '/data/'
    data_specrum    = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=lmax,
                                              resume=resume,
                                              basedir=datadir,
                                              setdef=setdef,
                                              jackknife=False, auto=False,
                                              kmask=kmask,
                                              cmbweighting=True)

    output['data_spectrum']=data_spectrum
    
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
    output['beams']=beams
    output['beams_interp']=beams_interp
    output['simbeams']=simbeams
    output['simbeams_interp']=simbeams_interp
        
    #see transfer_function.py

    '''
    ; By default, we make one transfer function per set, and 
    ; cross spectra are scaled  by the geometric mean of the
    ; appropriate single set spectra. However it
    ; is an option to make a transfer function for cross spectrum 
    '''

    beams_for_tf=simbeam_interp
    ntfs=nsets #  this might be changed below

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
        super_kernel[i,:,i,:]     = rebin_coupling_kernel(kernel, ellkern, banddef, transfer=transfer[i,:], beam = beams[i,:])
        sim_super_kernel[i,:,i,:] = rebin_coupling_kernel(kernel, ellkern, banddef, transfer=transfer[i,:], beam = simbeams[i,:])
        slice = [superkern[i,j,i,j]==0 for j in range(nbands)]
        try:
            iskip = np.where(slice)[-1][-1]
        except IndexError:
            iskip=0 #end up here if np.where returned empty array -- all false
        # leave first (or more) usually bogus bin out of inversion
        #don't try to divide by zero
        inv_super_kernel[i,iskip:,i,iskip:] = np.linalg.inv(super_kernel[i,iskip:,i,iskip:])
        inv_sim_super_kernel[i,iskip:,i,iskip:] = np.linalg.inv(sim_super_kernel[i,iskip:,i,iskip:])
        
    invkernmat      = np.reshape(inv_super_kernel,[nspectra*nbands, nbands*nspectra]))
    invkernmattr    = np.transpose(invkernmat)
    invsimkernmat   = np.reshape(inv_sim_super_kernel,[nspectra*nbands, nbands*nspectra]))
    invsimkernmattr = np.transpose(invsimkernmat)

    output['super_kernel']=super_kernel
    output['sim_super_kernel']=sim_super_kernel
    output['inv_super_kernel']=inv_super_kernel
    output['inv_sim_super_kernel']=inv_sim_super_kernel
    output['invkernmat']=invkernmat
    output['invkernmatt']=invkernmattr
    output['invsimkernmat']=invsimkernmat
    output['invsimkernmatt']=invsimkernmattr
                
    ##################
    # 6: Multiply data bandpowers by Inverse Kernel
    ##################
    spectrum = np.reshape(invkernmat * data_spectrum.spectrum,[nspectra,nbands])

    output['spectrum']=spectrum
    ##################
    # 7: Apply inverse kernel to the covariance matrices
    ##################
    sample_cov = np.reshape(invsimkernmat * mc_spectrum.cov * invsimkernmattr,[nspectra,nbands,nspectra, nbands])
    meas_cov = np.reshape(invkernmat * data_spectrum.cov * invkernmattr,[nspectra,nbands,nspectra, nbands])

    output['sample_cov']=sample_cov
    output['meas_cov']=meas_cov
    ##################
    # 8: Combine covariances yield total covariance estimate
    ##################
    cov = meas_cov + sample_cov

    output['cov']=cov
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


    output['windowfunc']=windowfunc

    
    return output
