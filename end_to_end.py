
import numpy as np
import utils
import unbiased_multispec as spec
import time,os
import pickle as pkl
    
    
    
def end_to_end(mapfiles,
               mcmapfiles,
               banddef,
               beam_arr,
               theoryfiles,
               workdir,
               setdef=None,
               setdef_mc1=None,
               setdef_mc2=None,
               simbeam_arr=None,
               do_window_func = True,
               lmax=13000,
               nside=8192,
               kmask=None,
               mask=None,
               kernel_file=None,
               checkpoint=True,
               resume=True):    
    '''
    should add argument to this...
    '''

        
    #storage dictionary for later monitoring
    output = {}
    output['mask']=mask
    fskyw2 = np.mean(mask*mask)
    output['fskyw2']=fskyw2
    output['kmask']=kmask
    output['lmax']=lmax
    output['nside']=nside
    output['resume']=resume
    output['mapfiles']=mapfiles
    output['mcmapfiles']=mcmapfiles
    output['banddef']=banddef
    output['theoryfiles']=theoryfiles
    output['beam_arr']=beam_arr
    output['simbeam_arr']=simbeam_arr
    output['kernel_file']=kernel_file
    output['workdir']=workdir
    output['do_window_func']=do_window_func
    output['setdef']=setdef
    output['setdef_mc1']=setdef_mc1
    output['setdef_mc2']=setdef_mc2
    
    if checkpoint:
        with open(os.path.join(workdir,'tmp_start.pkl'),'wb') as fp:
            pkl.dump(output,fp)

    ##################
    # 1: Make (or load) mode coupling kernel
    ##################
    lasttime=time.time()
    print('load mll')

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

    newtime=time.time()
    print('run sim unbiased: last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
        
    mcdir = workdir + '/mc/'

    try:
        if not resume:
            raise Exception("Bounce out")
        with open(mcdir+'mc_spectrum.pkl', 'rb') as f:
            mc_spectrum = pkl.load(f)
    except:
    # will used alm's in mcdir+'shts_processed.bin'
        mc_spectrum      = spec.unbiased_multispec(mcmapfiles,mask,banddef,nside,
                                                lmax=lmax,
                                                resume=resume,
                                                basedir=mcdir,
                                                persistdir=mcdir,
                                                setdef=setdef_mc1,
                                                setdef2=setdef_mc2,
                                                jackknife=False, auto=False,
                                                kmask=kmask,
                                                cmbweighting=True)
        with open(mcdir+'mc_spectrum.pkl', 'wb') as f:
            pkl.dump(mc_spectrum,f)
            
            
            
    try:
        if not resume:
            raise Exception("Bounce out")
        with open(mcdir+'mc_spectrum_fine.pkl', 'rb') as f:
            mc_spectrum_fine = pkl.load(f)
    except:    
        mc_spectrum_fine = spec.unbiased_multispec(mcmapfiles,mask,banddef_fine,nside,
                                                lmax=lmax,
                                                resume=True, #reuse the SHTs
                                                basedir=mcdir,
                                                persistdir=mcdir,
                                                setdef=setdef_mc1,
                                                setdef2=setdef_mc2,
                                                jackknife=False, auto=False,
                                                kmask=kmask,
                                                skipcov=True,
                                                cmbweighting=True)
        with open(mcdir+'mc_spectrum_fine.pkl', 'wb') as f:
            pkl.dump(mc_spectrum_fine,f)

    output['mc_spectrum']=mc_spectrum
    output['mc_spectrum_fine']=mc_spectrum_fine

    ##################
    # 3: Calculate spectra and covariances of data
    ##################
    
    newtime=time.time()
    print('run data unbiased: last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    datadir = workdir + '/data/'
    try:
        if not resume:
            raise Exception("Bounce out")
        with open(datadir+'data_spectrum.pkl', 'rb') as f:
            data_spectrum = pkl.load(f)
    except:    
        data_spectrum    = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                                lmax=lmax,
                                                resume=resume,
                                                basedir=datadir,
                                                persistdir=datadir,
                                                setdef=setdef,
                                                jackknife=False, auto=False,
                                                kmask=kmask,
                                                cmbweighting=True)

        with open(datadir+'data_spectrum.pkl', 'wb') as f:
            pkl.dump(data_spectrum,f)
    
    output['data_spectrum']=data_spectrum
    
    ##################fun
    # 4: Calculate the Transfer functions
    ##################
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    print('run Tf')

    nkern = len(ellkern)

    
    nspectra=mc_spectrum_fine.spectrum.shape[1]
    #Get Beams
    nsets = setdef.shape[1]
    assert beam_arr.shape[1] == nsets+1
    assert beam_arr.shape[0] >= lmax+1

    beams = utils.fill_in_beams(beam_arr,ellkern)
    beams_interp = utils.explode_beams(beams)
    if simbeam_arr is None:
        simbeams_interp=beams_interp
        simbeams=beams
    else:
        assert(simbeam_arr.shape[1] == nsets+1)
        simbeams_interp = utils.fill_in_beams(simbeam_arr,ellkern)
        simbeams = utils.explode_beams(simbeams_interp)
    output['beams']=beams
    output['beams_interp']=beams_interp
    output['simbeams']=simbeams
    output['simbeams_interp']=simbeams_interp
    
    if add_pixel_to_beam_for_sims:
        #add pixel window function to beam
        pdb.set_trace()
    
    #get theory
    assert(len(theoryfiles) == nsets)
    #same code should work as beams -- interpolate file values to chosen ells
    # may need to change for file formats TBD
    theory_dls_interp = utils.fill_in_theory(theoryfiles,ellkern)
    output['theory_dls_interp']=theory_dls_interp
    niter = 5
    output['niter']=niter
    
    '''
    ; By default, we make one transfer function per set, and tr 
    ; cross spectra are scaled  by the geometric mean of the
    ; appropriate single set spectra. However it may be a future 
    ; option to make a transfer function for cross spectrum 
    '''

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
    
    transfer_iter = np.zeros([ntfs,niter,nkern])
    for i in range(nsets):        
        cl_mc = mc_spectrum_fine.spectrum[:,i]
        transfer_iter[i,0,:] = utils.initial_estimate(cl_mc, theory_dls_interp[i,:], simbeams_interp[i,:], fskyw2)
        for j in range(1,niter):
            transfer_iter[i,j,:] = utils.transfer_iteration( transfer_iter[i,j-1,:], cl_mc, theory_dls_interp[i,:], simbeams_interp[i,:], fskyw2, kernel)
    
    transfer = np.zeros([nspectra,nkern])
    k=0
    for i in range(nsets): 
        for j in range(i,nsets):
            transfer[k,:] = np.sqrt(transfer_iter[i,-1,:]*transfer_iter[j,-1,:])
    
    output['transfer_iter'] = transfer_iter
    output['transfer']=transfer
            
    ##################
    # 5: Create binned kernel, including mode-coupling, beams, pixel WFs, transfer functions
    #     Optionally allow different beams in sims (and thus kernel)
    #    Invert the kernel
    ##################
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    print('binned kernels')
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
        
    invkernmat      = np.reshape(inv_super_kernel,[nspectra*nbands, nbands*nspectra])
    invkernmattr    = np.transpose(invkernmat)
    invsimkernmat   = np.reshape(inv_sim_super_kernel,[nspectra*nbands, nbands*nspectra])
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
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    print('unbias spectra')
    spectrum = np.reshape(np.matmul(invkernmat, data_spectrum.spectrum),[nspectra,nbands])

    output['spectrum']=spectrum
    ##################
    # 7: Apply inverse kernel to the covariance matrices
    ##################
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    print('unbias cov')
    sample_cov = np.reshape(np.matmul(np.matmul(invsimkernmat , mc_spectrum.cov) , invsimkernmattr),[nspectra,nbands,nspectra, nbands])
    meas_cov = np.reshape(np.matmul(np.matmul(invkernmat , data_spectrum.cov), invkernmattr),[nspectra,nbands,nspectra, nbands])

    output['sample_cov']=sample_cov
    output['meas_cov']=meas_cov
    ##################
    # 8: Combine covariances yield total covariance estimate
    ##################
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    print('combine cov')
    cov = meas_cov + sample_cov

    output['cov']=cov
    ##################
    # 9: Optionally: Calculate bandpower window functions
    ##################
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime

    if do_window_func:
        print('window funcs')
        assert(win_minell < win_maxell)
        nlwin = win_maxell-win_minell+1
        windowfunc = np.zeros([nbands*nspectra,nlwin],dtype=np.float32)
        for i in range(nspectra):
            iskip = iskips[i]
            transdic = {'ell':ellkern, 
                        'kernel':kernel,
                        'transfer':transfer[i,:],
                        'bl':beams[i,:]}
            windowfunc[iskip+i*nbands:nbands+i*nbands,:] = window_function_calc(banddef,transdic,nskip=iskip,ellmin = win_minell,ellmax=win_maxell)


    output['windowfunc']=windowfunc

    
    return output
