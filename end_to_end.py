import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
from spectra_port import utils
from spectra_port import unbiased_multispec as spec
import time
import pickle as pkl
import pdb
    
def trim_end_to_end_output(fullsize):
    fullsize['data_spectrum'].allspectra=None
    fullsize['data_spectrum'].kmask=None
    fullsize['data_spectrum'].window=None
    fullsize['mc_spectrum'].allspectra=None
    fullsize['mc_spectrum'].kmask=None
    fullsize['mc_spectrum'].window=None
    fullsize['mc_spectrum_fine'].allspectra=None
    fullsize['mc_spectrum_fine'].kmask=None
    fullsize['mc_spectrum_fine'].window=None
    fullsize['mc_spectrum_fine'].cov=None
    fullsize['mc_spectrum_fine'].est2_cov=None
    fullsize['mc_spectrum_fine'].est1_cov=None
    fullsize['mask']=None
    fullsize['kmask']=None
    return fullsize
    
def end_to_end(mapfiles,
               mcmapfiles,
               banddef,
               beam_arr,
               theoryfiles,
               workdir,
               mcstub='/mc/',
               datastub='/data/',
               setdef=None,
               setdef_mc1=None,
               setdef_mc2=None,
               simbeam_arr=None,
               do_window_func = True,
               win_minell=10,
               win_maxell=13000,
               lmax=13000,
               nside=8192,
               kmask=None,
               kmask_on_the_fly_ranges = None, 
               kmask_on_the_fly = None,
               weighted_average_for_kmask= False, 
               mask=None,
               kernel_file=None,
               sim_kernel_file=None,
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
    print('end of lmax of {}'.format(lmax))
    output['nside']=nside
    output['resume']=resume
    output['mapfiles']=mapfiles
    output['mcmapfiles']=mcmapfiles
    output['banddef']=banddef
    output['theoryfiles']=theoryfiles
    output['beam_arr']=beam_arr
    output['simbeam_arr']=simbeam_arr
    output['kernel_file']=kernel_file
    output['sim_kernel_file']=sim_kernel_file
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
    
    if True:
        info, kernel = utils.load_mll(kernel_file)
        if sim_kernel_file is not None:
            siminfo, simkernel = utils.load_mll(sim_kernel_file)
        else:
            simkernel=kernel
            siminfo=info
        # dimensions --
        # kernel[ell_out, ell_in]
        #
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
    if True:
        newtime=time.time()
        print('run sim unbiased: last step took {:.0f}'.format(newtime-lasttime))
        lasttime=newtime
            
        mcdir = workdir + mcstub

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
                                                    kmask_on_the_fly_ranges = kmask_on_the_fly_ranges, 
                                                    kmask_on_the_fly = kmask_on_the_fly,
                                                    weighted_average_for_kmask=weighted_average_for_kmask,
                                                    cmbweighting=True)
            with open(mcdir+'mc_spectrum.pkl', 'wb') as f:
                pkl.dump(mc_spectrum,f)
                
        #return
            
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
                                                    kmask_on_the_fly_ranges = kmask_on_the_fly_ranges, 
                                                    kmask_on_the_fly = kmask_on_the_fly,
                                                    weighted_average_for_kmask=weighted_average_for_kmask,
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
    datadir = workdir + datastub
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
                                                # kmask=kmask,
                                                kmask_on_the_fly_ranges = kmask_on_the_fly_ranges, 
                                                kmask_on_the_fly = kmask_on_the_fly,
                                                weighted_average_for_kmask=weighted_average_for_kmask,
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

    beams_interp = utils.fill_in_beams(beam_arr,ellkern)
    beams = utils.explode_beams(beams_interp)
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
    
    #haven't checked stability of this with different kernels for sim and data. could be underdamped in the iterative solver
    transfer_iter = np.zeros([ntfs,niter,nkern])
    ii=0
    for i in range(nsets):        
        dl_mc = mc_spectrum_fine.spectrum[:,ii]
        ii += (nsets -i)
        transfer_iter[i,0,:] = utils.transfer_initial_estimate(dl_mc, theory_dls_interp[i,:], simbeams_interp[i,:], fskyw2)
        for j in range(1,niter):
            transfer_iter[i,j,:] = utils.transfer_iteration( transfer_iter[i,j-1,:], dl_mc, theory_dls_interp[i,:], simbeams_interp[i,:], fskyw2, simkernel)
    
    transfer = np.zeros([nspectra,nkern])
    k=0
    for i in range(nsets): 
        for j in range(i,nsets):
            transfer[k,:] = np.sqrt(transfer_iter[i,-1,:]*transfer_iter[j,-1,:])
            k+=1
    output['transfer_iter'] = transfer_iter
    output['transfer']=transfer

    with open(datadir+'tmp_posttransfer.pkl', 'wb') as f:
        pkl.dump(output,f)

            
    ##################
    # 5: Create binned kernel, including mode-coupling, beams, pixel WFs, transfer functions
    #     Optionally allow different beams in sims (and thus kernel)
    #    Invert the kernel
    ##################
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    print('binned kernels')
    nbands = banddef.shape[0]-1
    super_kernel        = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    sim_super_kernel    = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    inv_super_kernel    = np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    inv_sim_super_kernel= np.zeros([nspectra, nbands, nspectra, nbands],dtype=np.float64)
    iskips = np.zeros(nspectra, dtype=np.int32)
    eskips = np.zeros(nspectra, dtype=np.int32)
    #pdb.set_trace()
    for i in range(nspectra):
        print(i,kernel.shape)
        super_kernel[i,:,i,:]     = utils.rebin_coupling_matrix(kernel, ellkern, banddef, transferfunc=transfer[i,:], beamfunc = beams[i,:])
        sim_super_kernel[i,:,i,:] = utils.rebin_coupling_matrix(simkernel, ellkern, banddef, transferfunc=transfer[i,:], beamfunc = simbeams[i,:])
        slice_kern = np.asarray([super_kernel[i,j,i,j] for j in range(nbands)])
        peak = np.max(slice_kern)
        ipeak = np.argmax(slice_kern)
        slice = slice_kern < 1e-3 * peak

        #we potentially want to slice elements off the end and almost certainly the start. 
        #start first
        try:
            iskip = np.where(slice[:ipeak])[-1][-1] +1 #want to start 1 after
        except IndexError:
            iskip=0 #end up here if np.where returned empty array -- all false
        try:
            eskip = np.where(slice[ipeak:])[-1][0]+ipeak
        except IndexError:
            eskip=len(slice) #end up here if np.where returned empty array -- all false
              
        iskips[i]=iskip
        eskips[i]=eskip
        # leave first (or more) usually bogus bin out of inversion
        #don't try to divide by zero
        inv_super_kernel[i,iskip:eskip,i,iskip:eskip] = np.linalg.inv(super_kernel[i,iskip:eskip,i,iskip:eskip])
        inv_sim_super_kernel[i,iskip:eskip,i,iskip:eskip] = np.linalg.inv(sim_super_kernel[i,iskip:eskip,i,iskip:eskip])
        
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
    output['iskips']=iskips
    output['eskips']=eskips

    ##################
    # 6: Multiply data bandpowers by Inverse Kernel
    ##################
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    print('unbias spectra')
    print(invkernmat.shape,data_spectrum.spectrum.shape)
    spectrum = np.reshape(np.matmul(invkernmat, np.reshape(data_spectrum.spectrum.T,[nspectra*nbands])),[nspectra,nbands])

    output['spectrum']=spectrum
    ##################
    # 7: Apply inverse kernel to the covariance matrices
    ##################
    newtime=time.time()
    print('last step took {:.0f}'.format(newtime-lasttime))
    lasttime=newtime
    print('unbias cov')
    scov = np.reshape(np.transpose(np.reshape(mc_spectrum.cov,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
    sample_cov = np.reshape(np.matmul(np.matmul(invsimkernmat , scov) , invsimkernmattr),[nspectra,nbands,nspectra, nbands])
    dcov = np.reshape(np.transpose(np.reshape(data_spectrum.cov,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
    meas_cov = np.reshape(np.matmul(np.matmul(invkernmat , dcov), invkernmattr),[nspectra,nbands,nspectra, nbands])

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
        windowfunc = np.zeros([nbands*nspectra, nlwin], dtype=np.float32)
        for i in range(nspectra):
            iskip = iskips[i]
            eskip = eskips[i]
            nb0 = eskip-iskip+1
            transdic = {'ell':ellkern, 
                        'Mll':kernel,
                        'invkernel':np.squeeze(inv_super_kernel[i,iskip:eskip,i,iskip:eskip]),
                        'transfer':np.squeeze(transfer[i,:]),
                        'bl':np.squeeze(beams[i,:]),
                        'eskip':eskip,
                        'iskip':iskip
                        }

            windowfunc[iskip+i*nbands:eskip+i*nbands,:] = (utils.window_function_calc(banddef, transdic, 
                                                                                      ellmin = win_minell, ellmax=win_maxell))#[iskip:eskip,:]


        output['windowfunc']=windowfunc
        output['win_minell']=win_minell
        output['win_maxell']=win_maxell
    
    return output
