


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
    inv_super_kernel[i,iskip:,i,iskip:] = np.inverse(

        #not done
##################
# 6: Multiply data bandpowers by Inverse Kernel
##################


##################
# 7: Apply inverse kernel to the covariance matrices
##################

##################
# 8: Combine covariances yield total covariance estimate
##################

##################
# 9: Optionally: Calculate bandpower window functions
##################



