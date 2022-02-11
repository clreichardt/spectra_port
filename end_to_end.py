


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



