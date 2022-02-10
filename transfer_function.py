import numpy as np



def initial_estimate(cl_mc, cl_theory, fsky_w2):
    """
    Calculates the initial estimates of the transfer function for a MASTER style pipeline
    Implements the definition of F0 above eq 18 in MASTER(astro-ph/0105302)
    This definition is :
    F0 = <C_l>_MC / (fsky * w2 * B_l^2 * C_th)
    
    Inputs:
    bl = beam function used in sims (N-vector)
    cl_theory = power spectrum used to create sims, can be Dl or Cl - must be same as cl_mc (N-vector) 
    cl_mc = power spectrum of processed sims, can be Dl or Cl - must be same as cl_theory (N-vector) 
    fsky_w2 = float (fsky_w2 factor from master for normalization of mask)
    Output: 
    Returns: initial estimate of transfer function (N-vector)
    """
    # The calculation is done in following steps:
    # 1. Calculate the denominator
    #   1.1. Multiply the theory Cls with the beam b_l**2 and set the negative values to zero
    #   1.2. Calculate the initial estimate F0
    #-----------------
    # 1.
    # 1.1
    cl_theory_beam2 = cl_theory * bl**2. # cl_theory_beam is just a moniker for Cl_theory with beam smoothing
    # 1.2
    F0 = cl_mc / ( fsky_w2 * cl_theory_beam2 )
    F0[np.where(cl_theory_beam2 < 0.)] = 1.
    return F0

def iteration( F0, M_ll, fsky_w2, bl, cl_theory, cl_mc):
    '''
    Calculates the iterative3 estimates of the transfer function for a MASTER style pipeline (astro-ph/0105302)
    
    NB: all of the N-vectors below must share the same underlying ell-binning. No checks are made that they are consistent. 
    Inputs:
    F0 = previous transfer function iteration (N-vector)
    M_ll = mode-coupling matrix (NxN array)
    bl = beam function used in sims (N-vector)
    cl_theory = power spectrum used to create sims, can be Dl or Cl - must be same as cl_mc (N-vector) 
    cl_mc = power spectrum of processed sims, can be Dl or Cl - must be same as cl_theory (N-vector)
    fsky_w2 = float (fsky_w2 factor from master for normalization of mask)
    Outputs: 
    Returns: next iteration of transfer function (N-vector)
    '''
    cl_theory_beam2 = cl_theory * bl**2
    F = F0 + (cl_mc -  M_ll * (F0 * cl_theory_beam2 ) ) / (cl_theory_beam2 * fsky_w2)
    F[np.where(cl_beam2 < 0.)] = 1.
    return F


def calculate_tf():
    # do  initial check

    # read and interpolate beams
    beams_interp = read_beams_file_and_interpolate( beams_file )
    
    # 
    F0 = initial_estimate()
    F = iteration(F0)
    return F

