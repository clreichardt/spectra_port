import numpy as np



def initial_estimate(cl_MC, cl_theory, fsky_w2,  delta_ell=1,   ):
    """
    Calculates the initial estimates of the transfer function for a MASTER style pipeline
    Implements the definition of F0 above eq 18 in MASTER(astro-ph/0105302)
    This definition is :
    F0 = <C_l>_MC / (fsky * w2 * B_l^2 * C_th)
    """
    # The calculation is done in following steps:
    # 1. Calculate the numerator. Smooth it at some delta ell. Default is delta ell = 50
    # 2. Calculate the denominator
    #   2.1. Multiply the theory Cls with the beam b_l**2 and set the negative values to zero
    #   2.2. Calculate the initial estimate F0
    #-----------------
    # 1.
    smooth_cl_MC = smooth( cl_MC , delta_ell )
    # 2.
    # 2.1
    cl_theory_beam2 = cl_theory * bl**2. # cl_theory_beam is just a moniker for Cl_theory with beam smoothing
    neg_ind = np.where(cl_theory_beam2 < 0.)
    #cl_theory_beam2[neg_ind] = 1. # not setting it to 0. to avoid zero division error in the next step
    # 2.2
    F0 = smooth_cl_MC / ( fsky_w2 * cl_theory_beam2 )
    F0[neg_ind] = 1.
    return F0

def iteration( F0, M_ll, bl, fsky_w2, cl_theory, delta_ell ):
    cl_beam2 = cl_theory * bl**2
    F = F0 + (cl_MC -  M_ll * (F0 * cl_beam2 ) ) / (cl_beam2 * fsky_w2)
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

def read_beams_file_and_interpolate( beams_file ):
    # read the file
    beams = np.loadtxt( beams_file )
    beams_interp = np.interp( ell, beams[:,0], beams[:,1] )
    # interpolate the beams
    # return the interpolated beams
    return beams
