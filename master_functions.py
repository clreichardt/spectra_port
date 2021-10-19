def transfer_initial(ell,cl,cl_MC,bl,fskyw2)
    '''
    ;purpose: to calculate the initial estimate of the transfer function for a MASTER style pipeline
    ;  Implements definition of F0 above eqn 18 in MASTER(astro-ph/0105302)
    ; This definition is:
    ; F0 = <C>_MC / ( f_sky * w2 * B_ell^2 * C^th) 
    ;        - for all ell
      ; this code is intended to be run as part of a suite to calculate the true, converged transfer function

    ; Arguments:

    ; fskyw2 == fsky*second moment of mask; as defined in eqn 9 of MASTER

    ; these can all be binned (assumes binning is the same)
    ;ell = n vector of ell
    ; cl = n of theoretical  C_ell (or D_ell)
    ; bl = b_ell at each ell (including appropriate pl)
    ; Cmc = n-vector of MC mean C_ell (or D_ell)

    ;optional: delell (default 50) = ell smooth range

    ; Outputs:
    ; returns F_ell^0

    '''
    cl2=cl*bl^2
    fl = cl_MC/(fskyw2 *cl2)
    fl[cl2 <= 0]=1.

    return(fl)

def transfer_iterate(ell,cl,cl_MC,bl,fskyw2,fl_prev,kernel)
    '''
    ;purpose: to calculate the initial estimate of the transfer function for a MASTER style pipeline
    ;  Implements definition of F0 above eqn 18 in MASTER(astro-ph/0105302)
    ; This definition is:
    ; F0 = <C>_MC / ( f_sky * w2 * B_ell^2 * C^th) 
    ;        - for all ell
    ; this code is intended to be run as part of a suite to calculate the true, converged transfer function

    ; Arguments:

    ; fskyw2 == fsky*second moment of mask; as defined in eqn 9 of MASTER

    ;ell = n vector of ell
    ; cl = n of theoretical  C_ell (or D_ell)
    ; bl = b_ell at each ell (including pixel)
    ; Cmc = n-vector of MC mean C_ell (ro D_ell)

    ;fl_prev = n-vector = last guess at transfr function

    ; kernal = nxn array = mode-mixing kernel

    ;optional: delbin (default 1) = ell smooth range

    ; Outputs:
    ; returns F_ell^i+1
    '''
    cl2=cl*bl^2

    fl = f0 + (cl_MC - np.matmult(kernel, (fl_prev*cl2)))/(fskyw2 * cl2)
    fl[cl2 <= 0] = 1.

    return(fl)

