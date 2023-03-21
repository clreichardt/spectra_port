import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import sys, os
from scipy.optimize import minimize
import scipy
import warnings

from pyparsing import null_debug_action
# sys.modules['unbiased_multispec'] = None
# sys.modules['spectra_port'] = None
os.chdir('/home/pc/hiell')
sys.path.append('/home/pc/hiell')



#----------------------------------------------------------------
# The covmat conditioning has two main bits: diagonals and off-diagonals
#
# Off diagonal:
# Off diagonal elements are easy bits. Just take the average of the a diagonals
# next to the main diagonal (i.e offset diagonal : cov[i, i+offset] ) and the
# only knob here is how far goes (ie offset) one go before setting all the rest of the 
# elements to zero. 
#
# Diagonals:
# Diagonal elements are to be estimated from the noisy estimates. This involves
# fititng with beam function and smoothing and all the other involved intricacies.
#
# The code is written in the same manner: the diag and off diag bits are broken 
# into two separate functions 
#----------------------------------------------------------------
# print(plt.style.available)
# exit()
plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn-talk')



def AverageOffDiags(cov, delta_min, delta_max, spec1, spec2, lmin=50, lmax=200, plot=False):
    """
    Loop through from delta_min to delta_max and take the off-diagonals.
    Average all the off-diagonals and assign these values to the off diagonal elements
    """
    # convert cov to corr
    corr = CorrelationFromCovariance(cov)
    corr2 = np.copy(corr)[lmin:lmax, lmin:lmax] 
    # i = np.arange(delta_min, delta_max) # for diagonal indices used in loop
    n = corr2.shape[0]
    offsets = []
    avgs = []
    # set all the offdiags above delta_max to zero
    corr2[np.triu_indices(n, k=delta_max)] = 0.0
    # set all the offdiags below delta_max to zero
    corr2[np.tril_indices(n, k=-delta_max)] = 0.0
    # set the diagonal to zero 
    corr2[np.diag_indices_from(corr2)] = 0.0
    
    for offset in range(delta_min, delta_max):
        for sign in [1,-1]:
            off_diag_avg = np.diagonal(corr2, sign*offset).mean()
            diag_ind = np.where(np.eye(n, k=sign*offset) == 1) # create an eye matrix with 1s at offset diag and get its inds
            corr2[ diag_ind ] = off_diag_avg
            
            
            
            offsets.append(sign*offset)
            avgs.append(off_diag_avg)
            
    offsets.append(0)
    avgs.append(0)
    # Remove the Poisson component from the averaged offdiags
    # avgs[:-1] = avgs[:-1] - np.mean(avgs[:-1])
     
    # Embed the cutout corr mat into a bigger covmat of same size as original           
    corr3 = np.zeros_like(corr)
    corr3[lmin:lmax, lmin:lmax] = corr2
    
    sort_ind = np.argsort(offsets)
    offsets = np.asarray(offsets)[sort_ind]
    avgs = np.asarray(avgs)[sort_ind]
    
    if plot: # plot the unmodified cov
        print(f'Plotting {spec1}_{spec2}')
        fig = plt.figure()
        gs = fig.add_gridspec(3,4)
        ax1 = fig.add_subplot(gs[:2,:2], aspect='equal')
        ax2 = fig.add_subplot(gs[:2,2:], aspect='equal')
        ax3 = fig.add_subplot(gs[2,0:4])
        gs.update(wspace=1.0, hspace=0.5)
        
        ax1.set_title('Unmodified corr')
        im = ax1.imshow(corr, origin='lower')
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2.set_title('Diagonally averaged corr')
        im2 = ax2.imshow(corr3, origin='lower', vmin=-0.03, vmax=0.03)
        cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        
        ax3.plot(offsets, avgs)
        ax3.set_xlabel('Offset from the diagonal')
        ax3.set_ylabel('Mean of the off-diagonal')
        ax3.axhline(y=0, linestyle='--', linewidth=1.0, color='k')
        ax3.axvline(x=0, linestyle='--', linewidth=1.0, color='k')
        
        plt.savefig(f'/home/pc/hiell/cov_conditioning/plots/offdiag/{spec1}_{spec2}.png')
        plt.clf()
        plt.close()
    return corr3

def CorrelationFromCovariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def ConditionOffDiagonals( covmat, delta_min=1, delta_max=50 ):
    """
    Get cov - a subset of covmat block corresponding to an index eg (90x90)x(90x90)
    pass this block to off-diagonal averaging function 
    """
    specs = ['90x90', '90x150', '90x220', '150x150', '150x220', '220x220' ]
    cov_mean = np.zeros((259,259))
    for i1 in range(6):
        for i2 in range(6):
            spec1 = specs[i1]
            spec2 = specs[i2]
            if i1 == i2:
                cov = covmat[i1, :, i2, :]
                cov = AverageOffDiags(cov, delta_min, delta_max, spec1, spec2, plot=True)
                cov_mean += cov
            # covmat[i1, :, i2, :] = cov
    # cov_mean /= 36
    
    plt.imshow(cov_mean, origin='lower', vmin=-0.003, vmax=0.003)
    plt.colorbar()
    plt.savefig('/home/pc/hiell/cov_conditioning/plots/offdiag/mean.png')
    plt.clf()
    
    return cov_mean

# ----------------------------------------------------------------
# Diagonal conditioning functions
# ----------------------------------------------------------------

def EigenvalueDecomp(data_cov, sig_cov, title):
    """
    Decompose the signal cov and get the eigenvector. 
    Then transform the new diagonal with those eigenvectors
    and see which modes are noisy
    """
    # for key in S:
    #     sig_diag = S[key]
    # do the eigenvalue decomposition
    # Symmetrize the covs
    data_cov = (data_cov + data_cov.T)/2
    sig_cov = (sig_cov + sig_cov.T)/2
    
    assert (data_cov == data_cov.T).all()
    assert (sig_cov == sig_cov.T).all()
    
    
    sig_eig_val, sig_eig_vec =  np.linalg.eig(sig_cov)
    sig_eig_vec_inv = np.linalg.inv(sig_eig_vec)
    d_eig_val = np.matmul(np.matmul(sig_eig_vec_inv, data_cov) , sig_eig_vec)
    
    plt.title(title)
    plt.plot(sig_eig_val, label='Signal')
    plt.plot(d_eig_val, label='Data')
    # plt.plot(d_eig_val/sig_eig_val, label='Noise/signal')
    plt.savefig(f'/home/pc/hiell/cov_conditioning/plots/eig_vals/{title}.png')
    plt.clf()
    return


def GetCompiledCov(S, N, i, j, h, k):
    """Gives the reformed diagonal given a the Cl and noise power estimates

    Args:
        S (dict): Dictionary for signal Cls
        N (dict): Dictionary for noise power spectra
        i (str): Index for the Cov
        j (str): Index for the Cov
        h (str): Index for the Cov
        k (str): Index for the Cov

    Raises:
        ValueError: Unknown index

    Returns:
        array: The reformed diagonal for ijhk block
    """
    
    # redefine the indices so that we can eliminate some redundant cases
    # 1. Internal swap as power spectra are symmetric: ji -> ij
    if int(i) > int(j):
        i , j = j , i
    if int(h) > int(k):
        h , k = k , h
    # 2. Covariance should be symmetric, power spectra commute: ij x hk = hk x ij 
    if (h == k and i != j):
        i , j , h , k = h , k , i, j
    # 3. Some special cases
    if (k == i and h != i):
        # for ijhi case always rearrange ijih
        h , k = k , h 
        
    # print(f'{i} {j} {h} {k}')
    
    if (i == j == h == k ):
        # iiii - 2 common map case
        return 2*S[f'{i}{i}']**2 + 4*S[f'{i}{i}']*N[f'{i}{i}'] + 2*N[f'{i}{i}']**2
    elif (i == h and j == k and i != j):
        # ijij - 2 common map case
        return S[f'{i}{i}']*S[f'{j}{j}'] + S[f'{i}{j}']**2 + (S[f'{i}{i}']*N[f'{j}{j}'] + S[f'{j}{j}']*N[f'{i}{i}'] ) +  (N[f'{i}{i}'] * N[f'{j}{j}'])
    elif (i == j == h and k!=i ):
        # iiik
        return 2*S[f'{i}{i}']*S[f'{i}{k}'] + 2*S[f'{i}{k}']*N[f'{i}{i}']
    elif (i != j != k and i == h ):
        # ijik - 1 common map
        return S[f'{i}{i}']*S[f'{j}{k}'] + S[f'{i}{k}']*S[f'{i}{j}'] + S[f'{j}{k}']*N[f'{i}{i}']
    elif (i != j != k and j == h ):
        # ijjk - same as above - 1 map common but different index
        return S[f'{j}{j}']*S[f'{i}{k}'] + S[f'{j}{k}']*S[f'{j}{i}'] + S[f'{i}{k}']*N[f'{j}{j}']
    elif (i != j != h and j == k ):
        # ijhj - same as above - 1 map common but different index
        return S[f'{j}{j}']*S[f'{i}{h}'] + S[f'{j}{h}']*S[f'{j}{i}'] + S[f'{i}{h}']*N[f'{j}{j}']
    elif (i != j != h and i == k ):
        # ijhi - same as above - 1 map common but different index
        return S[f'{i}{i}']*S[f'{j}{h}'] + S[f'{i}{h}']*S[f'{j}{i}'] + S[f'{j}{h}']*N[f'{i}{i}']
    elif (i == j and h == k and i != h): 
        # iihh - no map common case
        return 2*S[f'{i}{h}']**2
    elif (i==j and h != k) or (i != j and h==k):
        #iihk no map common, auto-cross case
        if (int(i) < int(h) and int(i) < int(k) ):
            return 2*( S[f'{i}{h}']*S[f'{i}{k}'] + S[f'{i}{h}']*N[f'{i}{k}'] + S[f'{i}{k}']*N[f'{i}{h}']) #+ N[f'{i}{h}']*N[f'{i}{k}'])
        elif (int(i) < int(h) and int(i) > int(k) ):
            return 2*( S[f'{i}{h}']*S[f'{k}{i}'] + S[f'{i}{h}']*N[f'{k}{i}'] + S[f'{k}{i}']*N[f'{i}{h}']) #+ N[f'{i}{h}']*N[f'{i}{k}'])
        elif (int(i) > int(h) and int(i) < int(k) ):
            return 2*( S[f'{h}{i}']*S[f'{i}{k}'] + S[f'{h}{i}']*N[f'{i}{k}'] + S[f'{i}{k}']*N[f'{h}{i}']) #+ N[f'{i}{h}']*N[f'{i}{k}'])
        else:
            return 2*( S[f'{h}{i}']*S[f'{k}{i}'] + S[f'{h}{i}']*N[f'{k}{i}'] + S[f'{k}{i}']*N[f'{h}{i}']) #+ N[f'{i}{h}']*N[f'{i}{k}'])
    else:
        # print(f'Index Unknown. You supplied -{i}-,-{j}-,-{h}-,-{k}-')
        raise ValueError(f'Index Unknown. You supplied -{i}-,-{j}-,-{h}-,-{k}-')
        # return np.zeros(259) 
    
    # if (i == j == h == k ):
    #     # iiii - 2 common map case
    #     return 2*S[f'{i}{i}']**2 + 4*S[f'{i}{i}']*N[f'{i}{i}'] + 2*N[f'{i}{i}']**2
    # elif (i == h and j == k and i != j) or (i == k and j == h and i != j):
    #     # ijij or ijji - 2 common map case
    #     return S[f'{i}{i}']*S[f'{j}{j}'] + S[f'{i}{j}']**2 + (S[f'{i}{i}']*N[f'{j}{j}'] + S[f'{j}{j}']*N[f'{i}{i}'] ) +  (N[f'{i}{i}'] * N[f'{j}{j}'])
    # elif (i == j == h and k!=i ):
    #     # iiik
    #     return 2*S[f'{i}{i}']*S[f'{i}{k}'] + 2*S[f'{i}{k}']*N[f'{i}{i}']
    # elif (i == j == k and h!=i ):
    #     # iihi - auto-cross, same as above but different index
    #     return 2*S[f'{i}{i}']*S[f'{i}{h}'] + 2*S[f'{i}{h}']*N[f'{i}{i}']
    # elif (i != j and j== k == h):
    #     # ijjj - auto-cross, same as above but different index
    #     return 2*S[f'{j}{j}']*S[f'{j}{i}'] + 2*S[f'{j}{i}']*N[f'{j}{j}']
    # elif (i != j != k and i == h ):
    #     # ijik - 1 common map
    #     return S[f'{i}{i}']*S[f'{j}{k}'] + S[f'{i}{k}']*S[f'{i}{j}'] + S[f'{j}{k}']*N[f'{i}{i}']
    # elif (i != j != k and j == h ):
    #     # ijjk - same as above - 1 map common but different index
    #     return S[f'{j}{j}']*S[f'{i}{k}'] + S[f'{j}{k}']*S[f'{j}{i}'] + S[f'{i}{k}']*N[f'{j}{j}']
    # elif (i != j != h and j == k ):
    #     # ijhj - same as above - 1 map common but different index
    #     return S[f'{j}{j}']*S[f'{i}{h}'] + S[f'{j}{h}']*S[f'{j}{i}'] + S[f'{i}{h}']*N[f'{j}{j}']
    # elif (i != j != h and i == k ):
    #     # ijhi - same as above - 1 map common but different index
    #     return S[f'{i}{i}']*S[f'{j}{h}'] + S[f'{i}{h}']*S[f'{j}{i}'] + S[f'{j}{h}']*N[f'{i}{i}']
    # elif (i == j and h == k and i != h): 
    #     # iihh - no map common case
    #     return 2*S[f'{i}{h}']**2
    # elif (i==j and h != k) or (i != j and h==k):
    #     #iihk no map common, auto-cross case
    #     return 2*( S[f'{i}{h}']*S[f'{i}{k}'] + S[f'{i}{h}']*N[f'{i}{k}'] + S[f'{i}{k}']*N[f'{i}{h}']) #+ N[f'{i}{h}']*N[f'{i}{k}'])
    # elif (i==j and h != k) or (i != j and h==k):
    #     # ijkk - no map common, auto-cross case
    #     return 2*( S[f'{k}{i}']*S[f'{k}{j}'] + S[f'{k}{i}']*N[f'{k}{j}'] + S[f'{k}{j}']*N[f'{k}{i}'] + N[f'{k}{i}']*N[f'{k}{j}'])
    # else: # iihk, ijkk
    #     # print(f'Index Unknown. You supplied -{i}-,-{j}-,-{h}-,-{k}-')
    #     raise ValueError(f'Index Unknown. You supplied -{i}-,-{j}-,-{h}-,-{k}-')
    #     # return np.zeros(259) 
    
def SmoothDiag(diag, smooth_pts=15):
    # box = np.ones(box_pts)/box_pts
    # diag_smooth = np.convolve(diag, box, mode='same')
    diag_smooth = scipy.signal.savgol_filter(diag, smooth_pts, 2)
    return diag_smooth

def CompileCov(cov, sim_cov, plot):
    """Takes the data and sim covs and finds the Noise power terms from the data cov
    

    Args:
        cov (array): The full data covariance matrix
        sim_cov (array): The full sim covariance matrix
        lmax (int, optional): _description_. Defaults to 13000.
    """
    # the following expressions are taken from Das et al. : https://arxiv.org/pdf/1301.1037.pdf
    # eq A8 - A15

    
    # Load the signal Cls from the sim cov to get the signal only part
    freq = ['90','150', '220']
    S = {}
    for i in range(3):
        for j in range(3):
            # print(f'{freq[i]}{freq[j]}')
            S[f'{freq[i]}{freq[j]}'] = np.sqrt( sim_cov[i,:,j,:].diagonal() / 2)
            
    # Smooth the signal
    S_smooth = {}
    for S_key in S:
            S_smooth[S_key] = SmoothDiag(S[S_key], 15)
            
    if plot:
        print('Making Signal plots')
        l = np.arange(0,13000, 50)[:-1] + 25.0
        for S_key in S:
            plt.title(f'Signal {S_key}')
            # plt.plot( np.fft.fftfreq(259), np.abs(np.fft.fft(S[f'{freq[i]}{freq[j]}'])) )
            plt.semilogy(l,S[S_key], 'o', markersize=2.0, label='Sims')
            if S_smooth[S_key] is not None:
                plt.semilogy( l, S_smooth[S_key] , label='Sav-Gol smooth') 
            plt.legend()
            plt.savefig(f'/home/pc/hiell/cov_conditioning/plots/sig/signal_terms_{S_key}.png')
            plt.clf()
    
     
    
    # Solved for N_ii from the Cov_iiii expression
    N = {} # dict for noise spectra
    print('Estimating Noise terms')
    for i, cov_ind in enumerate([0,3,5]):
        ind = f'{freq[i]}{freq[i]}' 
        # print('Term:',ind)
        N[ind]   = np.sqrt( cov[cov_ind, :, cov_ind, :].diagonal() / 2 ) - S[ind]
        
    # Estimate the cross noise terms from the C_iijj kind of terms
    cov_ind_list = [0,3,5]
    for i in range(3):
        for j  in range(i+1,3):
            ind = f'{freq[i]}{freq[j]}'
            # print('Term:',ind)
            N[ind] = cov[cov_ind_list[i], :, cov_ind_list[j], :].diagonal() - S[ind]
           
        
    # N_ij = 0 # this may not be true for spt3g but I will refine it later
    #
    # Smooth the Signal and noise terms
    # for S_key in S:
    #     S[S_key] = SmoothDiag(S[S_key], 35)
    # for N_key in N:
    #     N[N_key] = SmoothDiag(N[N_key], 10)   
    
    # fit a theoretical template for the noise_terms
    fit_noise = {}
    smooth_est = {}
    for N_key in N:
        if N_key in ['9090','150150','220220']:
            fit_noise[N_key], smooth_est[N_key] = FitTheoreticalTemplate(N_key, N[N_key], 50, 258, 10, 60, 1e7)
        elif N_key =='90150':
            fit_noise[N_key], smooth_est[N_key] = FitTheoreticalTemplate(N_key, N[N_key], 60, 150, 10, 50, 1e15)
        elif N_key =='90220':
            fit_noise[N_key], smooth_est[N_key] = FitTheoreticalTemplate(N_key, N[N_key], 50, 150, 10, 50, 1e13)
        elif N_key =='150220':
            fit_noise[N_key], smooth_est[N_key] = FitTheoreticalTemplate(N_key, N[N_key], 50, 150, 10, 50, 1e13)
            
            
            
        
    if plot:
        print('Making N_l plots')
        # l = np.arange(0,13000, 50)[:-1] + 25.0
        for N_key in N:
            plt.title(f'Noise {N_key}')
            plt.plot( l, N[N_key], 'o', markersize=2.5, label='Measured noise' )
            # plt.plot( 1e8*l**-0.7, label='test' )
            
            # plt.plot( SmoothDiag(N[N_key][:-2], 10), label='Sav-Gol smooth' )
            if fit_noise[N_key]is not None:
                plt.axvspan(l[50],l[258], alpha=0.3, color='gray', label='beam-fit area')
                plt.axvspan(l[10],l[50], alpha=0.1, color='red', label='1/f-fit area')
                
                plt.plot( l, fit_noise[N_key], '--',  label='template fit')
                plt.plot( l, smooth_est[N_key], linewidth=1.5,  label='final estimate')
                
            plt.xlabel('$\mathcal{\ell}$')
            plt.legend(loc='best')
            
            if N_key == '90150':
                plt.ylim(-1e12,1e12)
            elif N_key == '90220':
                plt.ylim(-1e12,1e12)
            elif N_key =='150220':
                plt.ylim(-1e12,1e12)
            elif N_key == '9090':
                plt.ylim(0,1e7)
            elif N_key == '150150':
                plt.ylim(0,1e7)
            elif N_key == '220220':
                plt.ylim(0,1e7)
            else:
                pass
                
            # plt.plot( np.fft.fftfreq(259), np.abs(np.fft.fft(N[N_key])) )
            plt.xlabel('$\mathcal{\ell}$')
            plt.savefig(f'/home/pc/hiell/cov_conditioning/plots/nls/noise_terms_{N_key}.png')
            plt.clf()
            
            plt.title(f'Residual noise {N_key}')
            r1 = ( N[N_key][2:-2] - smooth_est[N_key][2:-2] ) 
            r2 = (N[N_key][2:-2] - fit_noise[N_key][2:-2]) 

            chi1 = np.nansum(r1**2, where=(r1 != np.inf) )
            chi2 = (r2**2).sum()
            
            plt.plot( l[2:-2], r1, 'o', markersize=2.0, label=f'smooth estimate, chisq={chi1:.2e}' )
            plt.plot( l[2:-2], r2, 'o', markersize=2.0, alpha=0.5, label=f'template estimate, chisq={chi2:.2e}' )
            plt.xlabel('$\mathcal{\ell}$')
            plt.legend()
            
            if N_key in ['9090', '150150', '220220']:
                plt.ylim(-1e6, 1e6)
            else:
                plt.ylim(-1e13,1e13)
            
            plt.xlabel('$\mathcal{\ell}$')
            plt.savefig(f'/home/pc/hiell/cov_conditioning/plots/nls/residuals{N_key}.png')
            plt.clf()
            
            # l= np.arange(0,13000,50)[:-1]/5000
            plt.title(f'Scaled noise {N_key}')
            plt.plot( l, N[N_key]/fit_noise[N_key], 'o', markersize=2.0, label='Measured')
            # plt.plot( (N[N_key]/fit_noise[N_key]),  label='measured / template-fit')
            # plt.plot( l[8:], SmoothDiag( N[N_key]/fit_noise[N_key], 15)[8:], label='Sav-Gol Smoothed')
            plt.xlabel('$\mathcal{\ell}$')
            plt.legend()
            
            # plt.semilogy( N[N_key]/fit_noise[N_key] )
            
                
            # plt.ylim(0,1e2)
            # plt.plot( l* N[N_key]/fit_noise[N_key] , label='l')
            # plt.plot( np.sqrt(l)* N[N_key]/fit_noise[N_key], label='sqrt' )
            # plt.legend()
            

            plt.savefig(f'/home/pc/hiell/cov_conditioning/plots/nls/scaled_noise{N_key}.png')
            plt.clf()
            
    # package it all up into a covmat and return back 
    reformed_cov = np.zeros(cov.shape)
    for i1 in range(6):
        for i2 in range(6):
            specs = ['90x90', '90x150', '90x220', '150x150', '150x220', '220x220' ]
            ind1 = specs[i1]
            ind2 = specs[i2]
            i, j =  ind1.split('x')[0], ind1.split('x')[1]
            h, k =  ind2.split('x')[0], ind2.split('x')[1]
            
            diag = GetCompiledCov(S_smooth, fit_noise, i=i, j=j, k=k, h=h)
            # diag = GetCompiledCov(S, N, i=i, j=j, k=k, h=h)
            np.fill_diagonal( reformed_cov[i1,:,i2,:], diag )
            if plot:
                # print(f'Making reformed diagonal plot for {ind1}{ind2}')
                plt.title(f'({specs[i1]})x({specs[i2]})')
                try:
                    plt.semilogy( l, diag , 'o', markersize=2.0, label='points' )
                    plt.semilogy( l, diag, label='interp' )
                    # plt.plot( diag , 'o', markersize=2.0, label='points' )
                    # plt.plot( diag, label='interp' )
                except:
                    print(f'Making linear scale plot for {specs[i1]}{specs[i2]}')
                    plt.plot( diag )
                plt.xlabel('$\mathcal{\ell}$')
                plt.savefig(f'/home/pc/hiell/cov_conditioning/plots/reformed_diag/reformed_diag_{specs[i1]}{specs[i2]}.png')
                plt.clf()
                
    return reformed_cov  


def bin_template(bl):
    l = np.arange(0,13000)
    bl = bl[:13000]
    bl_binned,_,_ = scipy.stats.binned_statistic(l, bl, bins=259)
    return np.asarray(bl_binned)
    

def NoiseTemplate(freq1, freq2, length):
    beam_file = '/home/pc/spt3g_software/beams/products/compiled_2020_beams.txt'
    if freq1 =='90':
        B_l1 = np.loadtxt(beam_file)[:,1]
    elif freq1 =='150':
        B_l1 = np.loadtxt(beam_file)[:,2]
    elif freq1 =='220':
        B_l1 = np.loadtxt(beam_file)[:,3]
    else:
        raise ValueError('Unknown frequency input for beams')
    
    if freq2:
        if freq2 =='90':
            B_l2 = np.loadtxt(beam_file)[:,1]
        elif freq2 =='150':
            B_l2 = np.loadtxt(beam_file)[:,2]
        elif freq2 =='220':
            B_l2 = np.loadtxt(beam_file)[:,3]
        else:
            raise ValueError('Unknown frequency input for beams')
        
    B_l1 = bin_template(B_l1)
    B_l2 = bin_template(B_l2)
    
    assert( len(B_l1) == length)
    assert( len(B_l2) == length)
    
    l = np.arange(0,13000,50)[:-1] + 25
    
    if freq2 is None:
        template = 1./B_l1**2
    else:
        template = 1./(B_l1 * B_l2)
        
    return (l*(l+1)/(2*np.pi)) * template  * (1/np.sqrt(l))
   
def Chisq(x, template, data, nstart, nstop, fstart,fstop, amp):
    l = np.arange(0,13000,50)[:-1] + 25.0
    sig = np.zeros_like(l)
    sig[nstart:nstop] = x[0]*template[nstart:nstop]
    sig[fstart:fstop] += amp*(l[fstart:fstop]/x[2])**x[1]
    sig[:fstart] = data[:fstart] # should be minimum of fstart and nstart 
    chisq = np.sum( (data - sig )**2 ) # beam
    return chisq

def FitTheoreticalTemplate(freqs, data, nstart, nstop, fstart, fstop, amp):
    """Fit the theoretical template of the diagonal to the data
    Get the theoretical diagonal template and define a chisq for it.
    Minimize the chisq to get the best-fit
    Args:
        data (numpy array): The diagonal of the data covmat
    """
    if (len(freqs) == 4) or (len(freqs)==5):
        freq1 , freq2 = freqs[:2], freqs[2:]
    else:
        freq1, freq2 = freqs[:3], freqs[3:]
        
    l = np.arange(0,13000, 50)[:-1] +25.
        
    template = NoiseTemplate(freq1, freq2, len(data))
    
    fit = np.zeros_like(data)
    if freqs in ['9090', '150150', '220220']:
        res = minimize( Chisq, x0 = [10.0, -1, 500], args=(template, data, nstart, nstop, fstart, fstop, amp), bounds=( (0,1e3), (-2.5,-0.1), (300,1000) ) )
        fit += res.x[0] * template
        fit  += amp*(l/res.x[2])**res.x[1]
    else:
        res = minimize( Chisq, x0 = [0, -1, 500], args=(template, data, nstart, nstop, fstart, fstop, amp), bounds=( (0,1e3), (-8.,0.), (300,2000) ) )
        fit  += amp*(l/res.x[2])**res.x[1]
    
    # go the the template scaled space, smooth it there and return back to unscaled space    
    scaled_est = data / fit
    scaled_est[:8] = 1.0 
    scaled_est[258:] = 1.0
    smooth_est = SmoothDiag(scaled_est, 35)
    smooth_unscaled = smooth_est * fit
    smooth_unscaled[:8] = 0.0
    
    # Set the off diagonal blocks to 0 after the 100th bin (l=5000)
    if freqs not in ['9090', '150150', '220220']:
        smooth_unscaled[60:] = 0.
    
    if res.success:
        print(f'{freqs:<6} : {res.x[0]:8.2f} {res.x[1]:8.2f} {res.x[2]:8.2f},   Chisq = {res.fun:.2e}')
    else:
        print(f'Warning: Optimization result did not converge for {freq1}{freq2}')
        fit = data
    
    return fit, smooth_unscaled



def ConditionDiagonal( covmat, sim_cov, plot=True ):
    
    reformed_cov = CompileCov(covmat, sim_cov, plot=True)
    # for i in range(6):
    #     for j in range(6):
    #         EigenvalueDecomp(reformed_cov[i,:,j,:], sim_cov[i,:,j,:], f'{i}{j}')
            
    return reformed_cov


def plot_unconditioned_cov( covmat ):
    specs = ['90x90', '90x150', '90x220', '150x150', '150x220', '220x220' ]
    l = np.arange(0,13000,50)[:-1]
    for i in range(6):
        for j in range(6):
            plt.title(f'Unconditioned Diagonal {specs[i]} {specs[j]}')
            plt.semilogy( l, covmat[i,:,j,:].diagonal(), label=f'{i}{j}' )
            plt.xlabel('Multipole l')
            plt.savefig(f'/home/pc/hiell/cov_conditioning/plots/uncond_diag/unocnd_diag_{specs[i]}{specs[j]}.png')
            plt.clf()
              
            
    
    #Plot the diagonals
    return

if __name__ == '__main__':
    #load the covmat
    covfile = '/big_scratch/pc/xspec_2022/v5/nokweight/spectrum_small.pkl'
    with open( covfile , 'rb') as f:
            data = pkl.load(f)
    covmat = data['cov']
    
    invsimkernmat = data['invsimkernmat']
    invsimkernmattr = np.transpose(invsimkernmat)
    
    # Convert the sample cov from pseudo Cl space to the onsky space
    simcovfile = '/big_scratch/pc/xspec_2022/v5/nokweight/mc/mc_spectrum.pkl'
    with open( simcovfile , 'rb') as f:
            mc_spectrum = pkl.load(f)
    nbands = 259
    nspectra = 6
    scov = np.reshape(np.transpose(np.reshape(mc_spectrum.cov,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
    # scov = np.reshape(np.transpose(np.reshape(mc_spectrum.est1_cov,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
    # scov = np.reshape(np.transpose(np.reshape(mc_spectrum.est2_cov,[nbands,nspectra,nbands,nspectra]),[1,0,3,2]),[nbands*nspectra,nbands*nspectra])
    
    sim_cov = np.reshape(np.matmul(np.matmul(invsimkernmat , scov) , invsimkernmattr),[nspectra,nbands,nspectra, nbands])

    
    # sim_cov = 1e24*sim_cov.reshape((259,6,259,6))


    # convdition the diagonal 
    # covmat = ConditionDiagonal(covmat, sim_cov, plot=True)

    # condition the off-diagonal
    # covmat = ConditionOffDiagonals( covmat )
    # covmat = ConditionOffDiagonals( sim_cov )

    plot_unconditioned_cov(covmat)    

    # Save the conditioned covmat
