from spectra_port import unbiased_multispec as spec
from spectra_port import utils
from spectra_port import covariance_functions as cov

import pickle as pkl
import numpy as np


def print_bps(bpfile, bps,keep):
    nkeep = np.sum(keep)
    ns = bps.shape[0]

    with open(bpfile,'w') as fp:
        print("{} {}".format(ns, nkeep), file=fp)
        for i in range(ns):
            k=0
            loc = bps[i,keep[i,:]]
            for val in loc:
                print("{i} {e}".format(k,val),file=fp)
                k+=1
            

def print_cov(covfile,cov):
    with open(covfile,'wb') as fp:
        cov.astype('float64').tofile(fp)

def print_win(wfile,win,minl,maxl):
    with open(wfile,'wb') as fp:
        print(file=fp,"{} {}".format(minl,maxl))
        win.astype('float32').tofile(fp)
    

if __name__ == '__main__':
    dlfile='/big_scratch/cr/xspec_2022/spectrum_small.pkl'
    with open(dlfile,'rb') as fp:
        spec  = pkl.load(fp)


    spectrum = spec['spectrum']
    win = spec['windowfunc']


    print("Do Calibration first!!!")
    calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    calibration_factors *= 1e-3  #correction for units between sims and real data. The transfer function brings it over.  This ends up being brought to the 2 power so 1e-6 effectively.
    global_freq_index_array = np.zeros([6,2],dtype=np.int32)
    cals = np.ones(6)


    k=0
    for i in range(3):
        for j in range(i,3):
            cals[k] = calibration_factors[i] * calibration_factors[j]
            spectrum[k,:] *= cals[k]
            k+=1



    covfile='/big_scratch/cr/xspec_2022/covariance.pkl'
    with open(covfile,'rb') as fp:
        cov  = pkl.load(fp)

    #taking weights as inverse diagonal of cov:
    wts = 1./np.diag(cov)
    

    print("Warning! need to input true cal uncertainty")
    t_cal_unc = no.ones(3)* 0.01  

    orig_bands = spec.banddef
    final_bands = np.asarray([0,500,1000,1200,1400,1600,
                              1700,1800,1900,2000,2100,
                              2200,2300,2400,2500,2700,
                              3000,3300,3600,4000,4400,
                              4800,5300,5800,6400,7000,
                              7600,8300,9000,10000,11000,
                              12000])

    i0 = np.ones(6)*6
    i1 = np.ones(6)*30 #11000
    i1[0] = 28 #9000 for 90x90
    i1[1:3] = 29 #10000 for 90x150,90x220 
    spec_out,cov_out,win_out,transform = utils.weighted_rebin_spectrum(orig_bands,final_bands,spectrum,cov0=cov, win0=win,weights = wts)

    keep = np.zeros(spec_out.shape,dtype=bool)
    for i in range(6):
        keep[i0[i]:i1[i]] = True
    keep1d = keep.flatten()

    #ospec = (spec_out.flatten())[keep1d]
    bpfile='dls.txt'
    print_bps(bpfile,spec_out,keep)




    print("Decide if I want to do any band rescaling to reduce dynamic range!!!")
    cc = cov_out[keep1d,:]
    ocov = cc[:,keep1d]
    covfile='cov.bin'
    print_cov(covfile,ocov)

    owin = win_out[keep1d,:]
    winfile = 'windowfunc.bin'
    print_win(winfile, owin,    spec['win_minell'], spec['win_maxell'])

