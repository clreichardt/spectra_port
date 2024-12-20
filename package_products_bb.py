#from spectra_port import unbiased_multispec as spec
from spectra_port import utils
#from spectra_port import covariance_functions #as cov

import pickle as pkl
import numpy as np
import pdb


def print_bps_tex(bpfile, lmins, lmaxs, leffs, bps,keep,sigmas):
    nkeep = np.sum(keep)
    #ns = bps.shape[0]
    ns = lmins.shape[0]
    print('bps exists: ',bps.shape,np.sum(keep))

    #going to assume all have same number for simplicity
    #also assume 

    toprow = [0,3,5]
    botrow = [1,2,4]
    print(ns)
    fmtstr = "{:d} - {:d} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\\\\"
    fmtstr2 = "{:d} - {:d} & {:.1f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\"
    with open(bpfile,'w') as fp:
        #print("{:d} {:d}".format(ns, nkeep), file=fp)
        #2001 -  2200 &  2077   & 218.4 &    3.8   & 215.6 &    2.3   & 286.2 &    6.5   \\ 

        i=0
        bp0 = bps[toprow[i],keep[toprow[i],:]]
        i=1
        bp1 = bps[toprow[i],keep[toprow[i],:]]
        i=2
        bp2 = bps[toprow[i],keep[toprow[i],:]]

        for i in range(ns):
            if sigmas[i+ns*toprow[1]] >= 1:
                print(fmtstr.format(lmins[i],lmaxs[i],leffs[i,3], bp0[i],sigmas[i+ns*toprow[0]],
                                    bp1[i],sigmas[i+ns*toprow[1]],bp2[i],sigmas[i + ns*toprow[2]]))
            else:
                print(fmtstr2.format(lmins[i],lmaxs[i],leffs[i,3], bp0[i],sigmas[i+ns*toprow[0]],
                                    bp1[i],sigmas[i+ns*toprow[1]],bp2[i],sigmas[i + ns*toprow[2]]))
        i=0
        bp0 = bps[botrow[i],keep[botrow[i],:]]
        i=1
        bp1 = bps[botrow[i],keep[botrow[i],:]]
        i=2
        bp2 = bps[botrow[i],keep[botrow[i],:]]
        for i in range(ns):
            if sigmas[i+ns*botrow[1]] >= 1:
                print(fmtstr.format(lmins[i],lmaxs[i],leffs[i,3], bp0[i],sigmas[i+ns*botrow[0]],
                                    bp1[i],sigmas[i+ns*botrow[1]],bp2[i],sigmas[i + ns*botrow[2]]))
            else:
                print(fmtstr2.format(lmins[i],lmaxs[i],leffs[i,3], bp0[i],sigmas[i+ns*botrow[0]],
                                    bp1[i],sigmas[i+ns*botrow[1]],bp2[i],sigmas[i + ns*botrow[2]]))


def print_calcov(file, calcov):
    ns = calcov.shape[0]
    assert ns == 3
    with open(file,'w') as fp:
        #print("{:d} {:d}".format(ns, nkeep), file=fp)
        for i in range(ns):
            print("{:.5e}  {:.5e} {:.5e}".format(calcov[0,i],calcov[1,i],calcov[2,i]),file=fp)


def print_bps(bpfile, leffs, bps,keep,sigmas):
    nkeep = np.sum(keep)
    ns = bps.shape[0]
    print('bps exists: ',bps.shape,np.sum(keep))
    with open(bpfile,'w') as fp:
        #print("{:d} {:d}".format(ns, nkeep), file=fp)
        kk=0
        for i in range(ns):
            k=0
            loc = bps[i,keep[i,:]]

            for val in loc:

                print("{:d} {:f} {:.3e} {:.3e}".format(k,leffs[kk],val,sigmas[kk]),file=fp)
                k+=1
                kk+=1

def print_bps_jax(bpfile, bps,keep):
    nkeep = np.sum(keep)
    ns = bps.shape[0]
    print('bps exists: ',bps.shape,np.sum(keep))
    with open(bpfile,'w') as fp:
        #print("{:d} {:d}".format(ns, nkeep), file=fp)
        for i in range(ns):
            loc = bps[i,keep[i,:]]

            for val in loc:

                print("{:.3e}".format(val),file=fp)

            

def print_cov(covfile,cov):
    print('cov exists: ',cov.shape)
    with open(covfile,'wb') as fp:
        cov.astype('float64').tofile(fp)

def print_win(wfile,win,minl,maxl):
    print('win exists: ',win.shape)
    with open(wfile,'wb') as fp:
        fp.write("{:d} {:d}\n".format(minl,maxl).encode())
        #print("{} {}".format(minl,maxl).encode(),file=fp)

        win.astype('float32').tofile(fp)
        

if __name__ == '__main__':
    odir = 'FILL_THIS_IN_DIR'
    dlfile='FILL_THIS_IN.pkl'
    with open(dlfile,'rb') as fp:
        spec  = pkl.load(fp)

    #Hardwired numbers
    nb10 = 300 # number of dl=10 bins
    nfreq = 3 #3 freqs 90, 150, 220
    nfcombo = nfreq * (nfreq+1) // 2
    nbps = nfcombo * nb10
    
    win_minell=2 #minimum ell BPWFs are reported at -- CHECK
    win_maxell=3000 #maximum ell BPWFs are reported at -- CHECK  (can likely cut this down to a lower lmax)
    '''
    assuming These values:
    spec['spectrum'] = nb10 x nfcombo array of bandpowers (uncalibrated) -- be easy to make these calibrated by commenting out lines below applying calibration
    These are still at dl=10 from Namaster
    Fastest varying index should be the ellbins ie the first N are 90x90, the next N are 90x150, etc.
    Could straightforwardly change code to read in 1d array -- only uses this in 2d when applying calibration.
    
    spec['windowfunc'] = array Nbps x Nlwin. These are the windowfunctions from Namaster. 

    spec['covariance'] = conditioned and calibrated covariance for dl=10. This should be an Nbps x Npbs array
    '''
    
    #Could read this in. Currently hardwired
    orig_bands = np.arange(2,3002,10) # edit as appropriate to Namaster bins
    #bin i is orig_bands[i] <= ell < orig_bands[i+1]
    final_bands = np.asarray([2,32,102, 202, 302, 402, 502,3002])
    

    spectrum = spec['spectrum']
    win = spec['windowfunc']


    print("Do Calibration first!!!")

    #calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    #change to below when reran with latest PWFs/Tfs on 2023 Sep 08
    
    #sep 2023 w v2 beams
    #calibration_factors = np.asarray([ (0.9017)**-0.5, (0.9833)**-0.5, (0.9703)**-0.5 ])
    
    #Aug 2024
    calibration_factors = np.asarray([1.145, 1.130, 1.228])

    #calibration_factors *= 1e-3  #correction for units between sims and real data. The transfer function brings it over.  This ends up being brought to the 2 power so 1e-6 effectively.
    global_freq_index_array = np.zeros([nfcombo,2],dtype=np.int32)
    cals = np.ones(nfcombo)


    k=0
    for i in range(nfreq):
        for j in range(i,nfreq):
            cals[k] = calibration_factors[i] * calibration_factors[j]
            spectrum[k,:] *= cals[k]
            k+=1
    #again, unhardcode this
    spectrum = np.reshape(spectrum,nbps)

    cov = spec['covariance'] 
    

    #pdb.set_trace()

    #taking weights as inverse diagonal of cov:
    wts = 1./np.diag(cov)
    #pdb.set_trace()
    wts[np.diag(cov)==0]=0 #cov was 0 at crazy points. ignore these bins
    #pdb.set_trace()
    #Force it to be positive definite. May not need this.
    eval,evec = np.linalg.eig(cov)
    cov = np.matmul(np.matmul(evec,np.diag(np.abs(eval))),evec.T)



    nfb = len(final_bands)-1

    i0 = np.ones(nfcombo,dtype=np.int32)*1
    i1 = np.ones(nfcombo,dtype=np.int32)*6 #502
    #i1[0] = 28 #9000 for 90x90
    #i1[1:3] = 29 #10000 for 90x150,90x220 
    spec_out,cov_out,win_out,transform = utils.weighted_rebin_spectrum(orig_bands,final_bands,spectrum,cov0=cov, win0=win,weights = wts)

    with open(odir+'bptransform.npy','wb') as fp:
        transform.tofile(fp)


    spec_out = np.reshape(spec_out,[nfcombo,nfb])

    keep = np.zeros(spec_out.shape,dtype=bool)
    for i in range(nfcombo):
        keep[i,i0[i]:i1[i]] = True
    keep1d = keep.flatten()
    #pdb.set_trace()
    #ospec = (spec_out.flatten())[keep1d]



    cov_out = np.reshape(cov_out,[nfcombo*nfb,nfcombo*nfb])

    #pdb.set_trace()

    print("Decide if I want to do any band rescaling to reduce dynamic range!!!")
    cc = cov_out[keep1d,:]
    print(cc.shape)
    ocov = cc[:,keep1d]
    print(ocov.shape)
    print(np.diag(ocov)[1:])
    
    eval,evec = np.linalg.eig(ocov)
    print('evals <0: {} of {}'.format(np.sum(eval <= 0),ocov.shape[0]))
    print(eval[eval<0])
    print(eval[:10])
    #pdb.set_trace()
    covfile=odir+'cov.bin'

    owin = win_out[keep1d,:]
    print('bpwf shape:',owin.shape)
    winfile = odir+'windowfunc.bin'
    print_win(winfile, owin,    win_minell, win_maxell)
    l = np.arange(win_minell, win_maxell+1)
    nkept = np.sum(keep1d)
    leffs = np.zeros(nkept)
    for i in range(nkept):
        leffs[i] = np.sum(l * owin[i,:])
    sigmas = np.sqrt(np.diag(ocov))

    bpfile=odir+'dls.txt'
    print_bps(bpfile,leffs, spec_out,keep,sigmas)
    bpfile=odir+'dls_jax.txt'
    print_bps_jax(bpfile, spec_out,keep)
    #assuming keep same in all... may need to change this
    nb = nkept//6
    sigmas2 = np.reshape(sigmas,[nb,6])
    leffs2 = np.reshape(leffs,[nb,6])
    lmins = final_bands[i0[0]:i1[0]]+1
    lmaxs = final_bands[i0[0]+1:i1[0]+1]

    bptexfile=odir+'table_dls.tex'
    print_bps_tex(bptexfile, lmins, lmaxs, leffs2, spec_out,keep,sigmas)
    

