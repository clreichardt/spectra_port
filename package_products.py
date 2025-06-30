from spectra_port import unbiased_multispec as spec
from spectra_port import utils
from spectra_port import covariance_functions #as cov

import pickle as pkl
import numpy as np
import pdb
import os

import argparse
NORMAL=True
SZPOL=False

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-szpol', action='store_true',dest='szpol')
my_parser.add_argument('-onesimpwf', action='store_true',dest='onesimpwf')
my_parser.add_argument('-nosimpwf', action='store_true',dest='nosimpwf')

args = my_parser.parse_args()

SZPOL=args.szpol
NOSIMPWF=args.nosimpwf
ONESIMPWF=args.onesimpwf
ANYTRUE = False
for key in args.__dict__.keys():
    ANYTRUE = ANYTRUE or  args.__dict__[key]
    print(key, args.__dict__[key])
NORMAL = not SZPOL
print('normal',NORMAL)




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
                print(fmtstr.format(lmins[i],lmaxs[i],leffs[3,i], bp0[i],sigmas[i+ns*toprow[0]],
                                    bp1[i],sigmas[i+ns*toprow[1]],bp2[i],sigmas[i + ns*toprow[2]]),file=fp)
            else:
                print(fmtstr2.format(lmins[i],lmaxs[i],leffs[3,i], bp0[i],sigmas[i+ns*toprow[0]],
                                     bp1[i],sigmas[i+ns*toprow[1]],bp2[i],sigmas[i + ns*toprow[2]]),file=fp)
        i=0
        bp0 = bps[botrow[i],keep[botrow[i],:]]
        i=1
        bp1 = bps[botrow[i],keep[botrow[i],:]]
        i=2
        bp2 = bps[botrow[i],keep[botrow[i],:]]
        for i in range(ns):
            if sigmas[i+ns*botrow[1]] >= 1:
                print(fmtstr.format(lmins[i],lmaxs[i],leffs[3,i], bp0[i],sigmas[i+ns*botrow[0]],
                                    bp1[i],sigmas[i+ns*botrow[1]],bp2[i],sigmas[i + ns*botrow[2]]),file=fp)
            else:
                print(fmtstr2.format(lmins[i],lmaxs[i],leffs[3,i], bp0[i],sigmas[i+ns*botrow[0]],
                                     bp1[i],sigmas[i+ns*botrow[1]],bp2[i],sigmas[i + ns*botrow[2]]),file=fp)


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

    nfreq = 3
    nfcombo = nfreq * (nfreq+1) // 2

    dlfile='/big_scratch/cr/xspec_2022/spectrum_blv3rc4_small.pkl'
    covfile='/big_scratch/cr/xspec_2022/covariance_blv3rc4_extra1.1.pkl'
    odir='/home/creichardt/highell_dls_blv3rc4_fieldpwf_extra1p1/'

    final_bands = np.asarray([0,500,1000,1200,1400,1600,
                        1700,1800,1900,2000,2100,
                        2200,2300,2400,2500,2700,
                        3000,3300,3600,4000,4400,
                        4800,5300,5800,6400,7000,
                        7600,8300,9000,10000,11000,
                        12000])
    i0 = np.ones(nfcombo,dtype=np.int32)*6
    i1 = np.ones(nfcombo,dtype=np.int32)*30 #11000
    explodeis=i1+1

    calcov=np.zeros([3,3],dtype=np.float32)
    ''' Jan2024 cal file       
    SV90150 = .0040**2
    SV220x = SV90150
    calcov[0,0] = .0043**2
    calcov[0,1] = calcov[1,0] = SV90150
    calcov[1,1] = .0043**2
    calcov[:,2] = SV220x
    calcov[2,:] = SV220x
    calcov[2,2] = .0073**2
    '''
    #sep24 cal file
    SV90150 = .00546**2
    SV220x = SV90150
    calcov[0,0] = .00550**2
    calcov[0,1] = calcov[1,0] = SV90150
    calcov[1,1] = .00550**2
    calcov[:,2] = SV220x
    calcov[2,:] = SV220x
    calcov[2,2] = .00873**2

    if ONESIMPWF:
        print("using onesimpwf data products, binning, and calibration")
        
        dlfile='/big_scratch/cr/xspec_2022/spectrum_blv3rc4_1simpwf_small.pkl'
        covfile='/big_scratch/cr/xspec_2022/covariance_blv3rc4_1simpwf.pkl'
        odir='/home/creichardt/highell_dls_blv3rc4_fieldpwf_1simpwf/'

    if NOSIMPWF:
        print("using nosimpwf data products, binning, and calibration")
        
        dlfile='/big_scratch/cr/xspec_2022/spectrum_blv3rc4_nosimpwf_small.pkl'
        covfile='/big_scratch/cr/xspec_2022/covariance_blv3rc4_nosimpwf.pkl'
        odir='/home/creichardt/highell_dls_blv3rc4_fieldpwf_nosimpwf/'

    if SZPOL:
        print('outputting to SZPOL bins')

        odir='/home/creichardt/highell_dls_blv3rc4_fieldpwf_szpol/'

        print("using normal data products and calibration, and SZPOL binning")
        
        final_bands = np.asarray([0,500,1000,1850, 2000, 2200, 2500,2800,3100,3500,
                                  3900,4400,4900,5500,6200,7000,7800,8800,9800,11000,
                            12000])
        i0 = np.ones(nfcombo,dtype=np.int32)*3
        i1 = np.ones(nfcombo,dtype=np.int32)*18 #11000
        explodeis=i1+1
        explodeis[0] = 16

    

    try:
        os.mkdir(odir)
    except:
        pass

        
    calcovfile=odir+'calcov.txt'
    print_calcov(calcovfile, calcov)
    calcovfile=odir+'calcovx4.txt'
    print_calcov(calcovfile, calcov*4.)
    calcovfile=odir+'calcovx100.txt'
    print_calcov(calcovfile, calcov*100.)


    with open(dlfile,'rb') as fp:
        spec  = pkl.load(fp)


    spectrum = spec['spectrum']
    win = spec['windowfunc']


    

    #calibration_factors = np.asarray([ (0.9087)**-0.5, (0.9909)**-0.5, (0.9744)**-0.5 ])
    #change to below when reran with latest PWFs/Tfs on 2023 Sep 08
    
    #sep 2023 w v2 beams
    #calibration_factors = np.asarray([ (0.9017)**-0.5, (0.9833)**-0.5, (0.9703)**-0.5 ])
    
    #Jan 2024 w v3 beams
    #calibration_factors = np.asarray([ (0.8831)**-0.5, (0.9728)**-0.5, (0.9691)**-0.5 ])
    #Jun 2024 w v3beta6 beams
    #calibration_factors = np.asarray([ (0.8888)**-0.5, (0.9797)**-0.5, (0.9755)**-0.5 ])
    #Jun 2024: w v3beta7 beams and field PWF
    #calibration_factors = np.asarray([ (0.8880)**-0.5, (0.9789)**-0.5, (0.97505)**-0.5 ])
    #Sep 2024: Changed ell-range for tilt in Aylro et al. + RC4 beams
    calibration_factors = np.asarray([ (0.88546)**-0.5, (0.97518)**-0.5, (0.95894)**-0.5 ])
    calibration_factors *= 1e-3  #correction for units between sims and real data. The transfer function brings it over.  This ends up being brought to the 2 power so 1e-6 effectively.
    global_freq_index_array = np.zeros([nfcombo,2],dtype=np.int32)
    cals = np.ones(nfcombo)


    k=0
    for i in range(nfreq):
        for j in range(i,nfreq):
            cals[k] = calibration_factors[i] * calibration_factors[j]
            spectrum[k,:] *= cals[k]
            k+=1
    #again, unhardcode this
    spectrum = np.reshape(spectrum,nfcombo*259)



    with open(covfile,'rb') as fp:
        covobj  = pkl.load(fp)
     
    #should remove this hardcoding
    cov = np.reshape(covobj.cov,[nfcombo*259,nfcombo*259])
    #pdb.set_trace()  #This seems to be the wrong order...
    '''
    Spectrum is 259 90x90, followed by 259 90x150, etc.
    Not getting this right in cov space...
    '''

    #pdb.set_trace()

    #taking weights as inverse diagonal of cov:
    wts = 1./np.diag(cov)
    #pdb.set_trace()
    wts[np.diag(cov)==0]=0 #cov was 0 at crazy points. ignore these bins
    #pdb.set_trace()
    eval,evec = np.linalg.eig(cov)
    cov = np.matmul(np.matmul(evec,np.diag(np.abs(eval))),evec.T)

    print("Warning! need to input true cal uncertainty")
    t_cal_unc = np.ones(nfreq)* 0.01  

    orig_bands = spec['banddef']

    nfb = len(final_bands)-1


    #i1[0] = 28 #9000 for 90x90
    #i1[1:3] = 29 #10000 for 90x150,90x220 
    spec_out,cov_out,win_out,transform = utils.weighted_rebin_spectrum(orig_bands,final_bands,spectrum,cov0=cov, win0=win,weights = wts)




    with open(odir+'bptransform.npy','wb') as fp:
        transform.tofile(fp)

    print(np.diag(cov_out[5,:,5,:]))
    spec_out = np.reshape(spec_out,[nfcombo,nfb])

    keep = np.zeros(spec_out.shape,dtype=bool)
    for i in range(nfcombo):
        keep[i,i0[i]:i1[i]] = True
    keep1d = keep.flatten()
    #pdb.set_trace()
    #ospec = (spec_out.flatten())[keep1d]
    nkept = np.sum(keep1d)


    cov_out = np.reshape(cov_out,[nfcombo*nfb,nfcombo*nfb])

    #pdb.set_trace()

    print("Decide if I want to do any band rescaling to reduce dynamic range!!!")
    cc = cov_out[keep1d,:]
    print(cc.shape)
    ocov = cc[:,keep1d]
    print(ocov.shape)
    print(np.diag(ocov)[1:])
    maxdd = np.max(np.diag(ocov))
    explodeamount = 10000*maxdd #100xlargest diagonal sigma, or 10^4 in cov space
    explode_diag = np.zeros(nkept)
    running=0
    for i in range(nfcombo):
        thisn = i1[i]-i0[i]
        explodeany = i1[i]>explodeis[i]
        #print(i,explodeany)
        if explodeany:
            #print('Exploding:',running,explodeis[i],i1[i],i0[i],explodeamount)
            explode_diag[running:running+explodeis[i]-i0[i]]=0
            explode_diag[running+explodeis[i]-i0[i]:running+i1[i]-i0[i]]=explodeamount
            #print(explode_diag[running:running+thisn])
        else:
            explode_diag[running:running+thisn]=0
        running+=thisn
    print('Exploded this many:',np.sum(explode_diag > 0))
    ocov += np.diag(explode_diag)
    
    eval,evec = np.linalg.eig(ocov)
    print('evals <0: {} of {}'.format(np.sum(eval <= 0),ocov.shape[0]))
    print(eval[eval<0])
    print(eval[-5:])
    print(eval[:10])
    #pdb.set_trace()

    covfile=odir+'cov.bin'

    print_cov(covfile,ocov)
    bcovfile=odir+'beamcov_zero.bin'
    print_cov(bcovfile,ocov*0)
    


    owin = win_out[keep1d,:]
    print('bpwf shape:',owin.shape)
    winfile = odir+'windowfunc.bin'
    print_win(winfile, owin,    spec['win_minell'], spec['win_maxell'])
    l = np.arange(spec['win_minell'], spec['win_maxell']+1)

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
    #sigmas2 = np.reshape(sigmas,[nb,6])
    leffs2 = np.reshape(leffs,[6,nb])
    lmins = final_bands[i0[0]:i1[0]]+1
    lmaxs = final_bands[i0[0]+1:i1[0]+1]
    #pdb.set_trace()
    bptexfile=odir+'table_dls.tex'
    print_bps_tex(bptexfile, lmins, lmaxs, leffs2, spec_out,keep,sigmas)
    

