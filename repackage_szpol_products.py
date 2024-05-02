from spectra_port import unbiased_multispec as spec
from spectra_port import utils
from spectra_port import covariance_functions as cov

import pickle as pkl
import numpy as np
import pdb






            

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

    dir='/home/creichardt/szpol_old/likelihood/data/spt_hiell_2020/'
    odir = '/home/creichardt/szpol_old/'
    bps0 = np.loadtxt(dir+'spt_hiell_2020.bp_file')
    nb=90
    bps = np.zeros(nb,dtype=np.float32) #going to padd 90x90 to make all 6 x 15.
    bps[:13]=bps0[:13,1]
    bps[15:]=bps0[13:,1]
    np.savetxt(odir+'bandpowers_szpol.txt',bps,fmt='%.5e')

    cov0 = np.fromfile(dir+'spt_hiell_2020_cov.bin',dtype=np.float64)
    #this should be 88x88
    cov = np.zeros([nb,nb],dtype=np.float64)
    cov[:13,:13] = cov0[:13,:13]
    cov[14,14]=1.
    cov[13,13]=1.
    cov[:13,15:] = cov0[:13,13:] 
    cov[15:,:13] = cov0[13:,:13]
    cov[15:,15:] = cov0[13:,13:] 
    cov.astype(np.float64).tofile(odir+'cov_oldszpol.bin')



    cov0 = np.fromfile(dir+'spt_hiell_2020_beamerr.bin',dtype=np.float64)
    #this should be 88x88
    cov = np.zeros([nb,nb],dtype=np.float64)
    cov[:13,:13] = cov0[:13,:13]
    cov[14,14]=0.
    cov[13,13]=0.
    cov[:13,15:] = cov0[:13,13:] 
    cov[15:,:13] = cov0[13:,:13]
    cov[15:,15:] = cov0[13:,13:] 
    cov.astype(np.float64).tofile(odir+'fractional_beam_cov_oldszpol.bin')

    spectrum = spec['spectrum']
    win = spec['windowfunc']


    lmin,lmax i*4 size

      delta=(efflmax-efflmin+1)*8_8
      offset=2 * 4_8+1
      do i=1,nall
         read(tmp_file_unit,pos=((i-1)*delta + offset)) locwin
         windows(j0:j1,i)=locwin(j0:j1)
      end do



    cov_out = np.reshape(cov_out,[nfcombo*nfb,nfcombo*nfb])
    #pdb.set_trace()
    print("Decide if I want to do any band rescaling to reduce dynamic range!!!")
    cc = cov_out[keep1d,:]
    print(cc.shape)
    ocov = cc[:,keep1d]
    print(ocov.shape)

    eval,evec = np.linalg.eig(ocov)
    print('evals <0: {} of {}'.format(np.sum(eval <= 0),ocov.shape[0]))
    covfile='/home/creichardt/highell_dls/cov_szpolbins.bin'
    print_cov(covfile,ocov)

    owin = win_out[keep1d,:]
    winfile = '/home/creichardt/highell_dls/windowfunc_szpolbins.bin'
    print_win(winfile, owin,    spec['win_minell'], spec['win_maxell'])
    l = np.arange(spec['win_minell'], spec['win_maxell']+1)
    nkept = np.sum(keep1d)
    leffs = np.zeros(nkept)
    for i in range(nkept):
        leffs[i] = np.sum(l * owin[i,:])
    sigmas = np.sqrt(np.diag(ocov))

    bpfile='/home/creichardt/highell_dls/dls_szpolbins.txt'
    print_bps(bpfile,leffs, spec_out,keep,sigmas)

    #assuming keep same in all... may need to change this
    nb = nkept//6
    sigmas2 = np.reshape(sigmas,[nb,6])
    leffs2 = np.reshape(leffs,[nb,6])
    lmins = final_bands[i0[0]:i1[0]]+1
    lmaxs = final_bands[i0[0]+1:i1[0]+1]

    bptexfile='/home/creichardt/highell_dls/table_dls_szpolbins.tex'
    print_bps_tex(bptexfile, lmins, lmaxs, leffs2, spec_out,keep,sigmas)
