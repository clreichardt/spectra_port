import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import glob
import sys
sys.path.append('/home/creichardt/spt3g_software/build')
import healpy as hp

import unbiased_multispec as spec
import utils
import end_to_end
from spt3g import core
#from spt3g import core,maps, calibration
import argparse
import pickle as pkl
import pdb
import time

if __name__ == "__main__":
    dir='/sptgrid/analysis/highell_TT_19-20/v4/mockobs/v4.3_mask_0p4medwt_6mJy150ghzv2/'
    files = glob.glob(dir+'inputsky*')
    nf = len(files)
    usedinds = np.zeros(nf,dtype=np.int32)
    for i in range(nf):
        usedinds[i] = int(files[i].split('inputsky')[1])


    maskfile='/sptlocal/user/creichardt/hiell2022/mask_0p4medwt_6mJy150ghzv2.pkl'
    
    with open(maskfile,'rb') as fp:
        mask  = pkl.load(fp)
    
    mdir='/sptlocal/user/creichardt/out_maps/'
    cldir = '/sptlocal/user/creichardt/cls/'
    avgcl=0
    for ind in usedinds:
        savfile = cldir+'cl_95ghz_{}.pkl'.format(ind)
        try:
            with open(savfile,'rb') as fp:
                cl = pkl.load(fp)
        except:
            fname = mdir + 'sim_95ghz_{}.g3'.format(ind)
            mm2 = list(core.G3File(fname))
            ind2,tt2=mm2[0]['T'].nonzero_pixels()
            tt2 -= np.mean(tt2)
            masked = mask.copy()
            masked[ind2] *= tt2
            cl = hp.anafast(masked,lmax = 15000,iter=2,pol=False)
            with open(savfile,'wb') as fp:
                pkl.dump(cl,fp)
        avgcl += cl
    avgcl /= len(usedinds) 
    savfile = cldir+'cl_95ghz.pkl'
    with open(savfile,'wb') as fp:
        pkl.dump(avgcl,fp)
