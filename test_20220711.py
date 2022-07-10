import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import glob
import healpy as hp

import pickle as pkl


dir = '/big_scratch/cr/mmdl/'

files = glob.glob(dir+'*0002.fits')
l = np.arange(10101)
dlfac = l*(l+1)/(2*np.pi)

for file in files:
    map = hp.read_map(file)
    cl = hp.anafast(map)
    dl = cl * dlfac
    with open(file+'_dl.pkl','wb') as fp:
        pkl.dump(dl,fp)

        
