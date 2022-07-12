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

for file1 in files:
    print(file1)
    tmap = hp.read_map(file1)
    cl = hp.anafast(tmap,lmax=10100)
    dl = cl * dlfac
    file2=file1+'_dl.pkl'
    print('dumping to '+file2)
    with open(file2,'wb') as fp:
        pkl.dump(dl,fp)

        
