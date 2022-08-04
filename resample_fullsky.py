import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import glob
import healpy as hp
from spectra_port import utils
from spectra_port import unbiased_multispec as spec
import pickle as pkl



ind,map = spec.load_spt3g_healpix_ring_map('/sptlocal/user/pc/g3files_v2/combined_T_148ghz_00024.g3')
dir = '/big_scratch/cr/mmdl/'

#mask_file='/home/pc/hiell/mapcuts/apodization/apod_mask.npy'
#mask = np.load(mask_file)
print(ind.shape)
nside=8192
theta,phi = hp.pixelfunc.pix2ang(nside,ind)


thetamult = [-1,1,-1]
thetaoff = np.pi * np.asarray([1.,0,1.])
phioff = np.pi * np.asarray([0,1.,1.])
for i in range(3):
    newtheta = thetamult[i] * theta + thetaoff[i]
    newphi = phi + phioff[i]
    newphi[newphi > 2*np.phi] -= 2*np.pi
    newind = hp.ang2pix(nside,newtheta,newphi)
    print(newind.shape)
