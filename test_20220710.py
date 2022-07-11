import os
os.environ['OMP_NUM_THREADS'] = "8"
import glob
import sys
sys.path.append('/home/creichardt/spt3g_software/build')
#sys.path.insert(1,'/home/creichardt/.local/lib/python3.7')
import numpy as np
import healpy as hp

from spt3g import core,maps, calibration

import pickle as pkl

import matplotlib.pyplot as plt
import time
import pdb

from spectra_port import unbiased_multispec
sys.modules['unbiased_multispec'] = unbiased_multispec



maskf='/home/pc/hiell/mapcuts/apodization/apod_mask_v4.npy'
mask = np.load(maskf)

i9,o9  =unbiased_multispec.load_spt3g_healpix_ring_map('/sptgrid/analysis/highell_TT_19-20/v4/mockobs/v2.0_testinputsv2/inputsky002/bundles/bundle182_150GHz.g3.gz')
#pdb.set_trace()

l=np.arange(10151)
dlfac = l*(l+1)/(2*np.pi)
for i in range(100):
    premap = '/sptgrid/analysis/highell_TT_19-20/mmdl_inputskies_v2/90GHz/combined_T_090ghz_{:05d}.g3'.format(i)

    a9 = list(core.G3File(premap))
    tmapin = np.asarray(a9[0]['T'])
    mn = np.average(tmapin[i9])
    map = (tmapin-mn) * mask
    cl = hp.sphtfunc.anafast(map,lmax=10150)
    dl = cl * dlfac
    with open('test90_{}.pkl'.format(i),'wb') as fp:
        pkl.dump(dl,fp)

    premap = '/sptgrid/analysis/highell_TT_19-20/mmdl_inputskies_v2/150GHz/combined_T_148ghz_{:05d}.g3'.format(i)

    a9 = list(core.G3File(premap))
    tmapin = np.asarray(a9[0]['T'])
    mn = np.average(tmapin[i9])
    map = (tmapin-mn) * mask
    cl = hp.sphtfunc.anafast(map,lmax=10150)
    dl = cl * dlfac
    with open('test150_{}.pkl'.format(i),'wb') as fp:
        pkl.dump(dl,fp)

    premap = '/sptgrid/analysis/highell_TT_19-20/mmdl_inputskies_v2/220GHz/combined_T_219ghz_{:05d}.g3'.format(i)

    a9 = list(core.G3File(premap))
    tmapin = np.asarray(a9[0]['T'])
    mn = np.average(tmapin[i9])
    map = (tmapin-mn) * mask
    cl = hp.sphtfunc.anafast(map,lmax=10150)
    dl = cl * dlfac
    with open('test220_{}.pkl'.format(i),'wb') as fp:
        pkl.dump(dl,fp)

#mn = np.average(o9)
#map = mask
#map[i9] *= (o9-mn)
#cl2 = hp.sphtfunc.anafast(map,lmax=10500)
#with open('test2.pkl','wb') as fp:
#    pkl.dump(cl2,fp)
