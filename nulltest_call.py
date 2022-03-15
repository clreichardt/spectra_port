import numpy as np
import unbiased_multispec as spec
import os
import pickle

banddef = np.arange(0,1000+1,500)
nbundles = 200
setdef = np.zeros([int(nbundles/2), 2] , dtype=np.int32)
setdef[:,0] = np.arange(100)
setdef[:,1] = 100 + np.arange(100)
window = np.load('/home/pc/hiell/mapcuts/apodization/apod_mask.npy')

jack90 = spec.unbiased_multispec(
                                    mapfile = None,
                                    window = window,
                                    banddef = banddef,
                                    nside = 8192,
                                    lmax=13000,
                                    cmbweighting = True,
                                    jackknife = True,
                                    persistdir = '/big_scratch/pc/nulls/',
                                    setdef = setdef,
                                    basedir = '/big_scratch/pc/nulls/'
                                    )
with open('/big_scratch/pc/nulls/results90.bin', 'wb') as f:
    pickle.dump(jack90, f)
