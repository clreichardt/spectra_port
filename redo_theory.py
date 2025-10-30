'''
This was actually run in TestTfHiell.ipynb

copied here for reference

'''

import os
os.environ['OMP_NUM_THREADS'] = "8"
import glob
import sys
sys.path.append('/home/creichardt/spt3g_software/build')
sys.path.append('/home/creichardt/.local/lib/python3.10')
#sys.path.insert(1,'/home/creichardt/.local/lib/python3.7')
import numpy as np
import healpy as hp

#from spt3g import core,maps, calibration

import pickle as pkl

import matplotlib.pyplot as plt
import time
import scipy

from scipy.ndimage import gaussian_filter1d
from spectra_port import unbiased_multispec, utils, covariance_functions
sys.modules['unbiased_multispec'] = unbiased_multispec
sys.modules['covariance_functions'] = covariance_functions

from spt3g import core,maps, calibration

import pickle as pkl

import matplotlib.pyplot as plt
import time
import scipy

dlfile='/big_scratch/cr/xspec_2022/mc_raw/mc_spectrum_fine.pkl'
with open(dlfile,'rb') as fp:
    mcspec  = pkl.load(fp)

with open('/sptlocal/user/creichardt/hiell2022/mask_0p4medwt.pkl','rb') as fp:
          mask = pkl.load(fp)

wt = np.mean(mask**2) * 1e-6
print(wt)

ellkern = utils.band_centers(mcspec.banddef)

theoryfiles = ['/sptlocal/user/creichardt/hiell2022/sim_dls_90ghz.txt',
               '/sptlocal/user/creichardt/hiell2022/sim_dls_150ghz.txt',
              '/sptlocal/user/creichardt/hiell2022/sim_dls_220ghz.txt']
beam_arr = np.loadtxt('/home/creichardt/spt3g_software/beams/products/compiled_2020_beams.txt')
theory_dls_interp = utils.fill_in_theory(theoryfiles,ellkern)
beams_interp = utils.fill_in_beams(beam_arr,ellkern)
beams = utils.explode_beams(beams_interp)
cmbfile = ''


ratio90 = mcspec.spectrum[:,0]/ (theory_dls_interp[0,:]* beams_interp[0,:]**2)/wt
ratio150 = mcspec.spectrum[:,2]/ (theory_dls_interp[1,:]* beams_interp[1,:]**2)/wt


nl = ellkern.shape[0]
nlp = nl+2
modratio90 = np.ones(nlp)
modratio150 = np.ones(nlp)
modells = np.ones(nlp)

modratio90[:nl] = gaussian_filter1d(ratio90,5,mode='nearest')
modratio90[:785]=1.0
#last two already 1

modratio150[:nl] = gaussian_filter1d(ratio150,5,mode='nearest')

modratio150[:785]=1.0
#last two already 1

modells[:nl]=ellkern
modells[-2]=13500
modells[-1]=15001

oell = np.arange(0,15001)
o90 = np.interp(oell,modells,modratio90)
o150 = np.interp(oell,modells,modratio150)
        

theoryfiles = ['/sptlocal/user/creichardt/hiell2022/sim_dls_90ghz.txt',
               '/sptlocal/user/creichardt/hiell2022/sim_dls_150ghz.txt',
              '/sptlocal/user/creichardt/hiell2022/sim_dls_220ghz.txt']
otheoryfiles = ['/sptlocal/user/creichardt/hiell2022/sim_fieldv2_dls_90ghz.txt',
               '/sptlocal/user/creichardt/hiell2022/sim_fieldv2_dls_150ghz.txt',
               '/sptlocal/user/creichardt/hiell2022/sim_fieldv2_dls_220ghz.txt']

dl90 = np.loadtxt(theoryfiles[0])
dl150 = np.loadtxt(theoryfiles[1])
dl220 = np.loadtxt(theoryfiles[2])

#these are the same length as oell abouve

ll = dl90[:,0]
dl90[:,1] *= o90
dl150[:,1] *= o150

np.savetxt(otheoryfiles[0],dl90,fmt = ['%.0f', '%.8e'])
np.savetxt(otheoryfiles[1],dl150,fmt = ['%.0f', '%.8e'])
np.savetxt(otheoryfiles[2],dl220,fmt = ['%.0f', '%.8e']) #unchanged
np.savez('/sptlocal/user/creichardt/hiell2022/tf_scaling_modes.npz',ratio150=modratio150,ratio90=modratio90)
