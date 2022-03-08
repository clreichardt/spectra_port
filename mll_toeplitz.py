import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

"""
Taken from pymaster's example.
This script is used to calculate the MLL matrix with the Toeplitz approximation
from Namaster. It requires the apodization mask stored in a separate file apod_mask.npy.
---
It took 6 hours on 1 nersc node to calculate the matrix on 7th of March,2022.
"""
print("Setting up")

nside = 8192
ls = np.arange(3*nside)

print("Reading mask")
mask = np.load('apod_mask.npy')
# mask = np.ones(12*nside**2)
# 
print("Setting up workspaces and binning scheme")
mp_t = np.ones(12*nside**2)
f0 = nmt.NmtField(mask, [mp_t])
b = nmt.NmtBin.from_nside_linear(nside, 300)
leff = b.get_effective_ells()
print("Done setting up")

print("Calling C lib to compute coupling matrix with toeplitz approx")
wt = nmt.NmtWorkspace()
wt.compute_coupling_matrix(f0, f0, b, l_toeplitz=nside, l_exact=2500, dl_band=40)
c_tpltz = wt.get_coupling_matrix() 
cl_tpltz = wt.decouple_cell(nmt.compute_coupled_cell(f0, f0))

print("Saving the computed mll")
np.save('mll_tpltz.npy', c_tpltz)

