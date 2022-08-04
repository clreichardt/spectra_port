import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import glob
import healpy as hp
from spectra_port import utils
from spectra_port import unbiased_multispec as spec
import pickle as pkl


def get_indices():
    ind,tmap = spec.load_spt3g_healpix_ring_map('/sptlocal/user/pc/g3files_v2/combined_T_148ghz_00024.g3')
    nside=8192

    theta,phi = hp.pixelfunc.pix2ang(nside,ind)


    thetamult = [-1,1,-1]
    thetaoff = np.pi * np.asarray([1.,0,1.])
    phioff = np.pi * np.asarray([0,1.,1.])
    i=0
    newtheta = thetamult[i] * theta + thetaoff[i]
    newphi = phi + phioff[i]
    newphi[newphi > 2*np.pi] -= 2*np.pi
    ind1 = hp.ang2pix(nside,newtheta,newphi)
    i=1
    newtheta = thetamult[i] * theta + thetaoff[i]
    newphi = phi + phioff[i]
    newphi[newphi > 2*np.pi] -= 2*np.pi
    ind2 = hp.ang2pix(nside,newtheta,newphi)
    i=2
    newtheta = thetamult[i] * theta + thetaoff[i]
    newphi = phi + phioff[i]
    newphi[newphi > 2*np.pi] -= 2*np.pi
    ind3 = hp.ang2pix(nside,newtheta,newphi)

    return ind, ind1, ind2, ind3



#fullmap = hp.read_map('/sptlocal/user/creichardt/out_maps/xfer/sim_lmax15000_nside8192_interp3.0_method1_1_seed5929_fgseed001627_radioseed012517_150ghz_map.fits')
#mask_file='/home/pc/hiell/mapcuts/apodization/apod_mask.npy'
#mask = np.load(mask_file)
#print(ind.shape)
#

# Convert numpy arrays from nersc to g3 healpix format and save to g3
def save_as_g3(cutsky, ind, file_name,nside=8192):
    print(file_name)
    #
    #cutsky *= core.G3Units.uK
    cutsky = cutsky.astype(dtype=np.float64)
    store = ind, cutsky, nside
    m = maps.HealpixSkyMap(store)
    
    #create a g3 frame to store the date
    frame = core.G3Frame(core.G3FrameType.Map)
    m.ringparse = True
    frame['T'] = m
    writer = core.G3Writer(file_name)
    writer(frame)
    return


def loop_and_cut(file_list,ostub='/sptlocal/user/creichardt/out_maps/sim_150ghz_{}.g3'):

    nf = len(file_list)
    ind0,ind1,ind2,ind3 = get_indices()
    i=0
    for j in range(nf):
        ifile=file_list[j]
        print('reading: ',ifile)
        fullmap = hp.read_map(ifile)
        cutsky = fullmap[ind0]
        file_name=ostub.format(i)
        save_as_g3(cutsky, ind0, file_name)
        i=i+1
        cutsky = fullmap[ind1]
        file_name=ostub.format(i)
        save_as_g3(cutsky, ind0, file_name)
        i=i+1
        cutsky = fullmap[ind2]
        file_name=ostub.format(i)
        save_as_g3(cutsky, ind0, file_name)
        i=i+1
        cutsky = fullmap[ind3]
        file_name=ostub.format(i)
        save_as_g3(cutsky, ind0, file_name)
        i=i+1


def do_all():
    stub='sim*150ghz_map.fits'
    dir='/sptlocal/user/creichardt/out_maps/xfer/'
    file_list = glob.glob(dir+stub)
    print(file_list)
    loop_and_cut(file_list,ostub='/sptlocal/user/creichardt/out_maps/sim_150ghz_{}.g3')
    file_list2=[file.replace('150ghz_map.fits','220ghz_map.fits') for file in file_list]
    print(file_list2)
    loop_and_cut(file_list2,ostub='/sptlocal/user/creichardt/out_maps/sim_220ghz_{}.g3')

    file_list3=[file.replace('150ghz_map.fits','95ghz_map.fits') for file in file_list]
    print(file_list3)
    loop_and_cut(file_list3,ostub='/sptlocal/user/creichardt/out_maps/sim_95ghz_{}.g3')
    
if __name__ == "__main__":
    do_all()
