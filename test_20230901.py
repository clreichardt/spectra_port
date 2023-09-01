import healpy as hp
import numpy as np

nsidebig = 8192 * 4
npixbig = 12 * nsidebig**2
split = nsidebig**2
locind = np.arange(0,split)
mapfile = '/sptlocal/user/creichardt/out_maps/sim_220ghz_174.g3'
inmap = hp.read_map(mapfile)

omap = np.zeros(npixbig,dtype=np.float32)
for i in range(12):
    print(i)
    theta,phi = hp.pix2ang(nsidebig,locind)

    vals = hp.get_interp_val(inmap,theta,phi)
    omap[i*split:(i+1)*split] = vals
    locind += split

with open('hires.bin','wb') as fp:
    omap.tofile(fp)


