import healpy as hp
import numpy as np

nside1=8192
nside2=nside1*2
nside4 = nside1 * 4
nsides = [nisde1,nside2,nside4]
beam_arr = np.loadtxt('/home/creichardt/spt3g_software/beams/products/compiled_2020_beams.txt')
theoryfile = '/sptlocal/user/creichardt/hiell2022/sim_dls_220ghz.txt'

dls = np.loadtxt(theoryfile)
locl=dls[:,0]
dl220 = dls[:,1]
bl220 = np.interp(locl,beam_arr[:,0],beam_arr[:,3])
udl220 = dl220 * bl220**2
ucl220 = udl220 * (2*np.pi)/(locl * (locl+1))
ucl220[:2]=0.0

alms = hp.synalm(ucl220,lmax=15000)

for nside in nsides:
    hmap = hp.alm2map(alms,nside)
    hp.write_map('/sptlocal/user/creichardt/test_{}.fits'.format(nside),hmap)
    hmap=None
