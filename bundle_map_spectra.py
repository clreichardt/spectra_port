import numpy as np
import glob



def make_mean_spectrum(stub):
    flist = glob.glob(stub)
    odl = 0
    for file in flist:
        vec = np.loadtxt(file)
        odl += vec[:,1]
    odl[:2]=0
    vec[:,1]=odl
    return vec




if __name__ == "__main__":
    f90='/sptlocal/user/creichardt/out_maps/dl_95ghz.txt'
    dl90 = make_mean_spectrum('/sptlocal/user/creichardt/out_maps/xfer/sim*nobeam_95ghz.dat')
    np.savetxt(f90,dl90)
    f150='/sptlocal/user/creichardt/out_maps/dl_150ghz.txt'
    dl150 = make_mean_spectrum('/sptlocal/user/creichardt/out_maps/xfer/sim*nobeam_150ghz.dat')
    np.savetxt(f150,dl150)
    f220='/sptlocal/user/creichardt/out_maps/dl_220ghz.txt'
    dl220 = make_mean_spectrum('/sptlocal/user/creichardt/out_maps/xfer/sim*nobeam_220ghz.dat')
    np.savetxt(f220,dl220)