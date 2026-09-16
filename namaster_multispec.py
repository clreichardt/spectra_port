import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
#from spt3g import core,maps, calibration
from spectra_port import  unbiased_multispec
import time
import pymaster as nmt

AlmType = np.dtype(np.complex64)



def load_qu(path):
    """Load Q/U from a FITS map, handling either a (Q,U) or (T,Q,U) layout."""
    field_maps = hp.read_map(path, field=None, partial=False)
    field_maps = np.atleast_2d(field_maps)

    Q, U = field_maps[-2], field_maps[-1]
    Q[Q == hp.UNSEEN] = 0.0
    U[U == hp.UNSEEN] = 0.0
    return Q, U


def take_null_shts(map1filelist, map2filelist, shtfilelist,
                           nside,lmax,
                           purify_b = True,
                           mask  = None
                          ):
    oldtime = time.time()
    count=0
    if map2filelist is not None:
        assert len(map1filelist) == len(map2filelist) == len(shtfilelist)
        nf = len(map1filelist)
        for i in range(nf):
            Q,U = load_qu(map1filelist[i])
            Q2,U2 = load_qu(map2filelist[i])
            Q = 0.5*(Q-Q2)
            U = 0.5*(U-U2)
            del Q2,U2
            if mask is None:
                mask = np.ones(Q.shape[0],dtype=np.float64)
            assert Q.shape[0] == 12*nside**2

            field = nmt.NmtField(mask, [Q, -U], purify_e=False, purify_b=purify_b, lmax=lmax, lite=True)
            _, alm_B = field.get_alms() #first one is alm_E which we don't need for nulls
            with open(shtfilelist[i],'wb') as fp:
                (alm_B.astype(AlmType)).tofile(fp)

            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            oldtime=newtime
            unbiased_multispec.printinplace('SHT map: {}  Last one took: {:.1f} minutes'.format(count,timeinminutes))
            count += 1
            
    else:  #LR nulls don't have a 2nd map list
        assert len(map1filelist) == len(shtfilelist)
        nf = len(map1filelist)
        for i in range(nf):
            Q,U = load_qu(map1filelist[i])
            if mask is None:
                mask = np.ones(Q.shape[0],dtype=np.float64)

            field = nmt.NmtField(mask, [Q, -U], purify_e=False, purify_b=purify_b, lmax=lmax, lite=True)
            _, alm_B = field.get_alms() #first one is alm_E which we don't need for nulls
            with open(shtfilelist[i],'wb') as fp:
                (alm_B.astype(AlmType)).tofile(fp)

            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            oldtime=newtime
            unbiased_multispec.printinplace('SHT map: {}  Last one took: {:.1f} minutes'.format(count,timeinminutes))
            count += 1

