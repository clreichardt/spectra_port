import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import healpy as hp
#from spt3g import core,maps, calibration

import time
import pymaster as nmt
import astropy.io.fits as fits

AlmType = np.dtype(np.complex64)


ind_T=0
ind_Q=1
ind_U=2

def printinplace(myString):
    '''                                                                         
    Print in place -- ie overwriting the last one, not on a new line            
    '''
    digits = len(myString)
    delete = "\b" * (digits)
    print("{0}{1:{2}}".format(delete, myString, digits), end="")
    sys.stdout.flush()

def load_q(path,U=False):
    """Load Q/U from a FITS map, handling either a (Q,U) or (T,Q,U) layout."""
    ind=1
    if U:
        ind=2
    Q = hp.read_map(path, field=ind,dtype=np.float32)
    Q[Q == hp.UNSEEN] = 0.0
    return Q

def load_q_cut(path,U=False):
    """Load Q/U from a FITS map, handling either a (Q,U) or (T,Q,U) layout."""
    jnd=1
    if U:
        jnd=2
    with fits.open(path) as hdul:

        ind = hdul[1].data.field(0)
        q = hdul[1].data.field(1+jnd)
    return ind,q

def load_qu(path):
    """Load Q/U from a FITS map, handling either a (Q,U) or (T,Q,U) layout."""
    field_maps = hp.read_map(path, field=None, partial=False)
    field_maps = np.atleast_2d(field_maps)

    Q, U = field_maps[-2], field_maps[-1]
    Q[Q == hp.UNSEEN] = 0.0
    U[U == hp.UNSEEN] = 0.0
    return Q, U


def load_q(path,U=False):
    """Load Q/U from a FITS map, handling either a (Q,U) or (T,Q,U) layout."""
    ind=1
    if U:
        ind=2
    Q = hp.read_map(path, field=ind,dtype=np.float32)
    Q[Q == hp.UNSEEN] = 0.0
    return Q


def take_null_shts(map1filelist, map2filelist, shtfilelist,
                           nside,lmax,
                           purify_b = True,
                           mask  = None
                          ):
    oldtime = time.time()
    count=0
    fullU = np.zeros(12*8192**2,dtype=np.float64)
    fullQ = np.zeros(12*8192**2,dtype=np.float64)
    if map2filelist is not None:
        assert len(map1filelist) == len(map2filelist) == len(shtfilelist)
        nf = len(map1filelist)
        for i in range(nf):
            fullQ[:]=0.0
            ind,polmap = load_q_cut(map1filelist[i])
            fullQ[ind]=0.5*polmap
            ind,polmap = load_q_cut(map1filelist[i])
            fullQ[ind]-=0.5*polmap
            fullU[:]=0.0
            ind,polmap = load_q_cut(map1filelist[i],U=True)
            fullU[ind]=-0.5*polmap
            ind,polmap = load_q_cut(map1filelist[i],U=True)
            fullU[ind]+=0.5*polmap
            del ind,polmap

            if mask is None:
                mask = np.ones(12*8192**2,dtype=np.float64)

            print('done with load')
            #note U already multiplied by -1 above
            field = nmt.NmtField(mask, [fullQ, fullU], purify_e=False, purify_b=purify_b, lmax=lmax,lmax_mask=lmax, lite=True)
            print('field init done')
            _, alm_B = field.get_alms() #first one is alm_E which we don't need for nulls
            print('sht done')
            del field
            with open(shtfilelist[i],'wb') as fp:
                (alm_B.astype(AlmType)).tofile(fp)
            del alm_B
            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            oldtime=newtime
            printinplace('SHT map: {}  Last one took: {:.1f} minutes'.format(count,timeinminutes))
            count += 1
            
    else:  #LR nulls don't have a 2nd map list
        assert len(map1filelist) == len(shtfilelist)
        nf = len(map1filelist)
        for i in range(nf):
            Q,U = load_qu(map1filelist[i])
            if mask is None:
                mask = np.ones(Q.shape[0],dtype=np.float32)

            field = nmt.NmtField(mask, [Q, -U], purify_e=False, purify_b=purify_b, lmax=lmax, lite=True)
            _, alm_B = field.get_alms() #first one is alm_E which we don't need for nulls
            del field
            with open(shtfilelist[i],'wb') as fp:
                (alm_B.astype(AlmType)).tofile(fp)

            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            oldtime=newtime
            printinplace('SHT map: {}  Last one took: {:.1f} minutes'.format(count,timeinminutes))
            count += 1

