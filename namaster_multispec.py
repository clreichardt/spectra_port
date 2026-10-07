import os
#os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import healpy as hp
#from spt3g import core,maps, calibration

import time
import pymaster as nmt
import astropy.io.fits as fits
import sys
import gc
import pdb
from pathlib import Path
import pdb
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
                   mask  = None,
                   istart=0
                          ):
    oldtime = time.time()
    count=0
    fullU = np.zeros(12*8192**2,dtype=np.float64)
    fullQ = np.zeros(12*8192**2,dtype=np.float64)
    if map2filelist is not None:
        assert len(map1filelist) == len(map2filelist) == len(shtfilelist)
        nf = len(map1filelist)
        for i in range(istart,nf):
            if Path(shtfilelist[i]).is_file():
                continue #next loop
            fullQ[:]=0.0
            ind,polmap = load_q_cut(map1filelist[i])
            fullQ[ind]=0.5*polmap
            ind,polmap = load_q_cut(map2filelist[i])
            fullQ[ind]-=0.5*polmap
            fullU[:]=0.0
            ind,polmap = load_q_cut(map1filelist[i],U=True)
            fullU[ind]=-0.5*polmap
            ind,polmap = load_q_cut(map2filelist[i],U=True)
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


#####################################################################################################
# Below: ported from unbiased_multispec.py. These are pure numpy/healpy -- no spt3g dependency --
# and operate on the per-bundle alm files already produced above by take_null_shts. Unlike
# unbiased_multispec.reformat_shts (which read npz files with key 'alm'), reformat_shts here reads
# the raw complex64 binaries that take_null_shts writes directly.
#####################################################################################################

def get_first_index_ell(l):
    '''
    Index (in a healpy-ordered 1d alm array) of the m=0 element for ell=l.
    l=0 -> 0, l=1 -> 1, l=2 -> 3, l=3 -> 6, ...
    '''
    if type(l) is int:
        return int(l*(l+1)/2)
    elif type(l) is np.ndarray:
        return (l*(l+1)/2).astype(np.int64)
    else:
        pdb.set_trace()
        return -1


def reformat_shts(shtfilelist, processedshtfile,
                   lmax,
                   cmbweighting = True,
                   mask  = None,
                   kmask = None,
                   ell_reordering=None,
                   no_reorder=False,
                   ram_limit = None,
                  ):
    '''
    Ported from unbiased_multispec.reformat_shts.
    Reads raw complex64 alm files (as written by take_null_shts), applies cmbweighting
    (Cl -> Dl), an optional kmask, and an optional partial-sky mask normalization factor,
    then reorders to contiguous-per-ell ordering and concatenates to a single binary file.

    Output is expected to be Cl (Dl if cmbweighting=True) * mask normalization factor * kmask.
    '''
    if ram_limit is None:
        ram_limit = 32 * 2**30 # 32 GB

    inv_mask_factor = 1.
    if mask is not None:
        inv_mask_factor = np.sqrt(1./np.mean(mask**2))

    size = hp.sphtfunc.Alm.getsize(lmax)
    if kmask is not None:
        if kmask.shape[0] != size:
            raise Exception("kmask provided is wrong size ({} vs {}), exiting".format(size,kmask.shape[0]))
        local_kmask = kmask.astype(np.float32)
        print("using provided kmask")
    else:
        local_kmask = np.ones(size,dtype=np.float32)
        print("kmask  is unity")

    if cmbweighting:
        dummy_vec = np.arange(lmax+1,dtype=np.float32)
        dummy_vec = np.sqrt((dummy_vec*(dummy_vec+1.))/(2*np.pi)) # This will be squared since Cl =a*a
        j=0
        for i in range(lmax+1):
            nm = lmax+1-i
            local_kmask[j:j+nm]*=dummy_vec[i:]
            j=j+nm

    if ell_reordering is None:  # need to make it
        #have lmax+1 m=0's, followed by lmax m=1's.... (if does do l=0,m=0)
        dummy_vec = np.zeros(lmax+1,dtype=np.int64)
        k=0
        for i in np.arange(lmax+1):
            dummy_vec[i] = k
            k=k+lmax-i
        ell_reordering = np.zeros(size,dtype=np.int64)
        k=0
        for i in range(lmax+1):
            ell_reordering[k:k+i+1] = dummy_vec[0:i+1] + i
            k += i+1

    with open(processedshtfile,'wb') as fp:
        oldtime = time.time()
        count = 0
        for file in shtfilelist:
            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            oldtime=newtime
            printinplace('Reformat SHT: {}  Last one took: {:.1f} minutes'.format(count,timeinminutes))
            count += 1

            alms = np.fromfile(file, dtype=AlmType)
            assert lmax ==  hp.sphtfunc.Alm.getlmax(alms.shape[0])

            #apply weighting (ie cl-dl) and kmask
            alms = alms * local_kmask

            #adjust for partial-sky mask normalization factor
            alms = alms * inv_mask_factor

            #reorder and write to disk
            if no_reorder:
                (alms.astype(AlmType)).tofile(fp)
            else:
                (alms[ell_reordering].astype(AlmType)).tofile(fp)


def load_cross_spectra_data_from_disk_in_place(shtfile,data, startsht,stopsht, npersht, start, stop):
    '''
    Ported from unbiased_multispec.load_cross_spectra_data_from_disk_in_place.
    '''
    nelems = stop - start + 1
    nshts = stopsht - startsht + 1
    buffer_bytes = np.zeros(1,dtype=AlmType).nbytes
    assert data.shape[0] >= nshts and data.shape[1] >= nelems
    with open(shtfile,'rb') as fp:
        for i in range(nshts):
            j = i + startsht
            fp.seek((j*npersht+start) * buffer_bytes)
            data[i,:nelems] = np.fromfile(fp,count=nelems,dtype=AlmType)
    return data


def take_all_cross_spectra( processedshtfile, lmax,
                            setdef, banddef, ram_limit=None, auto = False,nshts=None,kmask_on_the_fly=None,
                            kmask_on_the_fly_ranges=None,
                            splitband=False):
    '''
    Ported from unbiased_multispec.take_all_cross_spectra.
    Returns set of all x-spectra (or auto-spectra if auto=True), binned per banddef.
    '''
    if ram_limit is None:
        ram_limit = 40 * 2**30 # default limit is 40 GB

    # Simplifying assumption axb == (a^c b + b^c a)
    # assume do *not* do x-spectra between same observation
    nsets   = setdef.shape[1] #nfreq
    setsize = setdef.shape[0] #nbundles
    nspectra=int((nsets*(nsets+1))/2 + 0.001)
    print(nsets,setsize,nspectra)
    if auto:
        nrealizations=setsize
    else:
        nrealizations=int( (setsize*(setsize-1))/2 + 0.001)

    nbands = banddef.shape[0]-1
    if nshts is  None:
        startsht = int(np.min(setdef)+0.001)
        stopsht = int(np.max(setdef)+0.001)
        nshts  = stopsht-startsht+1
        revsetdef = setdef - startsht
    else:
        startsht=0
        stopsht = nshts-1
        revsetdef=setdef

    npersht = hp.sphtfunc.Alm.getsize(lmax)
    print('check modes: {} {}'.format(lmax,npersht))
    allspectra_out = np.zeros([nbands,nspectra,nrealizations],dtype=np.float32)
    nmodes_out     = np.zeros(nbands, dtype = np.int32)

    max_nmodes=ram_limit/nshts/12 #64 b complex - uses 8 bytes, and gave it an extra x1.5 for other arrays

    print('take_all bandefs',banddef[0],banddef[-1],lmax)
    assert(banddef[0] == 0 and banddef[-1] <= lmax)
    #assumes banddef[0]=0
    #so first bin goes 1 - banddef[1]
    # second bin goes banddef[1]+1 - banddef[2], etc
    band_start_idx = get_first_index_ell(banddef+1)

    mmax = -1
    i=0 # i is the last bin to have finished. initially 0
    while (i < nbands):
        istop = np.where((band_start_idx - band_start_idx[i]) < max_nmodes)[0][-1]
        if istop <= i:
            raise Exception("Insufficient ram for processing even a single bin")
        nn = band_start_idx[istop]-band_start_idx[i]
        if nn > mmax:
            mmax = nn
        i=istop
    print("Memory limit on nmodes of {}, actual size requested is {}".format(max_nmodes,mmax))
    banddata_big = np.zeros([nshts, mmax],dtype=AlmType)

    i=0 # i is the last bin to have finished. initially 0
    while (i < nbands):
        istop = np.where((band_start_idx - band_start_idx[i]) < max_nmodes)[0][-1] # get out of tuple, then take last elem of array

        print('take_all_cross_spectra: loading bands {} {} of {}'.format(i,istop-1,nbands))
        load_cross_spectra_data_from_disk_in_place(processedshtfile, banddata_big,
                                                       startsht, stopsht,
                                                       npersht,
                                                       band_start_idx[i],
                                                       band_start_idx[istop]-1 )

        if kmask_on_the_fly_ranges is not None:
            nn = band_start_idx[istop] - band_start_idx[i]
            for k in range(kmask_on_the_fly_ranges.shape[0]):
                banddata_big[kmask_on_the_fly_ranges[k,0]:kmask_on_the_fly_ranges[k,1],:nn] *= kmask_on_the_fly[k,band_start_idx[i]:band_start_idx[istop]]
        #process this data
        for iprime in range(i, istop):
            printinplace('processing band {}    '.format(iprime))

            nmodes=(band_start_idx[iprime+1]-band_start_idx[iprime])
            nmodes_out[iprime]=nmodes
            aidx=band_start_idx[iprime]-band_start_idx[i]
            banddata=banddata_big[:,aidx:(aidx+nmodes)] # first index SHT; second index alm

            spectrum_idx=0
            for j in range(nsets):
                for k in range(j, nsets):
                    if not auto:

                        if splitband:
                            n2 = nmodes//2
                            tmpresult  = np.real(np.matmul(banddata[revsetdef[:,j],:n2],np.conj(banddata[revsetdef[:,k],:n2]).T)) #need to check dims -- intended to end up for 3 freqs with 3x3 matrix
                            tmpresult += np.real(np.matmul(banddata[revsetdef[:,j],n2:],np.conj(banddata[revsetdef[:,k],n2:]).T))
                        else:
                            tmpresult  = np.real(np.matmul(banddata[revsetdef[:,j],:],np.conj(banddata[revsetdef[:,k],:]).T)) #need to check dims -- intended to end up for 3 freqs with 3x3 matrix

                        tmpresult += tmpresult.T # imposing the ab + ba condition
                        tmpresult /= (2*nmodes)
                        a=0
                        for l in range(setsize-1):
                            rowlength=setsize-l-1
                            allspectra_out[iprime, spectrum_idx, a:(a+rowlength)]=tmpresult[l, l+1:setsize]
                            a+=rowlength
                    else:
                        if splitband:
                            n2 = nmodes//2
                            tmpresult=np.sum(np.real(banddata[revsetdef[:, j],:n2]*np.conj(banddata[revsetdef[:, k],:n2])), 1,dtype=np.float64)
                            tmpresult+=np.sum(np.real(banddata[revsetdef[:, j],n2:]*np.conj(banddata[revsetdef[:, k],n2:])), 1,dtype=np.float64)
                            tmpresult /= (nmodes)
                        else:
                            tmpresult=np.sum(np.real(banddata[revsetdef[:, j],:]*np.conj(banddata[revsetdef[:, k],:])), 1,dtype=np.float64) / (nmodes)
                        allspectra_out[iprime, spectrum_idx, :]=tmpresult.astype(np.float32)
                    spectrum_idx+=1
                    del tmpresult
                    gc.collect()
        i=istop
    del banddata_big
    gc.collect()
    return(allspectra_out, nmodes_out)


def process_all_cross_spectra(allspectra, nbands, nsets,setsize,
                              auto=False,
                              skipcov=False ):
    """
    Ported from unbiased_multispec.process_all_cross_spectra.
    Returns mean and covariance estimates.
    """
    print("Correlating Cross Spectra")
    nspectra = int( (nsets * (nsets+1))/2 + 0.001)

    if auto:
        nrealizations = setsize
    else:
        nrealizations=int( (setsize*(setsize-1))/2 + 0.001)

    allspectra = np.reshape(allspectra, [nbands*nspectra, nrealizations])
    #ordering of first bin is 0th bin of all spectra, then 1st bin, etc.

    spectrum = np.sum(allspectra,-1,dtype=np.float64)/nrealizations

    spectrum = np.reshape(spectrum,[nbands,nspectra])
    if skipcov:
        return spectrum,None,None,None
    spectrum_2d = np.tile(np.reshape(spectrum,[nbands*nspectra,1]), [1,nrealizations])

    cov1 = np.matmul((allspectra-spectrum_2d) , (allspectra-spectrum_2d).T)
    cov1/= (nrealizations*(nrealizations-1))

    cov2 = None
    if not auto:
        realization_to_complement=np.zeros([nrealizations, setsize],dtype=np.float64)

        for i in range(setsize):
            realization_idx = 0
            for j in range(setsize):
                for k in range(j+1,setsize):
                    if (i == j) or (i == k):
                        realization_to_complement[realization_idx, i]=1./(setsize-1)
                    realization_idx += 1

        allcomplementspectra=np.matmul(allspectra,realization_to_complement)
        spectrum_2d=np.tile(np.reshape(spectrum,[nbands*nspectra,1]), [1,setsize])

        cov2=np.matmul( (allcomplementspectra-spectrum_2d), (allcomplementspectra-spectrum_2d).T )
        cov2/=(setsize**2 / 2)
        cov=2*cov2-cov1

    else:
        cov=cov1*(nrealizations)

    return spectrum,cov,cov1,cov2


def correct_by_kmask_factor(allspectra_in, kmask_sq, banddef, eps = 1e-12):
    '''
    Ported from unbiased_multispec.correct_by_kmask_factor.
    Default behavior is for cross-spectra to return the average of alms**2 * kmask**2;
    this swaps that to be the weighted average of alms**2, for weight array of kmask**2.
    allspectra_in -- [l-bin, freq-combo, Nsims] (or [l-bin, Nsims])
    '''
    band_start_idx = get_first_index_ell(banddef+1)
    nbands = banddef.shape[0]-1
    factors = np.zeros(nbands)
    for i in range(nbands):
        factors[i]= np.mean(kmask_sq[band_start_idx[i]:band_start_idx[i+1]])
    factors[factors < eps] = 1 #if kmask is 0, want 0 not NaN
    newd = np.asarray(allspectra_in.shape)
    newd[:] = 1
    newd[0] = nbands
    factors = np.reshape(factors,newd)
    return allspectra_in/factors

