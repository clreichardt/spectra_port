import sys
from turtle import setundobuffer
import numpy as np
sys.path=["/home/creichardt/.local/lib/python3.7/site-packages/","/home/creichardt/spt3g_software/build","/home/creichardt/.local/lib/python3.7/site-packages/healpy-1.15.0-py3.7-linux-x86_64.egg"]+sys.path
import healpy

import os
from spt3g import core,maps, calibration
import pickle as pkl
AlmType = np.dtype(np.complex64)

def printinplace(myString):
    digits = len(myString)
    delete = "\b" * (digits)
    print("{0}{1:{2}}".format(delete, myString, digits), end="")
    sys.stdout.flush()


def load_spt3g_healpix_ring_map(file,require_order = 'Ring',require_nside=8192):
    # only taking 1st map 
    frames = core.G3File(file)
    for frame in frames:
        if frame.type == core.G3FrameType.Map:
            if require_nside is not None:
                assert(require_nside == frame['T'].nside)
            if require_order is not None:
                assert(frame['T'].nested == (require_order == 'Nest'))
            ind, map = frame['T'].nonzero_pixels()
            return( np.asarray(ind).astype(np.int64,casting='same_kind'), np.asarray(map).astype(np.float32,casting='same_kind') )
    raise Exception("No Map found in file: {}".format(file))

def take_and_reformat_shts(mapfilelist, processedshtfile,
                           nside,lmax,
                           cmbweighting = True, 
                           mask  = None,
                           kmask = None,
                           ell_reordering=None,
                           no_reorder=False,
                           ram_limit = None
                          ) -> 'May be done in Fortran - output is a file':
    ''' 
    Output is expected to be CL (Dl if cmbweighting=Trure) * mask normalization factor * kweights
    Output ordering is expected to 
    '''
    if ram_limit is None:
        ram_limit = 16 * 2**30 # 16 GB

    # number of bytes in a Dcomplex: 16
    # number of arrays we need to make to do this efficiently: 6 or less
    # number of pixels in an fft: winsize^2
    ram_required=16*6*lmax**2
    parallelism = int(np.ceil(ram_limit/ram_required))

    map_scratch = np.zeros(12*nside**2,dtype=np.float32)

    inv_mask_factor = 1.
    if mask is not None:
        inv_mask_factor = 1./np.mean(mask**2)

    #ie do parallelism SHTs at once...
    size = healpy.sphtfunc.Alm.getsize(lmax)
    if ell_reordering is None:  # need to make it
        #have lmax+1 m=0's, followed by lmax-1 m=1's.... (if does do l=0,m=0)
        # healpy has indexing routines, but they only take 1 at a time...
        #make dummy vec for use
        dummy_vec = np.zeros(lmax+1,dtype=np.uint)
        k=0
        for i in np.arange(lmax+1):
            dummy_vec[i] = k
            k=k+lmax-i
        ell_reordering = np.zeros(size,dtype=np.uint)
        k=0
        for i in range(lmax+1):
            ell_reordering[k:k+i+1] = dummy_vec[0:i+1] + i
            k += i+1

    print('Warning: not using pixel weights in SHT')
    with open(processedshtfile,'wb') as fp:

        count = 0 
        for file in mapfilelist:
            printinplace('SHT map: {}'.format(count))
            count += 1

            #TBD get a map
            ring_indices, map_tmp = load_spt3g_healpix_ring_map(file)

            map_scratch[:]=0 #reset
            map_scratch[ring_indices]=map_tmp #fill in the temperature map

            # if not already masked, apply mask
            #may need to change the next line based on formatting
            if mask is not None:
                map_scratch  = mask*map_scratch

            #gets alms
            alms = healpy.sphtfunc.map2alm(map_scratch,lmax = lmax, pol=False, use_pixel_weights=False, iter = 1,datapath='/sptlocal/user/creichardt/healpy-data/')

            #possibly downsample alms to save later CPU cycles
            # TBD if worthwhile

            #apply weighting:
            if kmask is not None:
                alms *= kmask

            #TBD, possibly adjust for mask factor here
            alms *= inv_mask_factor

            # Get reindexing

            #reorder and write to disk
            #need to check sizing
            #32 bit floats/64b complex should be fine for this. will need to bump up by one for aggregation
            if no_reorder:
                (alms.astype(AlmType)).tofile(fp)
            else:
                (alms[ell_reordering].astype(AlmType)).tofile(fp)

def get_first_index_ell(l):
    # l=0 - 0
    # l = 1 -> 1 (0+1)
    # l = 2 ->  3 (1+2)
    # l = 3 -> 6 (3+3)
    return int(l*(l+1)/2)


def generate_jackknife_shts( processed_shtfile, jackknife_shtfile,  lmax,
                             setdef) -> 'Does differencing to make SHT equiv file for nulls, returns new setdef':
    buffer_size = healpy.sphtfunc.Alm.getsize(lmax)
    buffer_bytes= buffer_size * AlmType.nbytes
    buffera = np.zeros(buffersize,dtype=AlmType)
    bufferb = np.zeros(buffersize,dtype=AlmType)
    setsize = setdef.shape[0]
    nsets   = setdef.shape[1]

    with open(processed_shtfile,'rb') as fin, open(jackknife_shtfile,'wb') as fout:
        for i in range(setsize):
            #need to do stuff here

            fin.seek( setdef[i,0] * buffer_bytes )
            buffera  = np.fromfile(fin,dtype=AlmType,count=buffer_size)
            fin.seek( setdef[i,1] * buffer_bytes )
            bufferb  = np.fromfile(fin,dtype=AlmType,count=buffer_size)

            buffera -= bufferb
            buffera *= 0.5
            fout.seek( i * buffer_bytes )
            buffera.tofile(fout)

    return(np.arange(setsize,dtype=np.int32))


def take_all_cross_spectra( processedshtfile, lmax,
                            setdef, banddef, ram_limit=None, auto = False) -> 'Returns set of all x-spectra, binned':
    '''
    ;; Step 1, copy all of the fft files and apply scalings masks etc


    ;; Step 2 (this function):  average all the bands to create binned x-spectra
    '''
    if ram_limit is None:
        ram_limit = 16 * 2**30 # default limit is 16 GB


    # Simplifying assumption axb == (a^c b + b^c a)
    # assume do *not* do x-spectra between same observation
    nsets   = setdef.shape[1]
    setsize = setdef.shape[0]
    nspectra=np.int((nsets*(nsets+1))/2 + 0.001)
    print(nsets,setsize,nspectra)
    if auto:
        nrealizations=setsize
    else:
        nrealizations=np.int( (setsize*(setsize-1))/2 + 0.001)

    nbands = banddef.shape[0]-1
    nshts  = np.int(np.max(setdef)+1.001)

    allspectra_out = np.zeros([nbands,nspectra,nrealizations],dtype=np.float32)
    nmodes_out     = np.zeros(nbands, dtype = np.int32)

    tmpresult = np.zeros([setsize,setsize],dtype=np.float64)

    # number of bytes in a Dcomplex: 16
    # number of arrays we need to make to do this efficiently: 6 or less
    # number of pixels in an fft: winsize^2
    ram_required=16*6*lmax**2
    max_nmodes=ram_limit/nshts/32 #64 b complex 

    #code=reverse_linefeed_code()

    i=0 # i is the last bin to have finished. initially 0
    while (i < nbands):
        istop = np.where((band_start_idx - band_start_idx[i]) < max_nmodes)[-1]

        if istop <= i:
            raise Exception("Insufficient ram for processing even a single bin")

        print('take_all_cross_spectra: loading bands {} {}'.format(i,istop-1))
        # technical: delete the last iteration of banddata_big first
        banddata_big=0
        # get data for as many bins as will fit in our ramlimit
        banddata_big=load_cross_spectra_data_from_disk(processedshtfile, 
                                                        nshts,   
                                                        start=bandstartidx[i],
                                                        stop=bandstartidx[istop]-1)
        #process this data
        for iprime in range(i, istop):
            printinplace('processing band {}    '.format(iprime))
                
            nmodes=(bandstartidx[iprime+1]-bandstartidx[iprime])
            nmodes_out[iprime]=nmodes
            aidx=bandstartidx[iprime]-bandstartidx[i]
            banddata=banddata_big[:,aidx:(aidx+nmodes-1)] # first index SHT; second index alm

            spectrum_idx=0
            for j in range(nsets):
                for k in range(j, nsets):
                    if not auto:
                        tmpresult  = np.real(np.matmult(banddata[setdef[:,j],:],banddata[setdef[:,k],:].T)) #need to check dims -- intended to end up for 3 freqs with 3x3 matrix
                        tmpresult += tmpresult.T # imposing the ab + ba condition
                        tmpresult /= (2*nmodes)
                        #it had a factor of 1/(reso**2 winsize**2) 
                        # leaving this out for curved sky
                        a=0
                        for l in range(setsize-1):
                            rowlength=setsize-l-1
                            allspectra_out[iprime, spectrum_idx, a:(a+rowlength-1)]=tmpresult[l, l+1:setsize-1]
                            a+=rowlength
                    else:
                        idx=lindgen(setsize)
                        tmpresult=np.sum(np.real(banddata[setdef[:, j],:]*conj(banddata[setdef[:, k],:])), 1,dtype=np.float64) / (nmodes) # had been in flatsky: *reso^2*winsize^2)
                        allspectra_out[iprime, spectrum_idx, :]=tmpresult.astype(np.float32)
                    spectrum_idx+=1

        i=istop
    return(allspectra_out)


def process_all_cross_spectra(allspectra, nbands, nsets,setsize, 
                              auto=False) -> 'Returns mean and covarariance estimates':

    print("Correlating Cross Spectra")
    nspectra = (nsets * (nsets+1))/2


    spectrumreformed = np.zeros([nbands,nspectra],dtype=np.float32)

    if auto:
        nrealizations = setsize
    else:
        nrealizations=(setsize*(setsize-1))/2

    allspectra = np.reshape(allspectra, [nbands*nspectra, nrealizations])
    #cov  = np.zeros([nbands*nspectra, nbands*nspectra],dtype=np.float64)

    spectrum = np.sum(allspectra,2,dtype=np.float64)/nrealizations

    # [nbands*nspectra, nrealizations])
    spectrum_2d = np.tile(spectrum, [1,nrealizations])

    cov1 = np.matmul((allspectra-spectrum_2d).T , allspectra-spectrum_2d)
    cov1/= (nrealizations*(nrealizations-1))

    cov2 = None
    if not auto:
        realization_to_complement=np.zeros([nrealizations, setsize],dtype=np.float64)

        for i in range(setsize):
            realization_idx = 0
            for j in range(setsize):
                for j in range(j+1,setsize):
                    if (i == j) or (i == k):
                        realization_to_complement[realization_idx, i]=1./(setsize-1)
                    realization_idx += 1

        allcomplementspectra=np.matmult(realization_to_complement,allspectra)
        #nbands*nspectra, setsize)
        spectrum_2d=np.tile(spectrum, [1,setsize])

        cov2=np.matmult( (allcomplementspectra-spectrum_2d).T, (allcomplementspectra-spectrum_2d) )
        cov2/=(setsize**2 / 2)
        cov=2*cov2-cov1

    else:
        cov=cov1*(nrealizations) 


    spectrum_dict = {}
    spectrum_dict['spectrum']=spectrum
    spectrum_dict['allspectra']=allspectra
    spectrum_dict['cov']=cov
    spectrum_dict['cov1']=cov1
    spectrum_dict['cov2']=cov2

    return spectrum_dict
