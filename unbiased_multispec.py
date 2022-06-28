from genericpath import getsize
#from statistics import covariance
import sys
from turtle import setundobuffer
import numpy as np
#sys.path=["/home/creichardt/.local/lib/python3.7/site-packages/","/home/creichardt/spt3g_software/build","/home/creichardt/.local/lib/python3.7/site-packages/healpy-1.15.0-py3.7-linux-x86_64.egg"]+sys.path
import healpy

import os
from spt3g import core,maps, calibration
from spectra_port import utils
import pickle as pkl
import pdb
import time
AlmType = np.dtype(np.complex64)


def name_tempdir(basedir):
    while True:
        rand = np.random.randint(0,999999)
        path = "{}/workdir_{:6d}".format(basedir, rand)
        if not os.path.exists(path):
            return path

def printinplace(myString):
    digits = len(myString)
    delete = "\b" * (digits)
    print("{0}{1:{2}}".format(delete, myString, digits), end="")
    sys.stdout.flush()


def load_spt3g_healpix_ring_map(file,require_order = 'Ring',require_nside=8192,map_key='T'):
    # only taking 1st map 
    frames = core.G3File(file)
    for frame in frames:
        if frame.type == core.G3FrameType.Map:
            if require_nside is not None:
                assert(require_nside == frame[map_key].nside)
            if require_order is not None:
                assert(frame[map_key].nested == (require_order == 'Nest'))
            if type(frame[map_key]) is np.ndarray:
                ind = frame[map_key].nonzero()
                map = frame[map_key][frame[map_key] != 0]
            else:
                ind, map = frame[map_key].nonzero_pixels()
            return( np.asarray(ind).astype(np.int64,casting='same_kind'), np.asarray(map).astype(np.float32,casting='same_kind') )
    raise Exception("No Map found in file: {}".format(file))




def reformat_shts(shtfilelist, processedshtfile,
                           lmax,
                           cmbweighting = True, 
                           mask  = None,
                           kmask = None,
                           ell_reordering=None,
                           no_reorder=False,
                           ram_limit = None,
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

    inv_mask_factor = 1.
    if mask is not None:
        inv_mask_factor = np.sqrt(1./np.mean(mask**2))

    #ie do parallelism SHTs at once...
    size = healpy.sphtfunc.Alm.getsize(lmax)
    if kmask is not None:
        if kmask.shape[0] != size:
            raise Exception("kmask provided is wrong size ({} vs {}), exiting".format(size,kmask.shape[0]))
        local_kmask = kmask.astype(np.float32)
    else:
        local_kmask = np.ones(size,dtype=np.float32)

        
    if cmbweighting:
        dummy_vec = np.arange(lmax+1,dtype=np.float32)
        dummy_vec = np.sqrt((dummy_vec*(dummy_vec+1.))/(2*np.pi)) # This will be squared since Cl =a*a
        j=0
        for i in range(lmax+1):
            nm = lmax+1-i
            local_kmask[j:j+nm]=dummy_vec[i:]
            j=j+nm


    if ell_reordering is None:  # need to make it
        #have lmax+1 m=0's, followed by lmax m=1's.... (if does do l=0,m=0)
        # healpy has indexing routines, but they only take 1 at a time...
        #make dummy vec for use
        dummy_vec = np.zeros(lmax+1,dtype=np.int)
        k=0
        for i in np.arange(lmax+1):
            dummy_vec[i] = k
            k=k+lmax-i
        ell_reordering = np.zeros(size,dtype=np.int)
        k=0
        for i in range(lmax+1):
            ell_reordering[k:k+i+1] = dummy_vec[0:i+1] + i
            k += i+1



    print('Warning: not using pixel weights in SHT')
    with open(processedshtfile,'wb') as fp:

        oldtime = time.time()
        count = 0 
        for file in shtfilelist:
            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            oldtime=newtime
            printinplace('SHT map: {}  Last one took: {:.1f} minutes'.format(count,timeinminutes))
            count += 1

            #TBD get SHT
            with np.load(file) as obs_alms:
                alms = obs_alms['alm']
            assert lmax ==  healpy.sphtfunc.Alm.getlmax(alms.shape[0])

            #possibly downsample alms to save later CPU cycles
            # TBD if worthwhile

            #apply weighting (ie cl-dl) and kmask 
            alms *= local_kmask

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

def take_and_reformat_shts(mapfilelist, processedshtfile,
                           nside,lmax,
                           cmbweighting = True, 
                           mask  = None,
                           kmask = None,
                           ell_reordering=None,
                           no_reorder=False,
                           ram_limit = None,
                           npmapformat=False,
                           pklmapformat=False,
                           map_key='T'
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
        inv_mask_factor = np.sqrt(1./np.mean(mask**2))

    if mask is not None:
        if type(mask) is np.ndarray:
            map_inds = mask.nonzero()
            cut_mask = mask[mask != 0]
        else:
            map_inds, cut_mask = mask.nonzero_pixels()
        npix = len(map_inds)
        
    #ie do parallelism SHTs at once...
    size = healpy.sphtfunc.Alm.getsize(lmax)
    if kmask is not None:
        if kmask.shape[0] != size:
            raise Exception("kmask provided is wrong size ({} vs {}), exiting".format(size,kmask.shape[0]))
        local_kmask = kmask.astype(np.float32)
    else:
        local_kmask = np.ones(size,dtype=np.float32)

        
    if cmbweighting:
        dummy_vec = np.arange(lmax+1,dtype=np.float32)
        dummy_vec = np.sqrt((dummy_vec*(dummy_vec+1.))/(2*np.pi)) #this will be squared later as Cl=a*a
        j=0
        for i in range(lmax+1):
            nm = lmax+1-i
            local_kmask[j:j+nm]=dummy_vec[i:]
            j=j+nm


    if ell_reordering is None:  # need to make it
        #have lmax+1 m=0's, followed by lmax m=1's.... (if does do l=0,m=0)
        # healpy has indexing routines, but they only take 1 at a time...
        #make dummy vec for use
        dummy_vec = np.zeros(lmax+1,dtype=np.int)
        k=0
        for i in np.arange(lmax+1):
            dummy_vec[i] = k
            k=k+lmax-i
        ell_reordering = np.zeros(size,dtype=np.int)
        k=0
        for i in range(lmax+1):
            ell_reordering[k:k+i+1] = dummy_vec[0:i+1] + i
            k += i+1



    print('Warning: not using pixel weights in SHT')
    with open(processedshtfile,'wb') as fp:

        oldtime = time.time()
        count = 0 
        for file in mapfilelist:
            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            oldtime=newtime
            printinplace('SHT map: {}  Last one took: {:.1f} minutes'.format(count,timeinminutes))
            count += 1

            #TBD get a map
            if pklmapformat:
                with open(file,'rb') as fb:
                    map_scratch = pkl.load(fb)
                if mask is not None:
                    map_scratch  = mask*map_scratch
            elif npmapformat:
                map_tmp = utils.load_spt3g_cutsky_healpix_ring_map(file,npix)
                map_scratch[map_inds]=map_tmp * cut_mask

            else:
                ring_indices, map_tmp = load_spt3g_healpix_ring_map(file,map_key=map_key)

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

            #apply weighting (ie cl-dl) and kmask 
            alms *= local_kmask

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
    if type(l) is int:
        return int(l*(l+1)/2)
    elif type(l) is np.ndarray:
        return (l*(l+1)/2).astype(np.int)
    else:
        pdb.set_trace()
        return -1

def generate_jackknife_shts( processed_shtfile, jackknife_shtfile,  lmax,
                             setdef) -> 'Does differencing to make SHT equiv file for nulls, returns new setdef':
    buffer_size = healpy.sphtfunc.Alm.getsize(lmax)
    buffer_bytes= buffer_size * np.zeros(1,dtype=AlmType).nbytes
    buffera = np.zeros(buffer_size,dtype=AlmType)
    bufferb = np.zeros(buffer_size,dtype=AlmType)
    setsize = setdef.shape[0]
    nsets   = setdef.shape[1]

    oldtime=time.time()
    with open(processed_shtfile,'rb') as fin, open(jackknife_shtfile,'wb') as fout:
        for i in range(setsize):
            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            oldtime=newtime
            printinplace('Creating null SHT : {} of {}  Last one took: {:.1f} minutes'.format(i,setsize,timeinminutes))
            #need to do stuff here

            fin.seek( setdef[i,0] * buffer_bytes )
            buffera  = np.fromfile(fin,dtype=AlmType,count=buffer_size)
            fin.seek( setdef[i,1] * buffer_bytes )
            bufferb  = np.fromfile(fin,dtype=AlmType,count=buffer_size)

            buffera -= bufferb
            buffera *= 0.5
            #fout.seek( i * buffer_bytes )
            (buffera.astype(AlmType)).tofile(fout)


    return(np.reshape(np.arange(setsize,dtype=np.int32),[setsize,1]))

def generate_coadd_shts( processed_shtfile, coadd_shtfile,  lmax,
                             setdef) -> 'Does differencing to make SHT equiv file for nulls, returns new setdef':
    '''
    Setdef in: Nbundles_out (dim0) x Nmaps_to_coadd (dim1)
    Setdef out: Nbundles_out (same as input dim0)
    '''
    
    buffer_size = healpy.sphtfunc.Alm.getsize(lmax)
    buffer_bytes= buffer_size * np.zeros(1,dtype=AlmType).nbytes
    buffera = np.zeros(buffer_size,dtype=AlmType)
    bufferb = np.zeros(buffer_size,dtype=AlmType)
    setsize = setdef.shape[0]
    nsets   = setdef.shape[1]

    with open(processed_shtfile,'rb') as fin, open(coadd_shtfile,'wb') as fout:
        for i in range(setsize):
            #need to do stuff here

            fin.seek( setdef[i,0] * buffer_bytes )
            buffera  = np.fromfile(fin,dtype=AlmType,count=buffer_size)
            for j in range(1,nsets):
                fin.seek( setdef[i,1] * buffer_bytes )
                bufferb  = np.fromfile(fin,dtype=AlmType,count=buffer_size)
                buffera += bufferb
            buffera *= (1./nsets)
            fout.seek( i * buffer_bytes )
            buffera.tofile(fout)

    return(np.reshape(np.arange(setsize,dtype=np.int32),[setsize,1]))


def load_cross_spectra_data_from_disk(shtfile, startsht,stopsht, npersht, start, stop):
    nelems = stop - start + 1
    nshts = stopsht - startsht + 1
    buffer_bytes = np.zeros(1,dtype=AlmType).nbytes
    data = np.zeros([nshts,nelems],dtype=AlmType)
    print(nshts,nelems)
    with open(shtfile,'r') as fp:
        for i in range(nshts):
            j = i + startsht
            fp.seek((j*npersht+start) * buffer_bytes)
            data[i,:] = np.fromfile(fp,count=nelems,dtype=AlmType)
    return data


def take_all_cross_spectra( processedshtfile, lmax,
                            setdef, banddef, ram_limit=None, auto = False,nshts=None) -> 'Returns set of all x-spectra, binned':
    '''
    ;; Step 1, copy all of the fft files and apply scalings masks etc


    ;; Step 2 (this function):  average all the bands to create binned x-spectra
    '''
    if ram_limit is None:
        ram_limit = 64 * 2**30 # default limit is 64 GB


    # Simplifying assumption axb == (a^c b + b^c a)
    # assume do *not* do x-spectra between same observation
    nsets   = setdef.shape[1] #nfreq
    setsize = setdef.shape[0] #nbundles
    nspectra=np.int((nsets*(nsets+1))/2 + 0.001)
    print(nsets,setsize,nspectra)
    if auto:
        nrealizations=setsize
    else:
        nrealizations=np.int( (setsize*(setsize-1))/2 + 0.001)

    nbands = banddef.shape[0]-1
    if nshts is  None:
        startsht = np.int(np.min(setdef)+0.001)
        stopsht = np.int(np.max(setdef)+0.001)
        nshts  = stopsht-startsht+1
        revsetdef = setdef - startsht
    else:
        startsht=0
        stopsht = nshts-1
        revsetdef=setdef

    npersht = healpy.sphtfunc.Alm.getsize(lmax)
    #pdb.set_trace()
    allspectra_out = np.zeros([nbands,nspectra,nrealizations],dtype=np.float32)
    nmodes_out     = np.zeros(nbands, dtype = np.int32)

    tmpresult = np.zeros([setsize,setsize],dtype=np.float64)

    # number of bytes in a Dcomplex: 16
    # number of arrays we need to make to do this efficiently: 6 or less
    # number of pixels in an fft: winsize^2
    #ram_required=16*6*lmax**2
    max_nmodes=ram_limit/nshts/12 #64 b complex - uses 8 bytes, and gave it an extra x1.5 for other arrays 


    assert(banddef[0] == 0 and banddef[-1] < lmax)
    #assumes banddef[0]=0
    #so first bin goes 1 - banddef[1]
    # second bin goes banddef[1]+1 - banddef[2], etc
    band_start_idx = get_first_index_ell(banddef+1)

    #code=reverse_linefeed_code()

    i=0 # i is the last bin to have finished. initially 0
    while (i < nbands):
        istop = np.where((band_start_idx - band_start_idx[i]) < max_nmodes)[0][-1] # get out of tuple, then take last elem of array

        if istop <= i:
            pdb.set_trace()
            raise Exception("Insufficient ram for processing even a single bin")

        print('take_all_cross_spectra: loading bands {} {}'.format(i,istop-1))
        # technical: delete the last iteration of banddata_big first
        banddata_big=0
        # get data for as many bins as will fit in our ramlimit

        banddata_big=load_cross_spectra_data_from_disk(processedshtfile, 
                                                       startsht, stopsht, 
                                                       npersht,   
                                                       band_start_idx[i],
                                                       band_start_idx[istop]-1 )
        #process this data
        for iprime in range(i, istop):
            printinplace('processing band {}    '.format(iprime))
                
            nmodes=(band_start_idx[iprime+1]-band_start_idx[iprime])
            nmodes_out[iprime]=nmodes
            aidx=band_start_idx[iprime]-band_start_idx[i]
            banddata=banddata_big[:,aidx:(aidx+nmodes-1)] # first index SHT; second index alm


            spectrum_idx=0
            for j in range(nsets):
                for k in range(j, nsets):
                    if not auto:

                        tmpresult  = np.real(np.matmul(banddata[revsetdef[:,j],:],np.conj(banddata[revsetdef[:,k],:]).T)) #need to check dims -- intended to end up for 3 freqs with 3x3 matrix

                        tmpresult += tmpresult.T # imposing the ab + ba condition
                        tmpresult /= (2*nmodes)
                        #it had a factor of 1/(reso**2 winsize**2) 
                        # leaving this out for curved sky
                        a=0
                        for l in range(setsize-1):
                            rowlength=setsize-l-1
                            allspectra_out[iprime, spectrum_idx, a:(a+rowlength)]=tmpresult[l, l+1:setsize]
                            a+=rowlength
                    else:
                        idx=np.arange(setsize,dtype=np.int)
                        tmpresult=np.sum(np.real(banddata[revsetdef[:, j],:]*np.conj(banddata[revsetdef[:, k],:])), 1,dtype=np.float64) / (nmodes) # had been in flatsky: *reso^2*winsize^2)
                        allspectra_out[iprime, spectrum_idx, :]=tmpresult.astype(np.float32)
                    spectrum_idx+=1
                    #           pdb.set_trace()
        i=istop
    return(allspectra_out, nmodes_out)


def take_all_sim_cross_spectra( processedshtfile, lmax,
                            setdef1,  banddef, setdef2=None, ram_limit=None, auto=False) -> 'Returns set of all x-spectra, binned':
    '''
    ;; Step 1, copy all of the fft files and apply scalings masks etc


    ;; Step 2 (this function):  average all the bands to create binned x-spectra
    ;; this assumes sims are created with two bundles
    '''
    if ram_limit is None:
        ram_limit = 32 * 2**30 # default limit is 32 GB


    # Simplifying assumption axb == (a^c b + b^c a)
    # assume do *not* do x-spectra between same observation
    nsets   = setdef1.shape[1]
    setsize = setdef1.shape[0]
    nspectra=np.int((nsets*(nsets+1))/2 + 0.001)
    print(nsets,setsize,nspectra)
    
    if auto is False:
        assert setdef2 is not None
    
    nrealizations=setsize

    nbands = banddef.shape[0]-1

    startsht = np.int(np.min([setdef1,setdef2])+0.001)
    stopsht = np.int(np.max([setdef1,setdef2])+0.001)
    nshts  = stopsht-startsht + 1
    revsetdef1 = setdef1 - startsht
    if setdef2 is not None:
        revsetdef2 = setdef2 - startsht
        
    npersht = healpy.sphtfunc.Alm.getsize(lmax)
    #pdb.set_trace()
    allspectra_out = np.zeros([nbands,nspectra,nrealizations],dtype=np.float32)
    nmodes_out     = np.zeros(nbands, dtype = np.int32)

    tmpresult = np.zeros([setsize,setsize],dtype=np.float64)

    # number of bytes in a Dcomplex: 16
    # number of arrays we need to make to do this efficiently: 6 or less
    # number of pixels in an fft: winsize^2
    ram_required=16*6*lmax**2
    max_nmodes=ram_limit/nshts/32 #64 b complex 


    assert(banddef[0] == 0 and banddef[-1] <= lmax)
    #assumes banddef[0]=0
    #so first bin goes 1 - banddef[1]
    # second bin goes banddef[1]+1 - banddef[2], etc
    band_start_idx = get_first_index_ell(banddef+1)

    #code=reverse_linefeed_code()

    i=0 # i is the last bin to have finished. initially 0
    while (i < nbands):
        istop = np.where((band_start_idx - band_start_idx[i]) < max_nmodes)[0][-1] # get out of tuple, then take last elem of array

        if istop <= i:
            print('ram hit:',max_modes, band_start_idx[i],band_start_idx[i+1])
            raise Exception("Insufficient ram for processing even a single bin")

        print('take_all_cross_spectra: loading bands {} {}'.format(i,istop-1))
        # technical: delete the last iteration of banddata_big first
        banddata_big=0
        # get data for as many bins as will fit in our ramlimit

        banddata_big=load_cross_spectra_data_from_disk(processedshtfile, 
                                                       startsht, stopsht, 
                                                       npersht,   
                                                       band_start_idx[i],
                                                       band_start_idx[istop]-1 )
        #process this data
        for iprime in range(i, istop):
            printinplace('processing band {}    '.format(iprime))
                
            nmodes=(band_start_idx[iprime+1]-band_start_idx[iprime])
            nmodes_out[iprime]=nmodes
            aidx=band_start_idx[iprime]-band_start_idx[i]
            banddata=banddata_big[:,aidx:(aidx+nmodes-1)] # first index SHT; second index alm


            spectrum_idx=0
            for j in range(nsets):
                for k in range(j, nsets):
                    if not auto:

                        #hypothetically, have 150a, 150b, 220a,220b 
                        #want to end with:
                        # 150a * 220b + 150b * 220a
                        #iew 1j* 2k + 2j* 1k
                        tmpresult  =np.sum(np.real(banddata[revsetdef1[:, j],:]*np.conj(banddata[revsetdef2[:, k],:])), 1,dtype=np.float64)
                        tmpresult +=np.sum(np.real(banddata[revsetdef2[:, j],:]*np.conj(banddata[revsetdef1[:, k],:])), 1,dtype=np.float64)
                        tmpresult /= (2*nmodes)
                        
                        allspectra_out[iprime, spectrum_idx, :]=tmpresult.astype(np.float32)

                    else:
                        #j/k are freqs
                        #first index is nrealizations
                        tmpresult=np.sum(np.real(banddata[revsetdef1[:, j],:]*np.conj(banddata[revsetdef1[:, k],:])), 1,dtype=np.float64) / (nmodes) # had been in flatsky: *reso^2*winsize^2)
                        #tmpresult is nrealizations long
                        allspectra_out[iprime, spectrum_idx, :]=tmpresult.astype(np.float32)
                    spectrum_idx+=1
                    #           pdb.set_trace()
        i=istop
    return(allspectra_out, nmodes_out)



def process_all_cross_spectra(allspectra, nbands, nsets,setsize, 
                              auto=False,
                              skipcov=False ) -> 'Returns mean and covarariance estimates':

    print("Correlating Cross Spectra")
    nspectra = int( (nsets * (nsets+1))/2 + 0.001)




    if auto:
        nrealizations = setsize
    else:
        nrealizations=int( (setsize*(setsize-1))/2 + 0.001)

    allspectra = np.reshape(allspectra, [nbands*nspectra, nrealizations])
    #cov  = np.zeros([nbands*nspectra, nbands*nspectra],dtype=np.float64)

    spectrum = np.sum(allspectra,-1,dtype=np.float64)/nrealizations

    spectrum = np.reshape(spectrum,[nbands,nspectra])
    if skipcov:
        return spectrum,None,None,None
    # [nbands*nspectra, nrealizations])
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
        #nbands*nspectra, setsize)
        spectrum_2d=np.tile(np.reshape(spectrum,[nbands*nspectra,1]), [1,setsize])

        cov2=np.matmul( (allcomplementspectra-spectrum_2d), (allcomplementspectra-spectrum_2d).T )
        cov2/=(setsize**2 / 2)
        cov=2*cov2-cov1

    else:
        cov=cov1*(nrealizations) 

    return spectrum,cov,cov1,cov2

'''
Create a class instance to simplify storing all the arguments along with the output
'''
class unbiased_multispec:
    def __init__(self,
                 # Maps/SHT flags ################################################
                 mapfile, #required -array of map filenames, g3 format
                 window, # required -- mask to apply for SHT
                 banddef, # required. [0,lmax_bin1, lmax_bin2, ...]
                 nside, #required. eg 8192
                 lmax=None, #optional, but should be set. Defaults to 2*nside      
                 cmbweighting=True, # True ==> return Dl. False ==> Return Cl
                 kmask = None, #If not none, must be the right size for the Alms. A numpy array/vector
                 setdef=None, # optional -- will take from mapfile array dimensions if not provided
                 setdef2 = None, #optional -- if provided will assume doing sim cross-spectra
                 jackknife = False, #If true, will difference SHTs to do null spectrum
                 auto=False, #If true will do autospectra instead of cross-spectra
                 apply_windowfactor = True, #if true, calculate and apply normalization correction for partial sky mask. 
                 map_key = 'T', #where to fetch maps from
                 skipcov=False, #don't calculate covariances
                 # Run time processing flags ################################################
                 ramlimit=64 * 2**30, # optional -- set to change default RAM limit from 64gb
                 resume=True, #optional -- will use existing files if true    
                 basedir=None, # strongly suggested. defaults to current directory and can use a lot of disk space
                 persistdir=None, # optional - can be unset. will create a temp directory within basedir
                 remove_temporary_files= False, # optional. Defaults to off (user has to do cleanup, but can restart runs later)
                 verbose = False ): #extra print statements
                #maybe sometime I'll put in more input file arguments...                  
        '''
                 # Outputs ################################################
                 allspectra -- array of all cross-spectra (binned according to banddef)
                 cov -- array estimated covariance
                 est1_cov -- array estimated covariance from estimator 1
                 est2_cov -- array estimated covariance from estimator 2
                 nmodes -- array of number of alms per bandpower bin (form banddef)
                 windowfactor -- value used to normalize spectrum for apodization window. May be 1 (ie not corrected)
        '''
        self.mapfile = mapfile
        self.window = window
        self.banddef = banddef
        self.nside = nside
        self.lmax = lmax
        if self.lmax is None: 
            self.lmax = 2*self.nside
        self.cmbweighting = cmbweighting
        self.kmask = kmask
        self.setdef = setdef
        self.jackknife = jackknife
        self.auto = auto
        self.apply_windowfactor = apply_windowfactor
        self.ramlimit = ramlimit
        self.resume = resume
        self.basedir = basedir
        self.persistdir = persistdir
        self.remove_temporary_files = remove_temporary_files
        self.verbose = verbose
        self.allspectra = None
        self.spectrum = None
        self.cov = None
        self.est1_cov = None
        self.est2_cov = None
        self.nmodes = None
        self.windowfactor = 1.0
                
                
        #################
        # figure out scratch directories
        #################
        try:
            if not os.path.isdir(self.basedir):
                raise TypeError  # to be caught below
        except TypeError:            
            self.basedir = os.getcwd()
        try:     
            if os.path.exists(self.persistdir) and not os.path.isdir(self.persistdir):
                print("WARNING -- Requested scratch exists, but is not a directory: {}".format(self.persistdir))
                raise TypeError
        except TypeError:
            self.persistdir = name_tempdir(self.basedir)
            print("WARNING -- using {} for scratch".format(self.persistdir))
            if not os.path.isdir(self.persistdir):
                os.makedirs(self.persistdir)

        #maybe at some point, we'll use status. right now nothing is done. Resume will only affect the full step level - no partial steps yet.
        status_file = self.persistdir + '/status.pkl'
        
        processed_sht_file = self.persistdir + '/shts_processed.bin'
        if not self.resume:
            try: 
                os.remove(processed_sht_file)
            except FileNotFoundError:
                pass
        
        
        #################
        # Figure out set def based on structure of map file names, if not provided
        #################
        if self.setdef is None:
            #may need to change this -- unsure if it's right or transpose
            #may also need to make it 2d
            #remove warning printout when checked
            self.setdef = self.mapfile.shape
            print('Warning - check set def: inferred {}'.format(self.setdef))
        
        #get SHTs done
        sht_size = os.path.getsize(processed_sht_file)
        desired_size = healpy.sphtfunc.Alm.getsize(lmax) * np.zeros(1,dtype=AlmType).nbytes
        if (sht_size < desired_size):  #this will be false if resume==False since deleted file above.
            print("Dont expect to be here")
            pdb.set_trace()
            take_and_reformat_shts(self.fuile, processed_sht_file,
                   self.nside,self.lmax,
                   cmbweighting = self.cmbweighting, 
                   mask  = self.window,
                   kmask = self.kmask,
                   ell_reordering=None,
                   no_reorder=False,
                   ram_limit = self.ramlimit, 
                   map_key=map_key
                  )
        
        use_setdef  = setdef
        use_shtfile = processed_sht_file
        if self.jackknife:
            jackknife_sht_file = self.persistdir + '/null_shts_processed.bin'
            use_setdef = generate_jackknife_shts( processed_sht_file, jackknife_sht_file,  self.lmax, self.setdef)
            use_shtfile = jackknife_sht_file

        self.use_setdef = use_setdef
        
        #figure out cross-spectra (or autospectra)
        if setdef2 is None:
            allspectra, nmodes= take_all_cross_spectra( use_shtfile, self.lmax,
                                                        self.use_setdef, self.banddef,  ram_limit=self.ramlimit, auto = self.auto) #-> 'Returns set of all x-spectra, binned':
        else:
            allspectra, nmodes= take_all_sim_cross_spectra( use_shtfile, self.lmax,
                                                        self.use_setdef,self.banddef, setdef2=setdef2, ram_limit=self.ramlimit, auto = self.auto) #-> 'Returns set of all x-spectra, binned':
                        
        self.allspectra = allspectra
        self.nmodes = nmodes
        
        
        #bring it all together
        nbands = banddef.shape[0]-1
        nsets   = use_setdef.shape[1]
        setsize = use_setdef.shape[0]

        process_auto = self.auto or (setdef2 is not None) # only get 1 per set for the sim crosses too
        spectrum,cov,cov1,cov2 = process_all_cross_spectra(self.allspectra, nbands, nsets,setsize, 
                                                            auto=process_auto)
        self.spectrum = spectrum
        self.cov      = cov
        self.est1_cov = cov1
        self.est2_cov = cov2
                                 
