import sys
import numpy as np

AlmType = np.complex64

def printinplace(myString):
    digits = len(myString)
    delete = "\b" * (digits)
    print("{0}{1:{2}}".format(delete, myString, digits), end="")
    sys.stdout.flush()

# holder class
class UnbiasedMultisetPspec:


    def __init__(self) -> 'Stores stuff for x-spectra':
        pass



    def take_and_reformat_shts(mapfile, processedshtfile,
                               nside,lmax,
                               cmbweighting = True, 
                               mask  = None,
                               kmask = None,
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

        #ie do parallelism SHTs at once...
        size = healpy.sphtfunc.Alm.getsize(lmax)
        with open(processedshtfile,'wb') as fp:

            for i in range(nmap):
                #TBD get a map

                # if not already masked, apply mask
                #may need to change the next line based on formatting
                map  = mask*map

                #gets alms
                alms = healpy.sphtfunc.map2alm(map,lmax = lmax, pol=False, use_pixel_weights=True, verbose=False)

                #apply weighting:
                alms = alms * kmask

                #TBD, possibly adjust for mask factor here

                #write to disk
                #need to check sizing
                #32 bit floats/64b complex should be fine for this. will need to bump up by one for aggregation
                ((alms.astype(AlmType)).tofile(fp)

        pass

    
    def generate_jackknife_shts( processed_shtfile, jackknife_shtfile,  lmax
                                 setdef) -> 'Does differencing to make SHT equiv file for nulls':
        buffersize=healpy.sphtfunc.Alm.getsize(lmax)
        buffera = np.zeros(buffersize,dtype=AlmType)
        bufferb = np.zeros(buffersize,dtype=AlmType)
        setsize = setdef.shape[0]
        nsets   = setdef.shape[1]

        with open(processed_shtfile,'rb') as fin, open(jackknife_shtfile,'wb') as fout:
            for i in range(setsize):
                j = setdef[i,0]
                k = setdef[i,1]
                #need to do stuff here


for i=0, setsize-1 do begin
    ; read the two relevant files
    point_lun, lun_in, setdef[i, 0]*ulong64(winsize)^2 * 16
    readu, lun_in, buffera
    point_lun, lun_in, setdef[i, 1]*ulong64(winsize)^2 * 16
    readu, lun_in, bufferb
    mapout=(buffera-bufferb)/2
    writeu, lun_out, mapout
endfor
free_lun, lun_in
free_lun, lun_out
setdef=reform(lindgen(setsize), setsize, 1)
end

        pass


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
        nspectra=(nsets*(nsets+1))/2
        if auto:
            nrealizations=setsize
        else:
            nrealizations=(setsize*(setsize-1))/2

        nbands = banddef.shape[0]-1
        nshts  = np.max(setdef)+1

        allspectra_out = np.zeros([nbands,nspectra,nrealizations],dtype=np.float64)
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
            for iprime=range(i, istop):
                printinplace('processing band {}    '.format(iprime))
                    '
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
                            for l=range(setsize-1):
                                rowlength=setsize-l-1
                                allspectra_out[iprime, spectrum_idx, a:(a+rowlength-1)]=tmpresult[l, l+1:setsize-1]
                                a+=rowlength
                        else:
                            idx=lindgen(setsize)
                            tmpresult=np.sum(/double,real_part(banddata[setdef[:, j],:]*conj(banddata[setdef[:, k],:])), 1)/(nmodes) # had been in flatsky: *reso^2*winsize^2)
                            allspectra_out[iprime, spectrum_idx, :]=tmpresult
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
            nrealizations=(1l*setsize*(setsize-1))/2

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
                    for j = range(j+1,setsize):
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
