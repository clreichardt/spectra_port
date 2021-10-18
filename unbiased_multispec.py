


# holder class
class UnbiasedMultisetPspec:


    def __init__(self) -> 'Stores stuff for x-spectra':
        pass



    def take_and_reformat_shts(mapfile, processedshtfile,
                               nside,lmax,
                               cmbweighting = True, 
                               kmask = None
                              ) -> 'May be done in Fortran - output is a file':
        pass

    
    def generate_jackknife_shts( processedshtfile, jackknifeshtfile, nside, lmax
                                 setdef) -> 'Does differencing to make SHT equiv file for nulls':
        pass


    def take_all_cross_spectra( processedshtfile, nside, lmax,
                                setdef, banddef, auto = False) -> 'Returns set of all x-spectra, binned':
        pass

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
