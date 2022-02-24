import os
os.environ['OMP_NUM_THREADS'] = "6"

import unbiased_multispec as spec
import numpy as np
from spt3g import core,maps, calibration



def combine_lr_bundles():
    nf=200
    idir='/sptgrid/analysis/highell_TT_19-20/v3/mockobs/inputsky000/bundles/'
    odir='/sptgrid/analysis/highell_TT_19-20/v3/mockobs/inputsky000/bundles/sum'
    for i in range(nf):
        print(i,nf)
        ifile = idir + 'bundle{:03d}_150GHz.g3.gz'.format(i)
        ofile = odir + 'bundle{:03d}_150GHz.g3.gz'.format(i)
        #hpmap = list(core.G3File('/sptgrid/analysis/eete+lensing_19-20/v2/data_maps/ra0hdec-44.75/healpix_105499116_150GHz.g3'))
        hpmap = list(core.G3File(ifile))    
        sum = 0.5*(hpmap[0]["T"]+hpmap[1]["T"]) #L+R/2
        
        frame = core.G3Frame(core.G3FrameType.Map)
        frame['tmap'] = sum
        writer= core.G3Writer(ofile)
        writer(frame)
    
    
    
    
def test_n_bundles():



    banddef = np.arange(0,11000+1,50)
    #[0,500,1000,1500,2000,2200,2500,2800,3100,3500,4000,4600,5200,6000,6800,7800,9000,11
    mcdir = workdir + '/mc/'
    mc_specrum = spec.unbiased_multispec(mapfiles, window, banddef, nside,
                                         lmax=lmax,
                                         resume=resume,
                                         basedir=mcdir,
                                         setdef=setdef_mc,
                                         jackknife=False, auto=True,
                                         kmask=None,
                                         cmbweighting=True)


def do_auto_spectra_firstN_bundles(Nuse,process_sht_file='/sptlocal/user/creichardt/hiell2022/sht_sim1_lmax100.bin',
                                    banddef=np.arange(0,101,50),lmax=100):
    tmp_sht = process_sht_file + '_auto{}'.format(Nuse)
    generate_coadd_shts(process_sht_file, tmp_sht,lmax,setdef)
    return do_cross_spectra_firstN_bundles(Nuse,process_sht_file)
                                           
def do_cross_spectra_firstN_bundles(Nuse, process_sht_file='/sptlocal/user/creichardt/hiell2022/sht_sim1_lmax100.bin',
                                    banddef=np.arange(0,101,50),lmax=100,
                                    auto=False):
    setdef = np.arange(0,Nuse)
    
        
    #figure out cross-spectra (or autospectra)
    allspectra, nmodes= take_all_cross_spectra( process_sht_file, lmax,
                                                setdef, banddef,auto=auto) #-> 'Returns set of all x-spectra, binned':
    allspectra = allspectra
    nmodes = nmodes
    
    
    #bring it all together
    nbands = banddef.shape[0]-1
    nsets   = use_setdef.shape[1]
    setsize = use_setdef.shape[0]

    spectrum,cov,cov1,cov2 = process_all_cross_spectra(allspectra, nbands, nsets,setsize)
    results = {}
    results['allspectra']=allspectra
    results['nmodes']=nmodes
    results['nbands']=nbands
    results['nsets']=nsets
    results['setsize']=setsize
    results['spectrum']=spectrum
    results['cov']=cov
    results['cov1']=cov1
    results['cov2']=cov2
    
def sht_bundles():
    mask = np.load('/home/pc/hiell/mapcuts/apodization/apod_mask.npy')
    nmap=200
    nside=8192
    lmax=13000 # test
    file_list = ['/sptgrid/user/ndhuang/high_ell/xfer_specs/{:03d}.npz'.format(i) for i in range(nmap)]
    processed_sht_file = '/sptlocal/user/creichardt/hiell2022/sht_sim1_lmax13000.bin'
    '''spec.take_and_reformat_shts(file_list, processed_sht_file,
                           nside,lmax,
                           cmbweighting = True, 
                           mask  = mask,
                           kmask = None,
                           ell_reordering=None,
                           no_reorder=False,
                           ram_limit = None,
                           npmapformat=False, 
                           map_key='coadd'
                          )
    '''
    spec.reformat_shts(file_list, processed_sht_file,
                           nside,lmax,
                           cmbweighting = True, 
                           mask  = mask,
                           kmask = None,
                           ell_reordering=None,
                           no_reorder=False,
                           ram_limit = None
                          )   


if __name__ == "__main__":
    sht_bundles()
    #combine_lr_bundles()
    #test_n_bundles()
