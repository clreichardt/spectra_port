import numpy as np
import glob
import os
from spectra_port import unbiased_multispec as spec
from spectra_port import utils
from spectra_port import end_to_end

PREP= False
END = False
NULL= True




def create_real_file_list(dir,stub='GHz_bundle_',sfreqs=['90','150','220'],estub='.npz',nbundle=200):
    nfreq=len(sfreqs)
    
    #file_list = np.zeros(6*nsim,dtype='<U265') #max len 256
    '''
    desired output order (for 200 bundles)
      0-199: 90 bundleA

      200-399: 150 bundleA

      400-599: 220 bundleA

    '''
    maxlen    = len(os.path.join(dir, sfreqs[2]+stub+'{:d}'.format(nbundle-1)+estub))+1
    file_list = np.zeros(3*nbundle,dtype='<U{:d}'.format(maxlen)) 

    for j in range(nfreq):
        for i in range(nbundle):
            file_list[j*nbundle + i]     = os.path.join(dir, sfreqs[j]+stub+'{:d}'.format(i)+estub)
            
    return file_list

def create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=100):
    nfreq=len(sfreqs)
    listA  = []
    listB  = []
    
    #file_list = np.zeros(6*nsim,dtype='<U265') #max len 256
    bundle_list = np.zeros([nsim,2],dtype=np.int)
    '''
    desired output order (for 100 sims)
      0-99: 90 bundleA
      100-199: 90 bundle B
      200-299: 150 bundleA
      300-399: 150 bundleB
      400-499: 220 bundleA
      500-599: 220 bundleB
    '''
    maxlen = 0
    for i in range(nsim):
        files = glob.glob(os.path.join(dir, dstub.format(i),bstub+'*_'+sfreqs[0]+estub))
        if len(files[0])+2 > maxlen:
            maxlen = len(files[0])+2
        if len(files[1])+2 > maxlen:
            maxlen = len(files[1])+2
        endlength= len(sfreqs[0]+estub)+1 # for '_'
        beglength = len(os.path.join(dir, dstub.format(i),bstub))

        bundle_list[i,0] = int(files[0][beglength:-endlength])
        bundle_list[i,1] = int(files[0][beglength:-endlength])

    file_list = np.zeros(6*nsim,dtype='<U{:d}'.format(maxlen)) 

    for i in range(nsim):
        for j in range(nfreq):
            file_list[j*2*nsim + i]     = os.path.join(dir, dstub.format(i),bstub+'{:03d}_'.format(bundle_list[i,0])+sfreqs[j]+estub)
            file_list[(j*2+1)*nsim + i] = os.path.join(dir, dstub.format(i),bstub+'{:03d}_'.format(bundle_list[i,1])+sfreqs[j]+estub)
            
    return file_list

def create_sim_setdefs(nsim,nfreq):
    ''' assumes 2 bundles per'''
    
    set1 = np.zeros([nsim,nfreq],dtype=np.int32)
    set2 = np.zeros([nsim,nfreq],dtype=np.int32)

    for i in range(nfreq):
        set1[:,i] = np.arange(0,nsim) + 2*i*nsim
        set2[:,i] = np.arange(0,nsim) + (2*i+1)*nsim
    return set1, set2


if __name__ == "__main__" and PREP is True:
    print("First sims")
    workdir = '/sptlocal/user/creichardt/xspec_2022/'
    lmax = 13000
    dir='/sptgrid/analysis/highell_TT_19-20/v3/mockobs/v1_2bundles/'
    '''
    mcshtfilelist = create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=100)
    processedshtfile = workdir + '/mc/shts_processed.bin'
    spec.reformat_shts(mcshtfilelist, processedshtfile,
                           lmax,
                           cmbweighting = True, 
                           mask  = None,
                           kmask = None,
                           ell_reordering=None,
                           no_reorder=False,
                           ram_limit = None,
                          )
    '''
    print("Now real")
    #    exit()

    datashtfilelist = create_real_file_list('/sptgrid/user/pc/obs_shts/',stub='GHz_bundle_',sfreqs=['90','150','220'],estub='.npz',nbundle=200)
    processedshtfile = workdir + '/data/shts_processed.bin'
    spec.reformat_shts(datashtfilelist, processedshtfile,
                           lmax,
                           cmbweighting = True, 
                           mask  = None,
                           kmask = None,
                           ell_reordering=None,
                           no_reorder=False,
                           ram_limit = None,
                          ) 


if __name__ == "__main__" and END == True:
    
    banddef = np.arange(0,11000,100)
    #banddef = np.asarray([0,1000,1500,2000,2200,2500,2800,3100,3400,3700,4000,4400,4800,5200,5700,6200,6800,7400,8000,9000,10000,11000,12000,13000])

    setdef_mc1, setdef_mc2 = create_sim_setdefs(100,3)

    setdef = np.zeros([200,3],dtype=np.int32)
    setdef[:,0]=np.arange(200)
    setdef[:,0]=np.arange(200)+200
    setdef[:,0]=np.arange(200)+400
    #nsets   = setdef.shape[1] #nfreq
    #setsize = setdef.shape[0] #nbundles
    
    #note beam is 90, 150, 220, so everything else needs to be too (or change beam array ordering)
    beam_arr = np.loadtxt('/home/creichardt/spt3g_software/beams/products/compiled_2020_beams.txt')
    

 #   >>> namaster_file='/sptlocal/user/pc/mll/mll_tpltz.npy'
#>>> kernel_file = '/sptlocal/user/creichardt/mll_dl_13000.npz'
#>>> utils.rebin_and_convert_namaster_mll(namaster_file,kernel_file,5,13000)
    kernel_file = '/sptlocal/user/creichardt/mll_dl_13000.npz'

    workdir = '/sptlocal/user/creichardt/xspec_2022/'
    file_out = workdir + 'spectrum.pkl'
    
    mask_file='/home/pc/hiell/mapcuts/apodization/apod_mask.npy'
    mask = np.load(mask_file)
    
    #may need to reformat theoryfiles
    theoryfiles = ['/home/pc/hiell/sims/90ghz_input_spectrum.txt',
                    '/home/pc/hiell/sims/150ghz_input_spectrum.txt',
                    '/home/pc/hiell/sims/220ghz_input_spectrum.txt']
    mapfiles = create_real_file_list('/sptgrid/user/pc/obs_shts/',stub='GHz_bundle_',sfreqs=['90','150','220'],estub='.npz',nbundle=200)
    dir='/sptgrid/analysis/highell_TT_19-20/v3/mockobs/v1_2bundles/'
    mcmapfiles = create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=100)
    
    
    output = end_to_end.end_to_end( mapfiles,
                         mcmapfiles,
                         banddef,
                         beam_arr,
                         theoryfiles,
                         workdir,
                         simbeam_arr=None,
                         setdef=setdef,
                         setdef_mc1=setdef_mc1,
                         setdef_mc2=setdef_mc2,
                         do_window_func=False, 
                         lmax=13000,
                         cl2dl=True,
                         nside=8192,
                         kmask=None,
                         mask=mask,
                         kernel_file =kernel_file,
                         resume=True
                       )
    with open(file_out,'wb') as fp:
        pkl.save(output,fp)
    
if __name__ == "__main__" and NULL == True:
    mask_file='/home/pc/hiell/mapcuts/apodization/apod_mask.npy'
    mask = np.load(mask_file)
    nside=8192
    banddef = np.arange(0,11000,500)
    workdir90='/big_scratch/cr/null90/'
    setdef = np.zeros([100,2],dtype=np.int32)
    setdef[:,0]=np.arange(0100,dtype=np.int32)
    setdef[:,1]=np.arange(100,200,dtype=np.int32)
    mapfiles = create_real_file_list('/sptgrid/user/pc/obs_shts/',stub='GHz_bundle_',sfreqs=['90','150','220'],estub='.npz',nbundle=200)
    null_spectrum_90      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir90,
                                              persistdir=workdir90,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum.pkl'
    with open(file_out,'wb') as fp:
        pkl.save(null_spectrum_90,fp)