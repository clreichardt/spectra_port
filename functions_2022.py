import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import glob

import unbiased_multispec as spec
import utils
import end_to_end
from spt3g import core,maps, calibration
import argparse
import pickle as pkl

PREP= False
END = False
NULL= False
COADD= False
SHT = False
CAL = False
TEST= False
NULLLR=False

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-prep', action='store_true',dest='prep')
my_parser.add_argument('-end', action='store_true',dest='end')
my_parser.add_argument('-null', action='store_true',dest='null')
my_parser.add_argument('-nulllr', action='store_true',dest='nulllr')
my_parser.add_argument('-coadd', action='store_true',dest='coadd')
my_parser.add_argument('-sht', action='store_true',dest='sht')
my_parser.add_argument('-cal', action='store_true',dest='cal')
my_parser.add_argument('-test', action='store_true',dest='test')
#my_parser.add_argument('-freq', default=None ,dest='freq')
args = my_parser.parse_args()

PREP=args.prep
END=args.end
NULL=args.null
COADD=args.coadd
CAL = args.cal
SHT = args.sht
TEST= args.test
NULLLR=args.nulllr


def create_bundle_maps_and_coadds(freq,nbundles=200):
    if freq == 150:
        dir='/sptgrid/user/pc/bundle_coadds/150GHz/'
        filestub = '150GHz_bundle_{}.g3'
        fileout = 'coadd_150ghz.pkl'
    if freq == 90:
        dir='/sptgrid/user/pc/bundle_coadds/90GHz/'
        filestub = '90GHz_bundle_{}.g3'
        fileout = 'coadd_90ghz.pkl'
    if freq == 220:
        dir='/sptgrid/user/pc/bundle_coadds/220GHz/'
        filestub = '220GHz_bundle_{}.g3'
        fileout = 'coadd_220ghz.pkl'
    ofilestub = filestub[:-3]+'.pkl'
    dirout = '/sptlocal/user/creichardt/hiell2022/bundles/'
    
    nside=8192
    coadd = np.zeros(12*nside**2,dtype=np.float32)
    for i in range(nbundles):
        a=list(core.G3File(dir+filestub.format(i)))
        loc_ind,loc_map = (a[0]['left']+a[0]['right']).nonzero_pixels()
        loc_ind,loc_wt = (a[0]['weight']).nonzero_pixels()
        loc_ind = np.asarray(loc_ind,dtype=np.int32)
        loc_map = np.asarray((loc_map/loc_wt),dtype=np.float32)
        with open(dirout+ofilestub.format(i),'wb') as fp:
            pkl.dump(loc_ind,fp)
            pkl.dump(loc_map,fp)
        coadd[loc_ind]+= loc_map
    ind = np.asarray(coadd.nonzero(),dtype=np.int32)
    store = (coadd[ind]/nbundles).astype(np.float32)
    with open(dirout+fileout,'wb') as fp:
        pkl.dump(ind,fp)
        pkl.dump(store,fp)
    

def create_real_file_list(dir, stub='bundle_',sfreqs=['90','150','220'],estub='GHz.npz', nbundle=200):
    nfreq=len(sfreqs)
    
    '''
    desired output order (for 200 bundles)
      0-199: 90 bundleA

      200-399: 150 bundleA

      400-599: 220 bundleA

    '''
    file_list = np.zeros(nfreq*nbundle,dtype='<U265') 

    for j in range(nfreq):
        for i in range(nbundle):
            file_list[j*nbundle + i]     = os.path.join(dir, stub+'{:d}'.format(i)+'_'+sfreqs[j]+estub)
    return file_list

def create_real_file_list_v3(dir, stub='GHz_bundle_',sfreqs=['90','150','220'],estub='.npz', nbundle=200):
    nfreq=len(sfreqs)
    
    '''
    desired output order (for 200 bundles)
      0-199: 90 bundleA

      200-399: 150 bundleA

      400-599: 220 bundleA

    '''
    file_list = np.zeros(nfreq*nbundle,dtype='<U265') 

    for j in range(nfreq):
        for i in range(nbundle):
            file_list[j*nbundle + i]     = os.path.join(dir, sfreqs[j]+stub+'{:d}'.format(i)+estub)
    return file_list


def create_real_file_list_v4(dir, stub='GHz_bundle_',sfreqs=['90','150','220'],estub='.npz', nbundle=200):
    nfreq=len(sfreqs)
    
    '''
    desired output order (for 200 bundles)
      0-199: 90 bundleA

      200-399: 150 bundleA

      400-599: 220 bundleA

    '''
    file_list = np.zeros(nfreq*nbundle,dtype='<U265') 

    for j in range(nfreq):
        for i in range(nbundle):
            file_list[j*nbundle + i]     = os.path.join(dir, stub+'{:d}_'.format(i)+sfreqs[j]+estub)
    return file_list

def create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=100):
    nfreq=len(sfreqs)
    listA  = []
    listB  = []
    
    #file_list = np.zeros(6*nsim,dtype='<U265') #max len 256
    bundle_list = np.zeros([nsim,2],dtype=np.int32)
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
    workdir = '/big_scratch/cr/xspec_2022/'
    lmax = 13000
    dir='/sptgrid/analysis/highell_TT_19-20/v4/mockobs/v2.0_testinputsv2/'
#/sptgrid/analysis/highell_TT_19-20/v4/mockobs/v1_2bundles/'
    

    if True:
        mcshtfilelist = create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=10)
        print(mcshtfilelist)        
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
        
    print("Now real")
    if False:
    #    exit()
        dir='/sptgrid/analysis/highell_TT_19-20/v4/obs_shts/'
        datashtfilelist = create_real_file_list(dir,stub='bundle_',sfreqs=['90','150','220'],estub='GHz.npz',nbundle=200)
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
    
    banddef = np.arange(0,13000,50)
    #banddef = np.asarray([0,1000,1500,2000,2200,2500,2800,3100,3400,3700,4000,4400,4800,5200,5700,6200,6800,7400,8000,9000,10000,11000,12000,13000])

    #change for testing
    #setdef_mc1, setdef_mc2 = create_sim_setdefs(100,3)
    setdef_mc1, setdef_mc2 = create_sim_setdefs(10,3)

    setdef = np.zeros([200,3],dtype=np.int32)
    setdef[:,0]=np.arange(200,dtype=np.int32)
    setdef[:,1]=np.arange(200,dtype=np.int32)+200
    setdef[:,2]=np.arange(200,dtype=np.int32)+400
    #nsets   = setdef.shape[1] #nfreq
    #setsize = setdef.shape[0] #nbundles
    
    #note beam is 90, 150, 220, so everything else needs to be too (or change beam array ordering)
    beam_arr = np.loadtxt('/home/creichardt/spt3g_software/beams/products/compiled_2020_beams.txt')
    
    kernel_file = '/sptlocal/user/creichardt/mll_dl_13000.npz'

    #workdir = '/sptlocal/user/creichardt/xspec_2022/'
    workdir = '/big_scratch/cr/xspec_2022/'
    file_out = workdir + 'spectrum.pkl'
    file_out_small = workdir + 'spectrum_small.pkl'
    
    mask_file='/home/pc/hiell/mapcuts/apodization/apod_mask.npy'
    mask = np.load(mask_file)
    
    #may need to reformat theoryfiles
    theoryfiles = ['/sptlocal/user/creichardt/hiell2022/sim_dls_90ghz.txt',
                   '/sptlocal/user/creichardt/hiell2022/sim_dls_150ghz.txt',
                   '/sptlocal/user/creichardt/hiell2022/sim_dls_220ghz.txt']

    
    dir='/sptgrid/analysis/highell_TT_19-20/v4/obs_shts/'
    mapfiles = create_real_file_list(dir,stub='bundle_',sfreqs=['90','150','220'],estub='GHz.npz',nbundle=200)

    #change for testing
    dir='/sptgrid/analysis/highell_TT_19-20/v4/mockobs/v2.0_testinputsv2/'
    mcmapfiles = create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=10)
    #dir='/sptgrid/analysis/highell_TT_19-20/v4/mockobs/v1_2bundles/'
    #mcmapfiles = create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=100)
    
    
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
#                         cl2dl=True,
                         nside=8192,
                         kmask=None,
                         mask=mask,
                         kernel_file =kernel_file,
                         resume=True, 
                         checkpoint=True
                       )
    with open(file_out,'wb') as fp:
        pkl.dump(output,fp)
    with open(file_out_small,'wb') as fp:
        pkl.dump(end_to_end.trim_end_to_end_output(output),fp)



if __name__ == "__main__" and NULLLR == True:

    print('doing null')

    mask_file='/home/pc/hiell/mapcuts/apodization/apod_mask.npy'
    mask = np.load(mask_file)
    nside=8192
    banddef = np.arange(0,12000,500)
    os.system('ln -s /big_scratch/pc/lr_null/sht_lr_90.bin /big_scratch/cr/xspec_2022/datalr/shts_processed.bin')
    workdir='/big_scratch/cr/xspec_2022/datalr/'
    setdef = np.zeros([200,1],dtype=np.int32)
    setdef[:,0]=np.arange(0,200,dtype=np.int32)
    mapfiles = create_real_file_list('/sptgrid/user/pc/obs_shts/',stub='GHz_bundle_',sfreqs=['90','150','220'],estub='.npz',nbundle=200)
    spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                            lmax=13000,
                                            resume=True,
                                            basedir=workdir,
                                            persistdir=workdir,
                                            setdef=setdef,
                                            jackknife=False, auto=False,
                                            kmask=None,
                                            cmbweighting=True)
    file_out = workdir + 'spectrum90_lrnull.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(spectrum,fp)
        del spectrum

    os.system('ln -s /big_scratch/pc/lr_null/sht_lr_150.bin /big_scratch/cr/xspec_2022/datalr/shts_processed.bin')
    spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                            lmax=13000,
                                            resume=True,
                                            basedir=workdir,
                                            persistdir=workdir,
                                            setdef=setdef,
                                            jackknife=False, auto=False,
                                            kmask=None,
                                            cmbweighting=True)
    file_out = workdir + 'spectrum150_lrnull.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(spectrum,fp)
        del spectrum
    
    os.system('ln -s /big_scratch/pc/lr_null/sht_lr_220.bin /big_scratch/cr/xspec_2022/datalr/shts_processed.bin')
    spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                            lmax=13000,
                                            resume=True,
                                            basedir=workdir,
                                            persistdir=workdir,
                                            setdef=setdef,
                                            jackknife=False, auto=False,
                                            kmask=None,
                                            cmbweighting=True)
    file_out = workdir + 'spectrum220_lrnull.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(spectrum,fp)
        del spectrum
    
    
if __name__ == "__main__" and NULL == True:

    print('doing null')

    mask_file='/home/pc/hiell/mapcuts/apodization/apod_mask.npy'
    mask = np.load(mask_file)
    nside=8192
    banddef = np.arange(0,12000,500)

    workdir='/big_scratch/cr/xspec_2022/data/'
    setdef = np.zeros([200,1],dtype=np.int32)
    setdef[:,0]=np.arange(0,200,dtype=np.int32)
    mapfiles = create_real_file_list('/sptgrid/user/pc/obs_shts/',stub='GHz_bundle_',sfreqs=['90','150','220'],estub='.npz',nbundle=200)
    if False:
        spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=False, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
        file_out = workdir + 'spectrum90_nullbins.pkl'
        with open(file_out,'wb') as fp:
            pkl.dump(spectrum,fp)
        del spectrum
    setdef[:,0]+=200
    spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=False, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'spectrum150_nullbins.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(spectrum,fp)
    del spectrum
    setdef[:,0]+=200
    spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=False, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'spectrum220_nullbins.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(spectrum,fp)
    del spectrum
    setdef = np.zeros([45,2],dtype=np.int32)
    setdef[:,0]=np.arange(0,45,dtype=np.int32)
    setdef[:,1]=np.arange(45,90,dtype=np.int32)
    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_90_year1.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)
    del null_spectrum
    setdef[:,0]=np.arange(0,45,dtype=np.int32)+110
    setdef[:,1]=np.arange(45,90,dtype=np.int32)+110
    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_90_year2.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)
    del null_spectrum        
        
        
    setdef = np.zeros([45,2],dtype=np.int32)
    setdef[:,0]=np.arange(0,45,dtype=np.int32)+200
    setdef[:,1]=np.arange(45,90,dtype=np.int32)+200
    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_150_year1.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)
    del null_spectrum
    setdef[:,0]=np.arange(0,45,dtype=np.int32)+110+200
    setdef[:,1]=np.arange(45,90,dtype=np.int32)+110+200
    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_150_year2.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)
    del null_spectrum
    
    setdef[:,0]=np.arange(0,45,dtype=np.int32)+400
    setdef[:,1]=np.arange(45,90,dtype=np.int32)+400
    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_220_year1.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)
    del null_spectrum
    setdef[:,0]=np.arange(0,45,dtype=np.int32)+110+400
    setdef[:,1]=np.arange(45,90,dtype=np.int32)+110+400
    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_220_year2.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)
    del null_spectrum

    setdef = np.zeros([100,2],dtype=np.int32)
    setdef[:,0]=np.arange(0,100,dtype=np.int32)
    setdef[:,1]=np.arange(100,200,dtype=np.int32)
    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_90.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)




    setdef = np.zeros([100,2],dtype=np.int32)
    setdef[:,0]=np.arange(0,100,dtype=np.int32)+200
    setdef[:,1]=np.arange(100,200,dtype=np.int32)+200

    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_150.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)

    setdef = np.zeros([100,2],dtype=np.int32)
    setdef[:,0]=np.arange(0,100,dtype=np.int32)+400
    setdef[:,1]=np.arange(100,200,dtype=np.int32)+400

    null_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=13000,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=True, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
    file_out = workdir + 'null_spectrum_220.pkl'
    with open(file_out,'wb') as fp:
        pkl.dump(null_spectrum,fp)


if __name__ == "__main__" and COADD == True:
    create_bundle_maps_and_coadds(90,nbundles=200)
    create_bundle_maps_and_coadds(150,nbundles=200)
    create_bundle_maps_and_coadds(220,nbundles=200)



if __name__ == "__main__" and SHT == True:
    subfield='ra0hdec-44.75'
    #subfield='ra0hdec-52.25'
    #subfield='ra0hdec-59.75'
    #subfield='ra0hdec-67.25'
    calworkdir = '/big_scratch/cr/xspec_2022/cal/'+subfield+'/'
    #os.makedirs(calworkdir+'data/',exist_ok=True)
    print(calworkdir)
    if True:
        dir='/sptlocal/user/creichardt/hiell2022/bundle10/'
        rlist = create_real_file_list_v4(dir, stub='bundle_',sfreqs=['90','150','220'],estub='GHz.pkl', nbundle=10)
        #mcshtfilelist = create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=100)
        print(rlist)
        lmax = 3100
        nside= 8192
        with open(dir+'../mask_50mJy_'+subfield+'.pkl','rb') as fp:
            mask = pkl.load(fp)
        #mask = None
        processedshtfile = calworkdir + '/data/shts_processed.bin'
        spec.take_and_reformat_shts(rlist, processedshtfile,
                                    nside,lmax,
                                    cmbweighting = True, 
                                    mask  = mask,
                                    kmask = None,
                                    ell_reordering=None,
                                    no_reorder=False,
                                    ram_limit = None,
                                    npmapformat=False,
                                    pklmapformat=True,
                                    map_key='T'
        ) 

    

if __name__ == "__main__" and CAL == True:

    subfields = ['ra0hdec-44.75','ra0hdec-52.25','ra0hdec-59.75','ra0hdec-67.25']
    subfields = ['ra0hdec-59.75','ra0hdec-67.25']
    subfields = ['ra0hdec-44.75','ra0hdec-52.25']
    subfields = ['ra0hdec-52.25']
    subfields = ['ra0hdec-52.25','ra0hdec-59.75','ra0hdec-67.25']
    subfields = ['ra0hdec-44.75']
    dir='/sptlocal/user/creichardt/hiell2022/bundle10/'
    mapfiles = create_real_file_list_v4(dir, stub='bundle_',sfreqs=['90','150','220'],estub='GHz.pkl', nbundle=10)
    #banddef = np.arange(0,3100,50)   
    banddef = np.asarray([0,188,288,388,  #dump bins
               424,  460,  496,  532,  568,  604,  640,  676,  712,  748,  # 4x planck binning = 36
               784,  820,  856,  892,  928,  964, 1000, 1036, 1072, 1108, 1144,
               1180, 1216, 1252, 1288, 1324, 1360, 1396, 1432, 1468, 1504,
               1538, 1572, 1606, 1640, 1674, 1708, 1742, 1776, 1810, 1844, #moved to 2x planck = 34
               1878, 1912, 1946, 1980, 2014, 
               2047, 2080, 2113, 2146, 2179, 2212, 2245, 2278, 2311, 2344, # moved to 1x planck = 33
               2377, 2410, 2443, 2476, 2509])

 
    for subfield in subfields:
        workdir='/big_scratch/cr/xspec_2022/cal/'+subfield+'/data/'
        shtfile = workdir+'shts_processed.bin'
        setdef = np.zeros([10,3],dtype=np.int32)
        setdef[:,0]=np.arange(0,10,dtype=np.int32)
        setdef[:,1]=np.arange(0,10,dtype=np.int32)+10
        setdef[:,2]=np.arange(0,10,dtype=np.int32)+20
        maskfile=dir+'../mask_50mJy_'+subfield+'.pkl'
        with open(maskfile,'rb') as fp:
            mask = pkl.load(fp)
        nside=8192
        cal_spectrum      = spec.unbiased_multispec(mapfiles,mask,banddef,nside,
                                              lmax=3100,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=False, auto=False,
                                              kmask=None,
                                              cmbweighting=True)
        file_out = workdir + 'cal_spectrum.pkl'
        with open(file_out,'wb') as fp:
            pkl.dump(cal_spectrum,fp)


if __name__ == "__main__" and TEST == True:
        #subfield='ra0hdec-67.25'
    workdir = '/big_scratch/cr/xspec_2022/'
    #os.makedirs(calworkdir+'data/',exist_ok=True)
    print(workdir)
    if False:
        dir='/sptlocal/user/pc/g3files/220GHz/'
        rlist = [dir+'combined_T_219ghz_00088.g3',dir+'combined_T_219ghz_00066.g3']

        #mcshtfilelist = create_sim_file_list(dir,dstub='inputsky{:03d}/',bstub='bundles/alm_bundle',sfreqs=['90','150','220'],estub='GHz.g3.gz.npz',nsim=100)
        print(rlist)
        lmax = 7100
        nside= 8192
        mfile='/sptlocal/user/creichardt/hiell2022/mask_ra0hdec-52.25.pkl'
        with open(mfile,'rb') as fp:
            mask = pkl.load(fp)
        #mask = None
        processedshtfile = workdir + '/test/shts_processed.bin'
        spec.take_and_reformat_shts(rlist, processedshtfile,
                                    nside,lmax,
                                    cmbweighting = True, 
                                    mask  = mask,
                                    kmask = None,
                                    ell_reordering=None,
                                    no_reorder=False,
                                    ram_limit = None,
                                    npmapformat=False,
                                    pklmapformat=False,
                                    map_key='T'
        )
    if True:
        dir='/sptlocal/user/pc/g3files/220GHz/'    

        rlist = np.zeros([2,1],dtype=object)
        rlist[0,0]=dir+'combined_T_219ghz_00088.g3'
        rlist[1,0]=dir+'combined_T_219ghz_00066.g3'
        workdir=workdir+'test/'
        shtfile = workdir+'shts_processed.bin'
        setdef=np.zeros([2,1],dtype=np.int)
        
        setdef[:,0]=np.arange(0,2,dtype=np.int32)

        maskfile='/sptlocal/user/creichardt/hiell2022/mask_ra0hdec-52.25.pkl'
        with open(maskfile,'rb') as fp:
            mask = pkl.load(fp)
        nside=8192
        banddef = np.arange(0,7100,50)
        mc_spectrum      = spec.unbiased_multispec(rlist,mask,banddef,nside,
                                              lmax=7100,
                                              resume=True,
                                              basedir=workdir,
                                              persistdir=workdir,
                                              setdef=setdef,
                                              jackknife=False, auto=True,
                                              kmask=None,
                                              cmbweighting=True)
        file_out = workdir + 'mc_spectrum.pkl'
        with open(file_out,'wb') as fp:
            pkl.dump(mc_spectrum,fp)
