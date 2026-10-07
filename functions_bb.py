import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import glob
import sys
sys.path.append('/home/creichardt/spt3g_software/build')
import healpy as hp
print('imported healpy')
#import unbiased_multispec as spec
import namaster_multispec as naspec
import utils
#import end_to_end
#from spt3g import core,maps, calibration
import argparse
#import pickle as pkl
import pdb
import time

from astropy.io import fits

SPARTAN=False

if SPARTAN:
    base_path='/data/gpfs/projects/punim1199/'
    out_base_path='/data/gpfs/projects/punim1199/bb_nulls/'
    mask_path='/data/gpfs/projects/punim1199/'
    null_base_path='/data/gpfs/projects/punim1199/bb_midl/'
else:
    base_path='/sptgrid/analysis/spt3g_d1_midell_tqu_healpix/real_data_maps/pre_null/'
    out_base_path='/sptlocal/user/creichardt/bb_nulls/'
    mask_path='/sptlocal/user/creichardt/bb2020/'
    null_base_path='/big_scratch/cr/bb_midl/'



NULLSHT=False
REFORMATNULL=False
NULL=False

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-nullsht', action='store_true',dest='nullsht')
my_parser.add_argument('-reformatnull', action='store_true',dest='reformatnull')
my_parser.add_argument('-null', action='store_true',dest='null')

args = my_parser.parse_args()

NULLSHT=args.nullsht
REFORMATNULL=args.reformatnull
NULL=args.null

#####################################################################################################
# Utility functions. Not expected to be called outside this file
#####################################################################################################
def generate_null_file_list(base_path,out_base_path,freq,null):
    # returns map1filelist, map2filelist, shtfilelist
    #if null=LR, map2filelist is none

    match null:
        case 'scan': #LR
            stub1='scan_right_bundle{:02d}_{}ghz.fits'
            stub2='scan_left_bundle{:02d}_{}ghz.fits'
        case 'sun': 
            stub1='sun_above_bundle{:02d}_{}ghz.fits'
            stub2='sun_below_bundle{:02d}_{}ghz.fits'
        case 'moon': 
            stub1='moon_above_bundle{:02d}_{}ghz.fits'
            stub2='moon_below_bundle{:02d}_{}ghz.fits'
        case 'year': 
            stub1='year_2019_bundle{:02d}_{}ghz.fits'
            stub2='year_2020_bundle{:02d}_{}ghz.fits'
        case 'azimuth': 
            stub1='azimuth_near_bundle{:02d}_{}ghz.fits'
            stub2='azimuth_far_bundle{:02d}_{}ghz.fits'
        case _:
            raise Exception('unknown null'+null)

    outstub='alm_null_{}_{:02d}_{}ghz.bin'
    nbundle=25
    map1filelist = ['']*nbundle
    map2filelist = ['']*nbundle
    shtfilelist  = ['']*nbundle
    for i in range(nbundle):
        map1filelist[i] = base_path+null+'/'+stub1.format(i,freq)
        map2filelist[i] = base_path+null+'/'+stub2.format(i,freq)
        shtfilelist[i]  = out_base_path+outstub.format(null,i,freq)
    return map1filelist, map2filelist, shtfilelist


def reformat_null_shts(freq, null, out_base_path,
                        lmax, mask,
                        cmbweighting=True,
                        kmask=None):
    '''
    Reformats the per-bundle purified-B alm files written by naspec.take_null_shts
    (cmbweighting, kmask, partial-sky mask normalization, ell-reordering) into a single
    binary file. Returns the path to that file.
    '''
    _, _, shtfilelist = generate_null_file_list(base_path, out_base_path, freq, null)

    processedshtfile = out_base_path+'processed_null_{}_{}ghz.bin'.format(null,freq)
    naspec.reformat_shts(shtfilelist, processedshtfile,
                          lmax,
                          cmbweighting=cmbweighting,
                          mask=mask,
                          kmask=kmask,
                          ell_reordering=None,
                          no_reorder=False)
    return processedshtfile


def compute_null_spectrum(freq, null, out_base_path, null_base_path,
                           lmax, banddef, nbundle=25):
    '''
    Computes the binned cross-spectrum and covariance across bundles from an
    already-reformatted null sht file (see reformat_null_shts).

    Cross- (not auto-) spectra are used: each bundle's alm is already the null
    (half-difference) map, so cross-correlating distinct bundles cancels noise bias.
    '''
    processedshtfile = out_base_path+'processed_null_{}_{}ghz.bin'.format(null,freq)

    setdef = np.arange(nbundle,dtype=np.int32).reshape(nbundle,1)
    allspectra, nmodes = naspec.take_all_cross_spectra(processedshtfile, lmax,
                                                        setdef, banddef, auto=False)
    spectrum,cov,cov1,cov2 = naspec.process_all_cross_spectra(allspectra, banddef.shape[0]-1,
                                                               1, nbundle, auto=False)

    result = {'spectrum':spectrum,'cov':cov,'cov1':cov1,'cov2':cov2,
              'allspectra':allspectra,'nmodes':nmodes,'banddef':banddef}
    outfile = null_base_path+'null_spectrum_{}_{}ghz.npz'.format(null,freq)
    np.savez(outfile,**result)
    return result


#####################################################################################################
# Top level calls, chosen with argparser
#####################################################################################################

if __name__ == "__main__" and NULLSHT is True:
    lmax=4500
    nside=8192


    freqs=['095','150','220']
    freqs=['095']
    nulls = ['azimuth','moon','sun','year','scan']

    nulls = ['sun'] # for testing

    mask_file=mask_path+'puremask8192_0p5medwt_500mJy_nodisk_15arcmin.npz'
    mask = np.load(mask_file)['mask']
    for freq in freqs:
        for null in nulls:
            print("On {} GHz and {}:".format(freq,null))
            map1filelist, map2filelist, shtfilelist = generate_null_file_list(base_path,out_base_path,freq,null)
            '''oldtime=time.time()
            #q = naspec.load_q(map1filelist[0],U=False)
            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0

            print('hp load time (min):',timeinminutes)
            oldtime=newtime
            ind,q = naspec.load_q_cut(map1filelist[0],U=False)
            print(np.min(ind),np.max(ind),np.max(ind)-np.min(ind),12*8192**2)
            newtime=time.time()
            timeinminutes = (newtime - oldtime)/60.0
            print('fits load time (min):',timeinminutes)
            oldtime=newtime
            '''
            naspec.take_null_shts(map1filelist, map2filelist, shtfilelist,
                                nside,lmax,
                                purify_b = True,
                                mask  = mask
                                )


if __name__ == "__main__" and REFORMATNULL is True:
    lmax=4500

    freqs=['095','150','220']
    #freqs=['095']
    nulls = ['azimuth','moon','year','scan','sun']

    #nulls = ['sun'] # for testing

    mask_file=mask_path+'puremask8192_0p5medwt_500mJy_nodisk_15arcmin.npz'
    mask = np.load(mask_file)['mask']

    for freq in freqs:
        for null in nulls:
            print("Reformatting null shts for {} GHz, {}:".format(freq,null))
            reformat_null_shts(freq, null, out_base_path,
                                                   lmax, mask,
                                                   cmbweighting=True)


if __name__ == "__main__" and NULL is True:
    print('doing null')
    lmax = 4500

    freqs = ['095','150','220']
    nulls = ['moon','azimuth','sun','year','scan']

    banddef = np.arange(0,lmax+500,500)

    for freq in freqs:
        for null in nulls:
            print("On {} GHz and {}:".format(freq,null))
            compute_null_spectrum(freq, null, out_base_path, null_base_path,
                                                   lmax, banddef)

