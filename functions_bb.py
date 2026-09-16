import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
import glob
import sys
sys.path.append('/home/creichardt/spt3g_software/build')
import healpy as hp
print('imported healpy')
import unbiased_multispec as spec
import namaster_multispec as naspec
import utils
import end_to_end
from spt3g import core,maps, calibration
import argparse
import pickle as pkl
import pdb
import time

from astropy.io import fits

NULLSHT=False

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-nullsht', action='store_true',dest='nullsht')

args = my_parser.parse_args()

NULLSHT=args.nullsht

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
        map1filelist[i] = base_path+stub1.format(i,freq)
        map2filelist[i] = base_path+stub2.format(i,freq)
        shtfilelist[i]  = out_base_path+outstub.format(null,i,freq)
    return map1filelist, map2filelist, shtfilelist


#####################################################################################################
# Top level calls, chosen with argparser
#####################################################################################################

if __name__ == "__main__" and NULLSHT is True:
    lmax=4500
    nside=8192
    base_path='/sptgrid/analysis/spt3g_d1_midell_tqu_healpix/real_data_maps/pre_null/'
    out_base_path='/scratch/cr/bb_nulls/'

    freqs=['095','150','220']
    nulls = ['azimuth','moon','sun','year']
    lrnull = 'scan'
    nulls = ['sun'] # for testing

    for freq in freqs:
        for null in nulls:
            print("On {} GHz and {}:".format(freq,null))
            map1filelist, map2filelist, shtfilelist = generate_null_file_list(base_path,out_base_path,freq,null)

            naspec.take_null_shts(map1filelist, map2filelist, shtfilelist,
                                nside,lmax,
                                purify_b = True,
                                mask  = None
                                )