import unbiased_multispec as multispec
import glob,os
import sys
#sys.path.append('/home/creichardt/spt3g_software/build')
from spt3g import core,maps, calibration
import numpy as np
import pickle as pkl

dir = '/sptgrid/analysis/eete+lensing_19-20/v2/data_maps/ra0hdec-44.75/'
file_list = glob.glob(dir+'healpix*150GHz.g3')

#cut to five:
file_list = file_list[:5]

shtfile='/scratch/cr/sht.bin'

multispec.take_and_reformat_shts(file_list, shtfile,
                           8192,15000,
                           cmbweighting = True, 
                           mask  = None,
                           kmask = None,
                           ram_limit = None
                          )
