#from statistics import covariance
import sys
from turtle import setundobuffer
import numpy as np
#sys.path=["/home/creichardt/.local/lib/python3.7/site-packages/","/home/creichardt/spt3g_software/build","/home/creichardt/.local/lib/python3.7/site-packages/healpy-1.15.0-py3.7-linux-x86_64.egg"]+sys.path


import os

import pickle as pkl

AlmType = np.dtype(np.complex64)


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

        
        use_setdef  = setdef
        use_shtfile = processed_sht_file

        self.use_setdef = use_setdef
        
                               
        self.allspectra = None
        self.nmodes = None
        
        
        #bring it all together


        self.spectrum = None
        self.cov      = None
        self.est1_cov = None
        self.est2_cov = None
                                 
