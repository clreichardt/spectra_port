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
import scipy.stats
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
    null_base_path='/sptlocal/user/creichardt/bb_midl/'



NULLSHT=False
REFORMATNULL=False
NULL=False
PRINTSTATS=False

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-nullsht', action='store_true',dest='nullsht')
my_parser.add_argument('-reformatnull', action='store_true',dest='reformatnull')
my_parser.add_argument('-null', action='store_true',dest='null')
my_parser.add_argument('-printstats', action='store_true',dest='printstats')

args = my_parser.parse_args()

NULLSHT=args.nullsht
REFORMATNULL=args.reformatnull
NULL=args.null
PRINTSTATS=args.printstats

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


def null_spectrum_chisq(freq, null, null_base_path, lmin=0, lmax=np.inf):
    '''
    Loads the null spectrum written by compute_null_spectrum and returns
    (chisq, dof) using only the diagonal of cov1.
    Only bins lying entirely within [lmin, lmax] (per the saved banddef) are used.
    Bins with non-positive variance (eg. fully masked bins) are excluded.
    '''
    infile = null_base_path+'null_spectrum_{}_{}ghz.npz'.format(null,freq)
    data = np.load(infile)
    spectrum = data['spectrum'].flatten()
    var = np.diag(data['cov1'])
    banddef = data['banddef']
    inrange = (banddef[:-1] >= lmin) & (banddef[1:] <= lmax)
    good = inrange & (var > 0)
    chisq = np.sum(spectrum[good]**2/var[good])
    dof = int(np.sum(good))
    return chisq, dof


def chisq_ptes(chisq, dof, ncovdof):
    '''
    Returns (PTE assuming a chisq distribution, PTE assuming an F-distribution).
    The F-distribution accounts for the variance being estimated from data;
    ncovdof is the number of degrees of freedom in that estimate.
    '''
    pte_chisq = scipy.stats.chi2.sf(chisq, dof)
    pte_myf = scipy.stats.chi2.sf(chisq * ((ncovdof-2)/(ncovdof)), dof) #note ncovdof = nbundles-1
    pte_f = scipy.stats.f.sf(chisq/dof, dof, ncovdof)
    return pte_chisq, pte_myf, pte_f


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


if __name__ == "__main__" and PRINTSTATS is True:
    lmin = 500     # use bins with lower edge >= lmin
    lmax = 4500  # use bins with upper edge <= lmax
    nbundle = 25
    ncovdof = nbundle - 1 # dof of the bundle-based variance estimate, for F-distribution PTEs
    freqs = ['095','150','220']
    nulls = ['moon','azimuth','sun','year','scan']

    # chisq and dof for each (freq, null) test
    results = {}
    for freq in freqs:
        for null in nulls:
            results[freq,null] = null_spectrum_chisq(freq, null, null_base_path,
                                                     lmin=lmin, lmax=lmax)

    print('Using bins within ell = [{}, {}]; F-distribution denominator dof = {}'.format(lmin,lmax,ncovdof))
    hdrfmt = '{:>6s} {:>10s} {:>10s} {:>5s} {:>10s} {:>10s} {:>10s}'
    rowfmt = '{:>6s} {:>10s} {:10.2f} {:5d} {:10.4f} {:10.4f} {:10.4f}'
    def print_row(freq, null, chisq, dof):
        print(rowfmt.format(freq,null,chisq,dof,*chisq_ptes(chisq,dof,ncovdof)))
    def print_sum(freq, null, keys):
        print_row(freq, null, sum(results[k][0] for k in keys), sum(results[k][1] for k in keys))

    print('\nIndividual null tests:')
    print(hdrfmt.format('freq','null','chisq','dof','PTE(chi2)','PTE(myF)','PTE(F)'))
    for freq in freqs:
        for null in nulls:
            print_row(freq, null, *results[freq,null])

    print('\nAll null tests, single frequency:')
    print(hdrfmt.format('freq','null','chisq','dof','PTE(chi2)','PTE(myF)','PTE(F)'))
    for freq in freqs:
        print_sum(freq, 'all', [(freq,null) for null in nulls])

    print('\nAll frequencies, single null test:')
    print(hdrfmt.format('freq','null','chisq','dof','PTE(chi2)','PTE(myF)','PTE(F)'))
    for null in nulls:
        print_sum('all', null, [(freq,null) for freq in freqs])

    print('\nEnsemble (all frequencies, all null tests):')
    print(hdrfmt.format('freq','null','chisq','dof','PTE(chi2)','PTE(myF)','PTE(F)'))
    print_sum('all', 'all', list(results.keys()))

    print('\nDistribution of individual PTEs:')
    ptes = np.asarray([chisq_ptes(chisq,dof,ncovdof) for chisq,dof in results.values()])
    for j, label in enumerate(['chi2','myF','F']):
        print('PTE({}): min = {:.4f}, N(<0.05) = {:d}/{:d}, KS vs uniform p = {:.4f}'.format(
            label, np.min(ptes[:,j]), int(np.sum(ptes[:,j] < 0.05)), ptes.shape[0],
            scipy.stats.kstest(ptes[:,j],'uniform').pvalue))
