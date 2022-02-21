import unbiased_multispec as um
import numpy as np
from spt3g import core,maps, calibration

if __name__ == "__main__":
    combine_lr_bundles()
    #test_n_bundles()


def combine_lr_bundles():
    nf=200
    idir='/sptgrid/analysis/highell_TT_19-20/v3/mockobs/inputsky000/bundles/'
    odir='/sptgrid/analysis/highell_TT_19-20/v3/mockobs/inputsky000/bundles/sum'
    for i in range(nf):
        ifile = idir + 'bundle{:03d}_150GHz.g3.gz'.format(i)
        ofile = odir + 'bundle{:03d}_150GHz.g3.gz'.format(i)
        #hpmap = list(core.G3File('/sptgrid/analysis/eete+lensing_19-20/v2/data_maps/ra0hdec-44.75/healpix_105499116_150GHz.g3'))
        hpmap = list(core.G3File(ifile))    
        sum = 0.5*(hpmap[0]["T"]+hpmap[1]["T"]) #L+R/2
        
        frame = core.G3Frame(core.G3FrameType.Map)
        frame['map'] = sum
        with core.G3Writer(ofile) as writer:
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
