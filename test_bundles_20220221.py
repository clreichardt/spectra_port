import unbiased_multispec as um
import numpy as np

if __name__ == "__main__":
    
    test_n_bundles()



def test_n_bundles():



    banddef = [0,500,1000,1500,2000,2200,2500,2800,3100,3500,4000,4600,5200,6000,6800,7800,9000,11
    mcdir = workdir + '/mc/'
    mc_specrum = spec.unbiased_multispec(mapfiles, window, banddef, nside,
                                         lmax=lmax,
                                         resume=resume,
                                         basedir=mcdir,
                                         setdef=setdef_mc,
                                         jackknife=False, auto=True,
                                         kmask=None,
                                         cmbweighting=True)
