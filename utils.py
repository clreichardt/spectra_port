#from re import A
import os
os.environ['OMP_NUM_THREADS'] = "6"
import numpy as np
# from spt3g import core
import pdb
#import healpy as hp

def quickplot(tmap,max=0.5):
    hp.visufunc.cartview(tmap,min=-1*max,max=max,latra=[-73,-38],lonra=[-55,55])

def cast_clonly_to_theory_format(filein,fileout):
    cls = np.loadtxt(filein)
    lmax = len(cls)-1
    l = np.arange(0,lmax+1)
    lfac = l * (l+1)/(2*np.pi)
    dl = cls * lfac
    dl[0:2]=0
    tmp = np.zeros([lmax+1,2])
    tmp[:,0]=l
    tmp[:,1]=dl
    np.savetxt(fileout,tmp)
    
    
def flatten_kmask(kmask, lmax):
    flat_kmask = np.zeros(hp.sphtfunc.Alm.getsize(lmax),dtype=np.float32)
    
    k=0
    for i in range(lmax+1):
        nn = lmax+1 - i
        flat_kmask[k:k+nn] = kmask[i:,i]    
        k+=nn
    return flat_kmask


def great_circle_distance(vec_border_lon, vec_border_lat, this_lon, this_lat):
    '''
    Finds minimum distance of a point (this_lon, this_lat) to a vector of long/lats
    Assumes all in radians
    returns distance in radians

    needs testing
    '''
    #assert vec_border_lon.shape == vec_border_lat.shape
    n = len(vec_border_lon)
    dist = 10. # you can never have 10 radian separation on a sphere...
    for i in range(n):
        local_lon = vec_border_lon[i]
        local_lat = vec_border_lat[i]
        this_dist = hp.angdist([local_lon,local_lat],[this_lon,this_lat])
        dist = min([this_dist,dist])
    return dist

def approx_distance(vec_border_lon, vec_border_lat, this_lon, this_lat):
    '''
    Finds approx minimum distance of a point (this_lon, this_lat) to a vector of long/lats
    Assumes all in radians
    will fail with edgewrapping  (eg 359 to 1). should fix this before wider use
    returns distance in radians

    needs testing
    '''
    #assert vec_border_lon.shape == vec_border_lat.shape
    n = len(vec_border_lon)
    dy = vec_border_lon - this_lon
    dx = (vec_border_lat - this_lat) * np.cos(this_lat)
    dist = np.sqrt(np.min(dx**2 + dy**2))
    return dist

def rectangle_points(vec_border_lon, vec_border_lat, this_lon, this_lat,delta_lon,delta_lat):
    '''
    find points within a range of point
    ie lat \in [this_lat-delta_lat,this_lat+delta_lat]
    ie lon \in [this_lon-delta_lon,this_lon+delta_lon]
    can return None if none in range
    '''
    pass

def pull_weight(file):
    ''' read g3 file
    return first weight
    as np array of float32
    
    expecting it to be full sky
    '''
    #grab weight from file
    #zero nan's possibly. not sure if needed...
    #weight[np.isnan(weight)]=0
    try:
        frames = core.G3File(file)
        frame = frames.next() #grab first frame
        return np.asarray(frame["weight"],dtype=np.float32)
    except:
        return None


def find_geom_mean_weight(filelist,norm=True):
    product = None
    count=0
    for file in filelist:
        if count % 10 == 0:
            print(count,file)
        loc_weight = pull_weight(file)
        if loc_weight is None:
            raise Exception("Failed to read/load wights from {}".format(file))
        if norm:
            loc_weight /= np.max(loc_weight)
        if product is None:
            product = loc_weight
        else:
            product *= loc_weight
        count += 1
    return product**(1/count)

def select_good_weight_region(weight,min_fraction):
    '''
    expect this to return a full sky map, true where good, false most places
    good == weight > min_fraction * median(non-zero weight)
    assumes untouched pixels are zero, not NAN!!!
    '''
    medwt = np.median(weight[weight > 0])
    threshold = medwt * min_fraction
    mask = np.zeros(weight.shape,dtype=np.bool)
    mask[weight > threshold] = True

    ###maybe smooth this???
    ### look at plots and decide

    return mask


def source_mask_profile(radius_rad, fwhm_rad, distances_rad):
    output = distances_rad * 0.0
    sigma = fwhm_rad / 2.35482 #sqrt(8*log(2))
    inds = distances_rad > radius_rad
    output[inds] = 1.0-np.exp(-0.5 * ((distances_rad[inds]-radius_rad)/sigma)**2)
    return output

def punch_holes(ras,decs,radii,nside,mask=None,pixel_list=None, ring=True, celestial=True,buffer=1.2,fwhm_arcmin=5.0):
    '''
    Inputs:
    ras, decs, radii: N-vectors of RA, Decl, and radius. All in degrees.
    nside: healpix nside
    Outputs: 
    12*nside**2 pixel full sky healpix map - 1 where good, 0 at sources, and taper in between
    Not implemented optional arguments below:
    Optional input: mask (ignored if passed pixel_list) -- will take non-zero pixels to create pixel_list
    Optional input: pixel_list - list of healpix pixel indices.
    optional: Celestial: in celestial
    optional: ring: in ring
    '''
    
    if len(radii) == 1:     
        radii = np.zeroes(ras.shape)+radii
    assert( ras.shape == decs.shape )
    assert( ras.shape == radii.shape)
    radii=np.deg2rad(np.asarray(radii))
    fwhm = np.deg2rad(fwhm_arcmin/60.) # 5'
    search_radii = buffer * radii + 2*fwhm
    thetas = 0.5*np.pi - np.deg2rad(decs) 
    phis = np.deg2rad(ras)
    phis[phis < 0] += 2*np.pi #wrap RAs
    vectors = hp.rotator.dir2vec(thetas,phi=phis)
    npts = len(ras)
    mask = np.ones(12*nside**2,dtype=np.float32)
    for i in range(npts):
        pixlist = hp.query_disc(nside,vectors[:,i],search_radii[i],inclusive=False,nest=(ring is False))
        loc_ang = hp.pix2ang(nside,pixlist,nest=(ring is False))
        #check this
        this_ang = [thetas[i],phis[i]]
        dist_list = hp.rotator.angdist(loc_ang,this_ang)
        value_list = source_mask_profile(radii[i],fwhm,dist_list)
        mask[pixlist] *= value_list
    return mask # this source-only mask can be multiplied by the edge taper mask

def fill_in_beams_fromfile(files,ells):
    nl = len(ells)
    nfreq = len(files)
    beams_interp = np.zeros([nspectra, nl],dtype=np.float32)
    for i in range(nfreq):
        bl = np.fromfile(files[i]) # this will need to be fixed
        beams_interp[i,:] = np.interp(ells,bl[0,:],bl[1,:])
        bad = beams_interp[i,:] < 0
    return beams_interp

def fill_in_theory(files,ells,cl2dl=False):
    nl = len(ells)
    
    nfreq = len(files)
    theory_interp = np.zeros([nfreq, nl],dtype=np.float32)
    for i in range(nfreq):
        dls = np.loadtxt(files[i])
        locl=dls[:,0]
        dl = dls[:,1]
        if cl2dl:
            dl = dl * locl*(locl+1)/(2*np.pi)
        dl[locl < 2]=0 #don't want 0,1
        theory_interp[i,:] = np.interp(ells,locl,dl)
    return theory_interp   
        
def fill_in_beams(beam_arr,ells):
    nl = len(ells)
    nfreq = beam_arr.shape[1]-1
    beams_interp = np.zeros([nfreq, nl],dtype=np.float32)
    ell_stored = beam_arr[:,0]
    
    for i in range(nfreq):
        bl = beam_arr[:,i+1]
        beams_interp[i,:] = np.interp(ells,ell_stored,bl) #this will need to be fixed
        bad = beams_interp[i,:] < 0
        beams_interp[i,bad]=0
    return beams_interp

def explode_beams(beams_interp):
    nfreq = beams_interp.shape[0]
    nl = beams_interp.shape[1]
    nsets = int(nfreq*(nfreq+1)/2)
    beams = np.zeros([nsets, nl],dtype=np.float32)
    k=0
    for i in range(nfreq):
        for j in range(i,nfreq):
            good = np.logical_and(beams_interp[i,:] > 0 ,beams_interp[j,:] > 0 )
            beams[k,good] = np.sqrt(beams_interp[i,good]*beams_interp[j,good])
            k += 1
    return beams


'''
no error checking. assumes right pixels in right order
'''
def load_spt3g_cutsky_healpix_ring_map(file,npix):
    map = np.fromfile(file,dtype=np.float32)
    assert (len(map) == npix)
    return map

#this code throws away high ells
# it also changes to be in DL instead of Cls
# it does *not* count Nmodes
#'/sptlocal/user/pc/mll/mll_tpltz.npy'
def convert_namaster_mll(namaster_file,mll_dl_file,lmax_out):
    mll = np.load(namaster_file)
    # initially setup for Cl's 
    # so we need to throw from Dl -> Cl, and then back
    
    nl = mll.shape[0]
    #square
    assert mll.shape == (nl,nl)
    lmax_in = nl

    # can't cut to higher number
    assert lmax_in >= lmax_out

    #Step 1: cut to 2:lmax_out
    mll = mll[1:lmax_out,1:lmax_out]
    
    #Step 2: 
    # used like np.matmul(mll,Dl_theory)
    l    = np.arange(2,lmax_out+1)
    lfac = l*( l+1) 
    lfac_x = (np.tile(lfac,(lmax_out-1,1))).T
    lfac_y = np.tile(1./lfac,(lmax_out-1,1))
       
    mll_dl = np.multiply(np.multiply(lfac_x,mll),lfac_y)
    
    np.save(mll_dl_file,mll_dl)


#this code throws away high ells
# it also changes to be in DL instead of Cls
# it also rebins the Mll matrix
# it does *not* count Nmodes

#'/sptlocal/user/pc/mll/mll_tpltz.npy'
def rebin_and_convert_namaster_mll(namaster_file,mll_dl_file,delta_l_out,lmax_out):
    mll = np.load(namaster_file)
    # initially setup for Cl's 
    # so we need to throw from Dl -> Cl, and then back
    
    nl = mll.shape[0]
    #square
    assert mll.shape == (nl,nl)
    lmax_in = nl

    # can't cut to higher number
    assert lmax_in >= lmax_out
    assert int(lmax_out/delta_l_out)*delta_l_out == lmax_out # integer number of bins

    #Step 1: cut to 2:lmax_out
    mll = mll[0:lmax_out,0:lmax_out]
    mll[0,:]=0
    mll[:,0]=0

    #Step 2: 
    # used like np.matmul(mll,Dl_theory)un
    l    = np.arange(1,lmax_out+1)
    lfac = l*( l+1) 
    lfac_x = (np.tile(lfac,(lmax_out,1))).T
    lfac_y = np.tile(1./lfac,(lmax_out,1))

    mll_dl = np.multiply(np.multiply(lfac_x,mll),lfac_y)
    mll_dl_out = 0
    # first assume a boxcar average of theory Dl's -- 
    for i in range(delta_l_out):
        for j in range(delta_l_out):
            mll_dl_out += mll_dl[i::delta_l_out,j::delta_l_out]
    mll_dl_out *= 1./(delta_l_out)   #we don't have the square because the sum of a row of Mll is unity. And we want that normalization to remain. 
    
    info = [1,lmax_out,delta_l_out]
    np.savez_compressed(mll_dl_file,mll_dl_out=mll_dl_out,info=info)
    
def load_mll(file):
    with np.load(file) as data:
        mll_dl = data['mll_dl_out']
        info = data['info']
    
    return info, mll_dl


def band_centers(banddef):
    ctr = (1+banddef[:-1]+banddef[1:])/2. #ie 0-5 -> 3
    return ctr

def bands_from_range(info):
    lmin,lmax,dl = info
    assert lmin ==1 # spectra code will be assuming banddef[0]=0
    banddef = np.arange(lmin-1,lmax+1,dl)
    #note: ith bin is banddef[i+1] >= ell > banddef[i]
    return banddef


def transfer_initial_estimate(cl_mc, cl_theory, bl, fsky_w2):
    """
    Calculates the initial estimates of the transfer function for a MASTER style pipeline
    Implements the definition of F0 above eq 18 in MASTER(astro-ph/0105302)
    This definition is :
    F0 = <C_l>_MC / (fsky * w2 * B_l^2 * C_th)
    
    Inputs:
    bl = beam function used in sims (N-vector). Should include pixel window function as appropriate.
    cl_theory = power spectrum used to create sims, can be Dl or Cl - must be same as cl_mc (N-vector) 
    cl_mc = power spectrum of processed sims, can be Dl or Cl - must be same as cl_theory (N-vector) 
    fsky_w2 = float (fsky_w2 factor from master for normalization of mask)
    Output: 
    Returns: initial estimate of transfer function (N-vector)
    """
    # The calculation is done in following steps:
    # 1. Calculate the denominator
    #   1.1. Multiply the theory Cls with the beam b_l**2 and set the negative values to zero
    #   1.2. Calculate the initial estimate F0
    #-----------------
    # 1.
    # 1.1
    cl_theory_beam2 = cl_theory * bl**2. # cl_theory_beam is just a moniker for Cl_theory with beam smoothing
    # 1.2
    F0 = cl_mc / ( fsky_w2 * cl_theory_beam2 )
    F0[np.where(cl_theory_beam2 < 0.)] = 1.
    F0[F0<0] = 0.0
    return F0

def transfer_iteration( F0, cl_mc, cl_theory, bl, fsky_w2, M_ll):
    '''
    Calculates the iterative3 estimates of the transfer function for a MASTER style pipeline (astro-ph/0105302)
    
    NB: all of the N-vectors below must share the same underlying ell-binning. No checks are made that they are consistent. 
    Inputs:
    F0 = previous transfer function iteration (N-vector)
    M_ll = mode-coupling matrix (NxN array)
    bl = beam function used in sims (N-vector)
    cl_theory = power spectrum used to create sims, can be Dl or Cl - must be same as cl_mc (N-vector) 
    cl_mc = power spectrum of processed sims, can be Dl or Cl - must be same as cl_theory (N-vector)
    fsky_w2 = float (fsky_w2 factor from master for normalization of mask)
    Outputs: 
    Returns: next iteration of transfer function (N-vector)
    '''
    cl_theory_beam2 = cl_theory * bl**2
    F = F0 + (cl_mc -  np.matmul(M_ll, (F0 * cl_theory_beam2 )) ) / (cl_theory_beam2 * fsky_w2)
    F[np.where(cl_theory_beam2 < 0.)] = 1.
    F[F<0] = 0.0
    return F



'''
; PURPOSE:
;
;  Take the mode-coupling matrix M_{uu'} and rebin it into bands
;  K_{bb\}.  At this step the Transfer function, the beam function and
;  the pixelization function are included. 
;
; CATEGORY:
;
;
;
; CALLING SEQUENCE:
;
;
;
; INPUTS:
;
;   matrix:   The mode coupling matrix M_{uu} (See
;   https://spt.uchicago.edu/trac/attachment/wiki/AnalysisNotesFeb2007/Pseudo-Cl-081021.pdf 
;   for a definition.
;   ell:  the mapping between u and \ell (Should be as long as M_{uu}
;         is on a side).   
;        This is usually something like: delta_ell*dindgen((size[muu])[1])
;   bindef:  The definition of the bins.   This should be an array of
;        floats, one element per bin.   Each element should be the
;        upper bound of each ell bin.   The lowest bin by default
;        extends to \ell=0.   (Yeah that's right \ell=0. We are on the
;        flat sky here, so \ell is a continuous variable...  I wouldn't talk about
;        fractional ells in publication, but for the purposes of flat
;        sky code, in makes sense to think about \ells this way). 
;
; OPTIONAL INPUTS:
;
;  transferfunc:  The transfer function as evaluated at each element
;      of the matrix ell.   (default: 1)
;  beamfunc:  The beam function as evaluated at each element of ell
;      (default: 1)
;  pixfunc:  The pixel transfer function as evaluated at each element
;      of ell.   I personnally think that the pixel transfer function
;      is something bogus invented by people who don't quite understand
;      aliasing, but it's in Hivon et al, so I've included it here.
;      (default: 1) 
;
; KEYWORD PARAMETERS:
;
;  nocmbweighting: Do not assume that the input spectrum is l(l+1) weighted
; 
; OUTPUTS:
;
;  Returns the matrix, K_{bb} as defined in the master paper
;
; OPTIONAL OUTPUTS:
;
;  None
;
; SIDE EFFECTS:
;
;  None known
;
; PROCEDURE:
;
;   Uses equation (25) of Hivon et al, arXiv:astro-ph/0105302v1
;
; MODIFICATION HISTORY:
;
;   Sept 23, 2008: Created, Martin Lueker 
;   Oct 28, 2008: Documentation header added, Martin Lueker
;   06/25/2015  : add option allowing negative transfer function, Zhen Hou
'''


def rebin_coupling_matrix( matrix, ell, bindef, transferfunc=None, 
                                beamfunc=None, pixelfunc=None, 
                                nocmbweighting=True, #Already in Dl's and doing weights in Dls
                                onesidecmbweighting=False,
                                master=True, 
                                allow_negative_transfer=False):
    nell=ell.shape[0]

    if beamfunc is None:
        beamfunc=np.ones(nell,dtype=np.float32)
    if pixelfunc is None:
        pixelfunc=np.ones(nell,dtype=np.float32)
    if transferfunc is None:
        transferfunc=np.ones(nell,dtype=np.float32)


    nbins=bindef.shape[0]-1

    '''
;; We may not want calculate the mode-to-mode coupling matrix
;; at delta-ell of one (for reasons of computational efficiency, 
;; or space, or whatever), hence the spacing in ell, may be 
;; different than unity ( as assumed in the master paper), thus
;; the new definition should be:
;;
;; P_{bl}=1/2/!PI * ell(ell+1) / sum_{ell in b} 1
    '''

    p = np.zeros([nbins,nell],dtype=np.float64)
    q = np.zeros([nell,nbins],dtype=np.float64)

    for i in range(nbins):
    
        idx=(ell <= bindef[i+1])*(ell > bindef[i])
        cnt = np.sum(idx)
    
        if nocmbweighting:
            p[i,idx] = 1/cnt
            q[idx,i] = 1.
        elif onesidecmbweighting:
            p[i,idx] = 1/cnt
            q[idx,i] = 1./(2*np.pi) * (ell[idx]*(ell[idx]+1))
        else:
            p[i,idx] = ((2*np.pi/cnt) / (ell[idx]*(ell[idx]+1)))
            q[idx,i] = 1./(2*np.pi) * (ell[idx]*(ell[idx]+1))


    scale = transferfunc*(pixelfunc*beamfunc)**2
    scale[scale<0]=0
    scale = np.reshape(scale,nell)   #likely not needed   
    if master:
        print('using master scaling')

        scaling = np.tile(scale,[nell,1]) #may be transpose of what want...
        #scaling = scaling.T #decided it was transpose. changed mind
    else:
        print('using non master scaling')
        scale = np.reshape(np.sqrt(scale),[1,nell])
        scaling = np.matmul(scale.T,scale)

    #may have some transposes of what is desired... will need to check some
    #print(p.shape,matrix.shape,scaling.shape,q.shape)
    result = np.matmul(p,np.matmul(np.multiply(matrix,scaling),q))
    #pdb.set_trace()
    return result


def window_function_calc(banddef, transfer_dic, ellmin = 10, ellmax=13000,no_mode_wt=False):
    '''
    ;this version assumes the real spectra are binned to final bins and
    ;tries to juryrig a correction for overlapping bins compared to the
    ;kernel binning.
    
    ;Plan -
    ; for all ell, make a vector of all zeroes, except 1 at ell
    ; Apply Tf, Mll to get a "simulated" Dl
    ; apply inversion and binner to get the output bins
    ; (use cuts as in code)
    '''

    nl = ellmax - ellmin + 1
    ells = np.arange(ellmin, ellmax+1)

    ellbin = transfer_dic['ell']
    nellbin = ellbin.shape[0]
    kernel = transfer_dic['Mll']
    inv_binned_kernel = transfer_dic['invkernel']
    bl     = transfer_dic['bl']
    transfer=transfer_dic['transfer']
    iskip = transfer_dic['iskip']
    eskip = transfer_dic['eskip']
    btrans  = transfer * bl**2

    #upper / lower edges
    #warning -- assumes equal spacing for all bins
    dell = 0.5 * (ellbin[1]-ellbin[0])
    ellbin0 = ellbin-dell
    ellbin1 = ellbin+dell

    usedbands = np.arange(iskip,eskip)
    nkept = len(usedbands)
    kept = np.arange(nkept)
    win = np.zeros([nkept,nl])
    nb = len(banddef)


    basebininds = np.zeros([3,nb],dtype=np.int64)
    wtbins = np.zeros([nellbin,nb])

    modefac = ellbin
    if no_mode_wt:
        modefac = modefac * 0 + 1.

    for i in range(nb-1):
        jnds, = np.nonzero( (ellbin1 > banddef[i]) * (ellbin0 <= banddef[i+1]))
        ni = jnds.shape[0]
        if ni > 0: #don't want 0 length
            basebininds[0,i]=ni
            basebininds[1,i]=jnds[0]
            basebininds[2,i]=jnds[-1]
            wtbins[jnds[0]:jnds[-1],i]=1.
            wtbins[jnds[0],i] = (ellbin1[jnds[0]]-banddef[i])/(2*dell)
            wtbins[jnds[ni-1],i] = (banddef[i+1]-ellbin0[jnds[-1]])/(2*dell)
            wtbins[:,i] *=modefac
            wtbins[:,i] /= np.sum(wtbins[:,i])
            
    basetmp = np.zeros(nkept)
    effbin  = ellbin1
    tmp2 = effbin * 0
    tmp = tmp2 * 0

    for i in range(nl):
        if i % 200 == 0:
            print(i)
        tmp[:]  = 0
        tmp2[:] = 0
        basetmp[:] = 0
        ind = (np.nonzero(ells[i] < effbin))[0]
        #should be in the first of these bins that satisfy ell < lmax_bin
        # we did Mll in Dl here 
        tmp2[ind[0]]= btrans[ind[0]] / ( 2 * dell)
        tmp2 = np.matmul(kernel, tmp2)
        #tmp2 *= ellbin * (ellbin+1)

        for j in usedbands:
            basetmp[j-iskip] = np.sum(tmp2 * wtbins[:,j])
        #maybe do with tile

        specfixed = np.matmul(inv_binned_kernel , basetmp)

        win[:,i] = specfixed

    return win


#changing binning of bandpowers, and optionally the cov/window functions
#once this is debugged should add weights
def rebin_spectrum(bands_in, bands_out, spec0, cov0=None, win0=None): 

    nbands_in=bands_in.shape[0] - 1
    nbands_out=bands_out.shape[0] - 1

    if len(spec0.shape) == 1:
        nsets = len(spec0) /nbands_in
    if len(spec0.shape) == 2:
        nsets = spec0.shape[0]

    transform=np.zeros([nbands_out,nbands_in])

    #assumes first element of bands is zero
    for i in range(nbands_out):
        transform[i,np.logical_and(bands_in[1:] > bands_out[i], bands_in[1:] <= bands_out[i+1])    ] = 1
        #normalize
        transform[i,:] /= np.sum(transform[i,:])
    
    spec_in=np.reshape(spec0, [nsets,nbands_in])
    spec_out=np.zeros([nsets,nbands_out])
    for i in range(nsets):
        #print(transform.shape)
        #print(spec_in[i,:].shape)
        spec_out[i,:]=np.matmul(transform,spec_in[i,:])
    
    if cov0 is not None:
        cov_in=np.reshape(cov0, [nsets,nbands_in, nsets, nbands_in])
        cov_out=np.zeros([nsets,nbands_out, nsets, nbands_out])
        for i in range(nsets):
            for j in range(nsets):
                cov_out[i,:,j,:,]=np.matmul(np.matmul(transform,cov_in[i,:,j,:]),transform.T)
        if len(cov_in.shape) ==2:
            #reshape to 2d to match input
            cov_out = np.reshape(cov_out,[nsets*nbands_out,nsets*nbands_out])
    else:
        cov_out = None

    if win0 is not None:
        nells=win0.shape[1]
        win_in=np.reshape(win0, [nsets,nbands_in,nells])
        win_out=np.zeros([nsets,nbands_out,nells],dtype=np.float32)
        for i in range(nsets):
            win_out[i,:,:]=np.matmul(transform,win_in[i,:,:])
        if len(win0.shape) == 2:
            #reform to 2d to match input
            win_out=np.reshape(win_out,[nsets*nbands_out,nells])
    else: 
        win_out = None
    
    return spec_out,cov_out,win_out,transform

#changing binning of bandpowers, and optionally the cov/window functions
#once this is debugged should add weights
def weighted_rebin_spectrum(bands_in, bands_out, spec0, cov0=None, win0=None, weights = None): 

    nbands_in=bands_in.shape[0] - 1
    nbands_out=bands_out.shape[0] - 1
    if len(spec0.shape) == 1:
        nsets = len(spec0) // nbands_in
    if len(spec0.shape) == 2:
        nsets = spec0.shape[0]

    if weights is not None:
        assert spec0.shape == weights.shape
        weights_in = np.reshape(weights, [nsets,nbands_in])
    else:
        print('Warning -- better to call the unweighted code for the no-weights case')
        #weights was none, need to fill it.
        weights_in = np.ones([nsets,nbands_in])

    
    transform=np.zeros([nsets,nbands_out,nbands_in])

    #assumes first element of bands is zero
    
    for i in range(nbands_out):
        section = np.logical_and(bands_in[1:] > bands_out[i], bands_in[1:] <= bands_out[i+1]) 
        for j in range(nsets):
            transform[j,i,section] = weights_in[j,section]
            #normalize
            transform[j,i,:] /= np.sum(transform[j,i,:])
    
    spec_in=np.reshape(spec0, [nsets,nbands_in])
    spec_out=np.zeros([nsets,nbands_out])
    for i in range(nsets):
        #print(transform.shape)
        #print(spec_in[i,:].shape)
        spec_out[i,:]=np.matmul(transform[i,:,:],spec_in[i,:])
    
    if cov0 is not None:
        cov_in=np.reshape(cov0, [nsets,nbands_in, nsets, nbands_in])
        cov_out=np.zeros([nsets,nbands_out, nsets, nbands_out])
        for i in range(nsets):
            cov_out[i,:,i,:]=np.matmul(np.matmul(transform[i,:,:],cov_in[i,:,i,:]),transform[i,:,:].T)
            for j in range(i+1,nsets):
                cov_out[i,:,j,:]=np.matmul(np.matmul(transform[i,:,:],cov_in[i,:,j,:]),transform[j,:,:].T)
                cov_out[j,:,i,:] = cov_out[i,:,j,:].T
        if len(cov_in.shape) ==2:
            #reshape to 2d to match input
            cov_out = np.reshape(cov_out,[nsets*nbands_out,nsets*nbands_out])
    else:
        cov_out = None

    if win0 is not None:
        nells=win0.shape[1]
        win_in=np.reshape(win0, [nsets,nbands_in,nells])
        win_out=np.zeros([nsets,nbands_out,nells],dtype=np.float32)
        for i in range(nsets):
            win_out[i,:,:]=np.matmul(transform[i,:,:],win_in[i,:,:])
        if len(win0.shape) == 2:
            #reform to 2d to match input
            win_out=np.reshape(win_out,[nsets*nbands_out,nells])
    else: 
        win_out = None
    
    return spec_out,cov_out,win_out,transform
