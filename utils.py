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
        for frame in frames:
            return np.asarray(frame["weight"],dtype=np.float32)
    except:
        return None


def find_geom_mean_weight(filelist):
    product = None
    count=0
    for file in filelist:
        loc_weight = pull_weight(file)
        if loc_weight is None:
            raise Exception("Failed to read/load wights from {}".format(file))
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
    medwt = np,median(weight[weight > 0])
    threshold = medwt * fraction
    mask = np.zeros(weight.shape,dtype=np.bool)
    mask[weight > threshold] = True

    ###maybe smooth this???
    ### look at plots and decide

    return mask
