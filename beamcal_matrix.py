

















    #assuming wrote some stuff before this including beam_arr*pix_arr
    nl = lcenter.shape[0]
    beam1d = np.zeros(3*nl)
    for i in range(3):
        beam1d[i*nl:(i+1)*nl] = beam_arr[lcenter,i+1] #possibly interp here

    beam2d = np.tile(beam1d,[1,3*nl]) #check this
    frac_bcov = bcov / beam2d / beam2d.T #elementwise division
    
    eval,evec = np.linalg.eig(frac_bcov)
    #cov = np.matmul(np.matmul(evec,np.diag(np.abs(eval))),evec.T)
    print(eval)
    pdb.set_trace()
    
    #think about how many to keep
    Nmax=10
    eval[Nmax:]=0
    
    norm_evec = np.matmul(np.diag(np.sqrt(eval)),evec.T) 
    pdb.set_trace() # again check if this is right transformation
    #possibly simplify this to a for loop
    norm_evec = np.zeros([Nmax,Nb],dtype=np.float64)
    for i in range(Nmax):
        norm_evec[i,:] = evec[i,:] * np.sqrt(eval[i])
    #again check this
    
    calcov =  [[0.0040^2, 0.003^2, 0],
            [.003^2, 0.0036^2, 0],
            [ 0, 0, 0]]


'''
     if n220 gt 0 then $
        if use_band_year[i220,iSptpol] then stop ;no 220s
     if use_band_year[i90,iSptpol] then begin
        allywts[diag,diag]=weight_years_use[*,i90,iSptpol]
        
        for j=0,npolbeam-1 do begin
           this_err1 = dblarr(nl)+1
           if revind[0] eq i90 then $
              this_err1 *= (1+savstruct.polbeamerrs[*,j,0])
           if revind[1] eq i90 then $
              this_err1 *= (1+savstruct.polbeamerrs[*,j,0])
           this_err1-=1.0
           this_err2 = dblarr(nl)+1
           if revind[2] eq i90 then $
              this_err2 *= (1+savstruct.polbeamerrs[*,j,0])
           if revind[3] eq i90 then $
              this_err2 *= (1+savstruct.polbeamerrs[*,j,0])
           this_err2-=1.0
           
           this_err1 *= scalefactor
           this_err2 *= scalefactor
           this_corr = this_err1 # this_err2
        
           corr += allywts # this_corr # allywts
        endfor
        ;got to here...
        ; need to make sure ordering is ok...
     endif
     if use_band_year[i150,iSptpol] then begin
        allywts[diag,diag]=weight_years_use[*,i150,iSptpol]

        for j=0,npolbeam-1 do begin
           this_err1 = dblarr(nl)+1
           if revind[0] eq i150 then $
              this_err1 *= (1+savstruct.polbeamerrs[*,j,1])
           if revind[1] eq i150 then $
              this_err1 *= (1+savstruct.polbeamerrs[*,j,1])
           this_err1-=1.0
           this_err2 = dblarr(nl)+1
           if revind[2] eq i150 then $
              this_err2 *= (1+savstruct.polbeamerrs[*,j,1])
           if revind[3] eq i150 then $
              this_err2 *= (1+savstruct.polbeamerrs[*,j,1])
           this_err2-=1.0
           
           this_err1 *= scalefactor
           this_err2 *= scalefactor
           this_corr = this_err1 # this_err2
        
           corr += allywts # this_corr # allywts
        endfor
        
        this_err1 *= scalefactor
        this_err2 *= scalefactor
        this_corr = this_err1 # this_err2
        
        corr += allywts # this_corr # allywts
     endif
     '''
