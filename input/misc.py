# -----------------------------------------
# -  misc routine for CoSyR   -
# -----------------------------------------

## generate test wavelets
def gen_test_wavelets(scaled_alpha, scaled_chi, n_alpha, n_chi, _gamma=10, unscale_coord=True, flatten = True, num_fields=1) :
    import numpy as np
    from scipy import stats

    print("wavelet parameters:", scaled_alpha, scaled_chi, n_alpha, n_chi, _gamma)
    peak_factor =  1.0 
    peak_shift = 0.0
    #print("peak_factor=", peak_factor, "peak_shift=", peak_shift)
    #if (np.mod(n_alpha,2) ==1) : peak_shift =  0.5*scaled_alpha/n_alpha

    # -- start modif --
    # alpha_axis = np.linspace(-scaled_alpha/2.0, scaled_alpha/2.0, n_alpha)
    # mean = scaled_alpha * 0.5
    # print("mean={}".format(mean))
    # dev = 190 # 190
    # dist = stats.norm(loc=mean, scale=dev)
    # bounds = dist.cdf([0, scaled_alpha])
    # pp = np.linspace(*bounds, n_alpha)
    # vals = dist.ppf(pp)
    # alpha_axis = (vals - mean)
    # print(alpha_axis)
    # # -- end modif --

    # -- start modif --
    # alpha_axis = np.linspace(-scaled_alpha/2.0, scaled_alpha/2.0, n_alpha)
    # chi_axis = np.linspace(-scaled_chi/2.0, scaled_chi/2.0, n_chi)
    # alpha = alpha_axis/_gamma**3.0
    # psi, eta = retard_angle_1D_ultra_relativistic(alpha)
    # -- end modif

    # -- start modif --
    alpha_max = (scaled_alpha*0.5)/1e6*_gamma**3.0
    number_of_wavelets_pos = 101  # for positive alpha axis
    number_of_wavelets_neg = 101   # for negative alpha axis

    psi_max=(24.0*alpha_max/_gamma**3.0)**(1.0/3.0)
    psi_max_negative_axis = alpha_max/_gamma**3.0/2.0

    print(alpha_max, psi_max, psi_max_negative_axis)

    psi_for_pos_axis = np.linspace(0.0, psi_max, number_of_wavelets_pos)
    psi_for_neg_axis = np.linspace(psi_max_negative_axis, 0.0, number_of_wavelets_neg, endpoint=False)

    alpha_axis = np.zeros(number_of_wavelets_pos + number_of_wavelets_neg)
    alpha_axis[:number_of_wavelets_neg] = -psi_for_neg_axis*2.0
    alpha_axis[number_of_wavelets_neg:] = psi_for_pos_axis**3.0/24.0

    alpha_axis *= _gamma**3.0
    alpha = alpha_axis/_gamma**3.0
    psi, eta = retard_angle_1D_ultra_relativistic(alpha)
    chi_axis = np.linspace(-scaled_chi/2.0, scaled_chi/2.0, n_chi)
    # -- end modif

    beta = np.sqrt(1.0-_gamma**(-2.0))

    wy, wx = np.meshgrid(chi_axis, alpha_axis)
    field = np.ones_like(wx) 
    shifted_alpha_axis = peak_factor*(alpha_axis + peak_shift)
    #print(shifted_alpha_axis[(np.int(n_alpha/2)-5):(np.int(n_alpha/2)+5)])
    #field = np.transpose(field.T/(np.sign(shifted_alpha_axis)*(shifted_alpha_axis)**2.0))
    #field = np.transpose(field.T*shifted_alpha_axis)

    # radiation far field/potential approx.
    far_field_approx = np.zeros_like(shifted_alpha_axis)
    positive_range = shifted_alpha_axis>0
    far_field_approx[positive_range] = (shifted_alpha_axis[positive_range])**(-1.0/3.0)   # for approximated potential
    #far_field_approx[positive_range] = (shifted_alpha_axis[positive_range])**(-4.0/3.0)   # for approximated field
    field = np.transpose(field.T*far_field_approx)

    # true radiation field
    #field = np.transpose(field.T*(0.5*(beta-np.sin(eta))/(1.0-beta*np.sin(eta))**3.0))

    # full lw potential
    #field = np.transpose(field.T*(beta*(1.0-beta*beta*np.cos(alpha + psi)) / psi /(2.0-beta*np.sin(eta))))

    filtered_x = np.abs(wx) < 100000.0 
    wx_filtered = wx[filtered_x]
    wy_filtered = wy[filtered_x]
    fld_filtered = field[filtered_x]

    print("fld min/max:", field.min(), field.max())

    ## unscale if necessary
    if unscale_coord :
        wx_filtered /= _gamma**3.0
        wy_filtered /= _gamma**2.0
    
    if (flatten) :
       fld1 = fld_filtered.flatten()
       flds = fld1
       if (num_fields>1) :
          flds = np.stack([fld1]*num_fields)
       return (wx_filtered.flatten(), wy_filtered.flatten(), flds)
    else :
       return wx, wy, field, psi, eta


def retard_angle_1D_ultra_relativistic(alpha_axis) :
    import numpy as np

    psi = np.zeros_like(alpha_axis)
    eta = np.zeros_like(alpha_axis)
    positive_axis = (alpha_axis>0)
    negative_axis = (alpha_axis<0)
    psi[positive_axis] = (24.0*alpha_axis[positive_axis])**(1.0/3.0)
    psi[negative_axis] = -0.5*alpha_axis[negative_axis]
    eta[positive_axis] = 0.5*(np.pi - alpha_axis[positive_axis] - psi[positive_axis]) 
    eta[negative_axis] = 0.5*(-np.pi - alpha_axis[negative_axis] - psi[negative_axis])

    return (psi, eta)
