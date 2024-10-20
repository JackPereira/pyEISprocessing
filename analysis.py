import numpy as np
from scipy import integrate
import warnings
import math

'''
======================================
Transformation functions
======================================
'''
def complex_to_polar(*args):
    '''
    Accepts complex input arrays and returns corresponding magnitude, phase angle arrays

    Parameters
    ----------
    *args : np.ndarrays, dtype = np.floatingcomplex
        Arbitrary number of complex arrays.

    Returns
    -------
    2*(n_*args) np.ndarrays, dtype = np.floating
        Returns r_1, phi_1, r_2, phi_2,... r_n,phi_n where (1,2,...n) are input arrays.
        r_n - magnitude of Z_n
        phi_n - phase angle of Z_n

    '''
    out = []
    for arg in args:
        r = np.abs(arg)
        phi = np.angle(arg)
        out.append((r,phi))
    return (val for tup in out for val in tup)

def polar_to_complex(*args):
    '''
    Accepts magnitude, phase angle arrays and returns corresponding complex arrays.

    Parameters
    ----------
    *args : np.ndarrays, dtype = np.floating
        Inputs must be formatted as per the return of complex_to_polar().
        Ex: polar_to_complex(r_1, phi_1, r_2, phi_2) will return Z_1, Z_2

    Returns
    -------
    (n_*args)/2 np.arrays, dtype = np.floatingcomplex
        See complex_to_polar() input for this output format.

    '''
    out = []
    r = args[::2]
    phi = args[1::2]
    
    for i in range(len(r)):
        complex_vals = r[i]*np.exp(1j*phi[i]) 
        out.append(complex_vals)
    return (val for val in out)

'''
======================================
Helper functions.
======================================
'''
def sort_indicies(vallist):
    '''
    Sorting function used in distributed_select_best().

    Parameters
    ----------
    vallist : list
        Input list of score values.

    Returns
    -------
    sorted_list : list
        List of indicies sorted from highest to lowest scores.
        
    '''
    idx = range(len(vallist))
    sorted_list = sorted(idx,key= lambda x: vallist[x],reverse=True)
    return sorted_list

def update_decay(decaylist, idx, order, alpha):
    '''
    Updates decay list used in distributed_select_best(). See the aforementioned
    for more details.

    Parameters
    ----------
    decaylist : list
        List of current decay values.
    idx : int
        Decaylist index of central value that decay is propagating from.
    order : int
        Hard cutoff for propagation distance.
    alpha : float
        Decay parameter. Decay is (current decay)*alpha^(1/n), where n is distance
        from the central value index.

    Returns
    -------
    decay_updated : list
        List of updated decay values.

    '''
    decay_updated = decaylist
    if idx != 0:
        l_ord = 1
        l_idx = idx - 1
        while l_ord <= order:
            if l_idx == -1:
                break
            else:
                decay_updated[l_idx] *= math.pow(alpha,1/(l_ord))
                l_idx -= 1
                l_ord += 1
    if idx != len(decaylist) - 1:
        u_ord = 1
        u_idx = idx + 1
        while u_ord <= order:
            if u_idx == len(decaylist):
                break
            else:
                decay_updated[u_idx] *= math.pow(alpha,1/(u_ord))
                u_idx += 1
                u_ord += 1
    return decay_updated

def pearson_corr(x,y):
    xysum = sum((x-np.mean(x))*(y-np.mean(y)))
    xsum = np.sum((x-np.mean(x))**2)
    ysum = np.sum((y-np.mean(y))**2)
    return xysum / np.sqrt(xsum*ysum)

'''
======================================
Functions for evaluating EIS dataset characteristics,
e.g., the location and area of peaks.
======================================
'''
def find_eis_peaks(EISdataset, n_peaks, cutoff_frac = 0.1,get_peaks=True,get_bounds=False,
                   get_area=False,print_stats=False,separate_returns=False):
    '''
    Finds values associated with EIS peaks and their bounds. Integrated into
    EISdataset() as 

    Parameters
    ----------
    EISdataset : dataset_processing.EISdataset()
        Dataset object holding Z data.
    n_peaks : int
        Number of peaks to search for.
    cutoff_frac : float, optional
        Hard cutoff for size of peaks, expressed as a fraction of the maximum imaginary
        component in a spectrum. Values under cutoff_frac*max will not be considered
        valid peaks.
    get_peaks : bool, optional
        If true, return the real, imaginary, and frequency values associated with peaks.
    get_bounds : bool, optional
        If True, return the real, imaginary, and frequency values associated with 
        lower and upper peak bounds. 
    get_area : bool, optional
        If True, return integrated peak area.
    print_stats : bool, optional
        If True, print statistics about peak finding and area (if applicable)
    separate_returns : bool, optional
        NOT IMPLEMENTED YET

    '''
    if not get_peaks and not get_area:
        raise Exception('At least one of get_peaks or get_area must be True')
    ndata = np.size(EISdataset.f_gen_)
    Zim_neg = -1*np.imag(EISdataset.Z_var_.copy())
    nsets = np.size(Zim_neg,axis=0)
    cutoff = cutoff_frac*np.max(Zim_neg,axis=1)
    freq = EISdataset.f_gen_
    dg = np.gradient(Zim_neg,EISdataset.f_gen_,axis=1)
    if get_peaks:
        peak_out = []
        peak_tags = []
    if get_bounds:
        bounds_out = []
        bound_tags = []
    if get_area:
        area_out = []
        a_tags = []
    if print_stats:
        lb_bottomout = 0
        ub_topout = 0
        fails = 0
    current_peak = 0
    while current_peak < n_peaks:
        p_idx = np.argmax(Zim_neg,axis=1)
        if get_peaks:
            p_re = np.zeros((nsets,1))
            p_im = np.zeros((nsets,1))
            p_freq = np.zeros((nsets,1))
        if get_bounds:
            lb_re = np.zeros((nsets,1))
            lb_im = np.zeros((nsets,1))
            lb_freq = np.zeros((nsets,1))
            ub_re = np.zeros((nsets,1))
            ub_im = np.zeros((nsets,1))
            ub_freq = np.zeros((nsets,1))
        if get_area:
            area = np.zeros((nsets,1))
    
        for n in range(nsets):
            if Zim_neg[n,p_idx[n]] < cutoff[n]:
                if print_stats:
                    fails += 1
                if n < nsets - 1:
                    n+=1
                else:
                    break
            if get_peaks:
                p_re[n] = np.real(EISdataset.Z_var_[n,p_idx[n]])
                p_im[n] = np.imag(EISdataset.Z_var_[n,p_idx[n]])
                p_freq[n] = freq[p_idx[n]]
            lb_idx = p_idx[n]
            ub_idx = p_idx[n]
            lb_found = False
            ub_found = False
            while not lb_found:
                lb_idx -= 1
                if lb_idx == 0:
                        if get_bounds:
                            lb_re[n] = np.real(EISdataset.Z_var_[n,lb_idx])
                            lb_im[n] = np.imag(EISdataset.Z_var_[n,lb_idx])
                            lb_freq[n] = freq[lb_idx]
                        if print_stats:
                            lb_bottomout += 1
                        lb_found = True
                else:
                    if dg[n,lb_idx] < 0 or abs(dg[n,lb_idx]) < 0.1:
                        if get_bounds:
                            lb_re[n] = np.real(EISdataset.Z_var_[n,lb_idx])
                            lb_im[n] = np.imag(EISdataset.Z_var_[n,lb_idx])
                            lb_freq[n] = freq[lb_idx]
                        lb_found = True
            while not ub_found:
                ub_idx += 1
                if ub_idx == ndata:
                    if get_bounds:
                        ub_re[n] = np.real(EISdataset.Z_var_[n,ub_idx - 1])
                        ub_im[n] = np.imag(EISdataset.Z_var_[n,ub_idx - 1])
                        ub_freq[n] = freq[ub_idx - 1]
                        if print_stats:
                            ub_topout += 1
                        ub_found = True
                elif dg[n,ub_idx] > 0 or abs(dg[n,lb_idx]) < 0.1:
                    if get_bounds:
                        ub_re[n] = np.real(EISdataset.Z_var_[n,ub_idx])
                        ub_im[n] = np.imag(EISdataset.Z_var_[n,ub_idx])
                        ub_freq[n] = freq[ub_idx]
                        ub_found = ub_idx
                        
            if get_area:
                area[n] = integrate.trapezoid(Zim_neg[n,lb_idx:ub_idx],x=freq[lb_idx:ub_idx])

            Zim_neg[n,lb_idx:ub_idx] = 0
            
        tl = 'Peak {}'.format(current_peak+1)
        if get_peaks:
            curvals = [p_re,p_im,p_freq]
            peak_out.extend(curvals)
            peak_tags.extend([tl + ' real',tl + ' imaginary',tl + ' freq'])
        if get_bounds:
            curvals = [lb_re,lb_im,lb_freq,ub_re,ub_im,ub_freq]
            bounds_out.extend(curvals)
            bound_tags.extend([tl + ' lower real',tl + ' lower imaginary',tl + ' lower freq',tl + ' upper real',tl + ' upper imaginary',tl + ' upper freq'])
        if get_area:
            area_out.append(area)
            a_tags.append(tl + ' area')
                
        current_peak += 1
        
        if print_stats:
            print('\n-------------\nPeak {}\nPeaks found: [{}]/[{}]\nFailed to find {} peaks above cutoff\nPeaks cutoff by lower freq range: [{}]\nPeaks cutoff by upper freq range: [{}]\n'.format(
                current_peak,nsets-lb_bottomout-ub_topout-fails,nsets,fails,lb_bottomout,ub_topout)
            )
            if get_area:
                print('Average area: {:.3e}\nStd dev: {:.3e}\nMax:{:.3e}\nMin:{:.3e}'.format(
                np.mean(area), np.std(area), np.max(area), np.min(area))
            )
        
        if np.all(Zim_neg < np.max(cutoff)):
            print('Only {} peak(s) found with given cutoff fraction; stopping early'.format(current_peak)) 
            # Displays 1 over iter, but that adjusts 0 index for readability
            break

        if not separate_returns:
            if current_peak == 1:
                returns = [x for x in (peak_out,bounds_out,area_out) if x is not None]
                tags = [y for y in (peak_tags,bound_tags,a_tags) if y is not None]
            elif current_peak < n_peaks:
                returns.extend([x for x in (peak_out,bounds_out,area_out) if x is not None])
                tags.extend([y for y in (peak_tags,bound_tags,a_tags) if y is not None])
                
    if not separate_returns:
        tags = [x for y in tags for x in y]
        tags = np.array(tags)
        returns = np.transpose(np.squeeze(np.vstack(returns)))
        return returns, tags
    else:
        raise NotImplementedError('Separated returns not implemented yet.')

def get_eis_extrema(EISdataset, include_peak = False):
    
    Zim_neg = -1*np.imag(EISdataset.Z_var_.copy())
    nsets = np.size(Zim_neg,axis=0)
    dg = np.gradient(Zim_neg,EISdataset.f_gen_,axis=1)
    mins = np.zeros((nsets,4))
    
    pidx = np.argmax(Zim_neg,axis=1)
    istop = np.zeros((np.size(pidx),1),dtype=bool)
    isbot = np.zeros((np.size(pidx),1),dtype=bool)
    istop[pidx == len(EISdataset.f_gen_)] == True
    isbot[pidx == 0] == True
    # pidx[pidx == len(EISdataset.f_gen_)] == len(EISdataset.f_gen_) - 1
    # pidx[pidx == 0] == 1

    for n in range(nsets):
        if not isbot[n]:
            hf_minidx = np.argmin(dg[:,:pidx[n]],axis=1)
            mins[n,0] = np.real(EISdataset.Z_var_[n,hf_minidx[n]])
            mins[n,1] = np.imag(EISdataset.Z_var_[n,hf_minidx[n]])
        if not istop[n]:
            lf_minidx = np.argmin(dg[:,pidx[n]:],axis=1)
            mins[n,2] = np.real(EISdataset.Z_var_[n,lf_minidx[n]])
            mins[n,3] = np.imag(EISdataset.Z_var_[n,lf_minidx[n]])
    tags = ['High frequency minimum, Re','High frequency minimum, Im','Low frequency minimum, Re','Low frequency minimum, Im']
    
    return mins,tags

def get_intercepts(EISdataset):
    
    Zim_neg = -1*np.imag(EISdataset.Z_var_.copy())
    nsets = np.size(Zim_neg,axis=0)
    intercepts = np.zeros((nsets,5))
'''
======================================
Functions for evaluating DRT dataset characteristics,
e.g., the location and area of peaks.
======================================
'''
def find_drt_peaks(DRTdataset, n_peaks, cutoff_frac = 0.1, get_gamma = True, get_tau = True, get_area = True,
                   separate_returns = False, print_stats = False, der = 1):

    ndata = np.size(DRTdataset.tau_)
    valid_g = DRTdataset.DRTdata_.copy()
    nsets = np.size(valid_g,axis=0)
    cutoff = cutoff_frac*np.max(valid_g,axis=1)
    tau = DRTdataset.tau_
    
    if der < 1:
        der == 1
        warnings.warn('Invalid derivative. Valid inputs are der = 1 or = 2. Defaulting to der = 1')
    dg = np.gradient(valid_g,tau,axis=1)
    for i in range(1,der+1):
        dg = np.gradient(dg,tau,axis=1)

    if get_gamma:
        gamma_out = []
        g_tags = []
    else:
        gamma_out = None
        g_tags = None
    if get_tau:
        tau_out = []
        t_tags = []
    else:
        tau_out = None
        t_tags = None
    if get_area:
        area_out = []
        a_tags = []
    else:
        area_out = None
        a_tags = None
    
    current_peak = 0
    while current_peak < n_peaks:
        p_idx = np.argmax(valid_g,axis=1)
        if print_stats:
            lb_bottomout = 0
            ub_topout = 0
            fails = 0
        
        if get_gamma:
            lb = np.zeros((nsets,1))
            ub = np.zeros((nsets,1))
            p = np.zeros((nsets,1))
        if get_tau:
            lb_tau = np.zeros((nsets,1))
            ub_tau = np.zeros((nsets,1))
            p_tau = np.zeros((nsets,1))
        if get_area:
            area = np.zeros((nsets,1))
            
        for n in range(nsets):
            if valid_g[n,p_idx[n]] < cutoff[n]:
                if print_stats:
                    fails += 1
                if n < nsets - 1:
                    n+=1
                else:
                    break
            if get_gamma:
                p[n] = valid_g[n,p_idx[n]]
            if get_tau:
                p_tau[n] = tau[p_idx[n]]
            lb_idx = p_idx[n]
            ub_idx = p_idx[n]
            lb_found = False
            ub_found = False
            while not lb_found:
                lb_idx -= 1
                if der == 1:
                    if dg[n,lb_idx] < 0 or abs(dg[n,lb_idx]) < 0.1:
                        if get_gamma:
                            lb[n] = valid_g[n,lb_idx]
                        if get_tau:
                            lb_tau[n] = tau[lb_idx]
                        lb_found = True
                else:
                    if dg[n,lb_idx] > 0 or abs(dg[n,lb_idx]) < 0.1:
                        if get_gamma:
                            lb[n] = valid_g[n,lb_idx]
                        if get_tau:
                            lb_tau[n] = tau[lb_idx]
                        lb_found = True
                if lb_idx == 0:
                        if get_gamma:
                            lb[n] = valid_g[n,lb_idx]
                        if get_tau:
                            lb_tau[n] = tau[lb_idx]
                        if print_stats:
                            lb_bottomout += 1
                        lb_found = True
            while not ub_found:
                ub_idx += 1
                if dg[n,ub_idx] > 0 or abs(dg[n,lb_idx]) < 0.1:
                    if get_gamma:
                        ub[n] = valid_g[n,ub_idx]
                    if get_tau:
                        ub_tau[n] = tau[ub_idx]
                    ub_found = True
                elif ub_idx == ndata - 1:
                    if get_gamma:
                        ub[n] = valid_g[n,ub_idx]
                    if get_tau:
                        ub_tau[n] = tau[ub_idx]
                    if print_stats:
                        ub_topout += 1
                    ub_found = True
            if get_area:
                area[n] = integrate.trapezoid(valid_g[n,lb_idx:ub_idx],x=tau[lb_idx:ub_idx])

            valid_g[n,lb_idx:ub_idx] = 0

        
        tl = 'Peak {}'.format(current_peak+1)
        if get_gamma:
            curvals = [p,lb,ub]
            gamma_out.extend(curvals)
            g_tags.extend([tl + ' top gamma',tl + ' lb gamma',tl + ' ub gamma'])
        if get_tau:
            curvals = [p_tau,lb_tau,ub_tau]
            tau_out.extend(curvals)
            t_tags.extend([tl + ' top tau',tl + ' lb tau',tl + ' ub tau'])
        if get_area:
            area_out.append(area)
            a_tags.append(tl + ' area')
            
        current_peak += 1
        if print_stats:
            print('\n-------------\nPeak {}\nPeaks found: [{}]/[{}]\nFailed to find {} peaks above cutoff\nPeaks cutoff by lower tau range: [{}]\nPeaks cutoff by upper tau range: [{}]\n'.format(
                current_peak,nsets-lb_bottomout-ub_topout-fails,nsets,fails,lb_bottomout,ub_topout)
            )
            if get_area:
                print('Average area: {:.3e}\nStd dev: {:.3e}\nMax:{:.3e}\nMin:{:.3e}'.format(
                np.mean(area), np.std(area), np.max(area), np.min(area))
            )
        
        if np.all(valid_g < np.max(cutoff)):
            print('Only {} peak(s) found with given cutoff fraction; stopping early'.format(current_peak)) 
            # Displays 1 over iter, but that adjusts 0 index for readability
            break
        

    if not separate_returns:
        valid_returns = [x for x in (gamma_out,tau_out,area_out) if x is not None]
        valid_tags = [y for y in (g_tags,t_tags,a_tags) if y is not None]
        tags = []
        tags.extend(valid_tags)
        tags = [x for y in tags for x in y]
        tags = np.array(tags)

        return np.transpose(np.squeeze(np.vstack(valid_returns))), tags
    else:
        raise NotImplementedError('Separated returns not implemented yet. I will get to it I swear (I say to myself at 3 AM)')
