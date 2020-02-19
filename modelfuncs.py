
import numpy as np
from numba import njit
from consav import golden_section_search

##############
# 1. Utility #
##############

@njit(fastmath=True)
def util(par,c,h):
    
    # a. consumption
    if par.gamma == 1:
        util = np.log(c)
    else:
        util = c**(1-par.gamma)/(1-par.gamma)
    
    # b. labor disutility 
    varphi_ = 1+1/par.varphi
    util -= par.chi*h**varphi_/varphi_

    return util
 
@njit(fastmath=True)
def utilfoc(par,c):
    if par.gamma == 1:
        utilfoc = 1/c
    else:
        utilfoc = c**(-par.gamma)
 
    return utilfoc
 
@njit(fastmath=True)
def utilfocinv(par,c):
    if par.gamma == 1:
        utilfocinv = 1/c
    else:
        utilfocinv = c**(-1/par.gamma)
     
    return utilfocinv

########################
# 2. Transaction costs #
########################

@njit(fastmath=True)
def transaction_cost(d,a,par):
    
    chi0 = par.kappa0
    chi1 = par.kappa1**(-par.kappa2)/(1+par.kappa2)
    chi2 = 1+par.kappa2
    
    # chi0*abs(d) + chi1*abs(d/max(a,kappa3*Yss))**chi2*max(a,kappa3*Yss)
    # = (chi0*abs(d/max(a,kappa3*Yss)) + chi1*abs(d/max(a,kappa3*Yss))**chi2)*max(a,kappa3*Yss)

    a = np.fmax(a,par.kappa3)

    rel = np.abs(d)/a
    x1 = chi0*rel
    x2 = chi1*rel**chi2
    
    return (x1+x2)*a
 
@njit(fastmath=True)
def transaction_cost_foc(pa,pb,a,par):
    
    a = np.fmax(a,par.kappa3)
        
    rel = pa/pb-1
 
    if rel>par.kappa0:
        d = par.kappa1*(rel-par.kappa0)**(1/par.kappa2)
    elif rel<-par.kappa0:
        d = -par.kappa1*(-rel-par.kappa0)**(1/par.kappa2)
    else:
        d = 0
  
    return d*a

##########################
# 3. optimal Consumption #
##########################

@njit(fastmath=True)
def optimal_consumption(par,vb,z,b,rb,w):

    bdrift = (rb+par.eta)*b + par.lumpsum_transfer + par.Piz*z

    # a. not at stationary point or limit
    if vb > -999.0:

        # i. hours
        h = (par.wtilde*z*vb/par.chi)**par.varphi
        h = np.fmin(h,1) # impose max hours
        
        # ii. consumption
        c = utilfocinv(par,vb)
        s = bdrift + h*par.wtilde*z - c

        # iii. utility
        if c > 0:
            Hc = util(par,c,h) + vb*s
        else:
            Hc = -1.0e12
 
    elif vb <= -999: # at stationary point
        
        # i. hours
        if par.varphi == 1:

            hmin = bdrift/(par.wtilde*z)/2
            h = np.sqrt(hmin**2+1/par.chi)-hmin

        else: # not implemented

            pass
        
        # ii. consumption
        c = bdrift + h*par.wtilde*z
        s = 0

        # iii. utility
        if c > 0:
            Hc = util(par,c,h)
        else:
            Hc = -1.0e12
     
    return c,h,s,Hc

###########
# 4. Misc #
###########
 
def diff(V,VB,VF,axis,dxf,dxb):
 
    diff = np.diff(V,axis=axis)
 
    forward = tuple(slice(None) if i != axis else slice(None,-1) for i in range(V.ndim))
    backward = tuple(slice(None) if i != axis else slice(1,None) for i in range(V.ndim))
 
    VF[forward] = diff/dxf[forward]
    VB[backward] = diff/dxb[backward]
 
    VF[VF<1e-8] = 1e-8
    VB[VB<1e-8] = 1e-8
  
def power_spaced_grid(n,k,low,high):

    assert n > 1, 'n must be at least 2 to make grids'
     
    if n == 2:
        y = np.append(low, high)
    else:
        y = low + (high-low)*np.linspace(0,1,n)**(1/k)   

    return y