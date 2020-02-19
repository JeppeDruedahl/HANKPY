import time
import ctypes as ct
from numba import njit, prange
import numpy as np
from scipy.sparse import spdiags, diags
from scipy.sparse.linalg import spsolve
from scipy import interpolate

from consav.misc import elapsed

# local
import modelfuncs 
import income_process
 
eps_low = 1e-12

##############
# 1. generic #
##############

def derivatives(par,sol):
    """ take numerical derivatives """

    # a. differences
    modelfuncs.diff(sol.v,sol.vaB,sol.vaF,axis=1,dxf=par.daaaf,dxb=par.daaab)
    modelfuncs.diff(sol.v,sol.vbB,sol.vbF,axis=2,dxf=par.dbbbf,dxb=par.dbbbb)
 
    # b. correct with boundary conditions
    sol.vaB[:,0,:] = 0
    sol.vaF[:,-1,:] = 1e-8
    sol.vbB[:,:,0] = -999.9
    sol.vbF[:,:,-1] = 1e-8

##################
# 2. preparation #
##################

def construct_switch(par,ast):
    """ split markov transition matrix """

    # i. generate diagonal vector and off-diagonal matrix
    par.switch_diag, ast.switch_off, par.switch_off = income_process.split_markov_matrix(par,par.z_markov)

    # ii. ensure sorted indices for UMFPACK
    ast.switch_off.sort_indices()

def prep(par,sol,solmethod):
    """ prepare sol and ast classes """

    class ast: None

    # a. construct switch matrix
    construct_switch(par,ast)
 
    # c. derivatives
    shape = (par.Nz,par.Na,par.Nb)
 
    sol.vbB = np.zeros(shape)
    sol.vbF = np.zeros(shape)
    sol.vaB = np.zeros(shape)
    sol.vaF = np.zeros(shape)
 
    sol.c_B = np.zeros(shape)
    sol.c_F = np.zeros(shape)
    sol.h_B = np.zeros(shape)
    sol.h_F = np.zeros(shape)
    sol.Hc_B = np.zeros(shape)
    sol.Hc_F = np.zeros(shape)
    sol.sbc_B = np.zeros(shape)
    sol.sbc_F = np.zeros(shape)
 
    sol.daBbB = np.zeros(shape)
    sol.daBbF = np.zeros(shape)
    sol.daFbB = np.zeros(shape)
    sol.HdaFbB = np.zeros(shape)
    sol.HdaBbF = np.zeros(shape)
    sol.HdaBbB = np.zeros(shape)
    sol.daBbF_adj = np.zeros(shape)
    sol.daBbB_adj = np.zeros(shape)
 
    # d. solution containers
    sol.v = np.zeros((par.Nz,par.Na,par.Nb))    
    sol.c = np.zeros(shape)
    sol.h = np.zeros(shape)
    sol.d = np.zeros(shape)
    sol.d_adj = np.zeros(shape)
    sol.s = np.zeros(shape)
    sol.g = np.zeros((par.Nz,par.Nab))

    # e. diagonals
    shape = (par.Nz,par.Nab)
    sol.centdiag = np.zeros(shape)
    sol.a_updiag = np.zeros(shape)
    sol.b_updiag = np.zeros(shape)
    sol.b_lowdiag = np.zeros(shape)
    sol.a_lowdiag = np.zeros(shape)
 
    # f. Q
    sol.Qps = np.zeros((par.Nz,par.Nab+1),dtype=np.int32) # element per column
 
    Nmax = par.Nab + 2*(par.Nab-1) + 2*(par.Nab-par.Nb)
    shape = (par.Nz,Nmax)
    sol.Qis = np.zeros(shape,dtype=np.int32) # indices
    sol.Qxs = np.zeros(shape) # data
 
    # pointers and pointers to pointers
    ast.Qs = [None]*par.Nz
 
    if solmethod == 'UMFPACK':
 
        ast.p_Qps = [None]*par.Nz
        ast.p_Qis = [None]*par.Nz
        ast.p_Qxs = [None]*par.Nz
 
        for iz in range(par.Nz):
 
            # pointers               
            ast.p_Qps[iz] = np.ctypeslib.as_ctypes(sol.Qps[iz])
            ast.p_Qis[iz] = np.ctypeslib.as_ctypes(sol.Qis[iz])
            ast.p_Qxs[iz] = np.ctypeslib.as_ctypes(sol.Qxs[iz])
 
        # pointers to pointers
        ast.pp_Qps = (ct.POINTER(ct.c_long)*par.Nz)(*ast.p_Qps,)
        ast.pp_Qis = (ct.POINTER(ct.c_long)*par.Nz)(*ast.p_Qis,)
        ast.pp_Qxs = (ct.POINTER(ct.c_double)*par.Nz)(*ast.p_Qxs,)
 
    # g. working memory
    sol.v = np.zeros((par.Nz,par.Na,par.Nb))
    sol.g = np.zeros((par.Nz,par.Nab))
    ast.RHS_HJB = np.zeros(par.Nzab)
    ast.RHS_KFE = np.zeros(par.Nzab)
    ast.Wi = np.zeros((par.Nz,par.Nab),dtype=np.int32)
    ast.W = np.zeros((par.Nz,5*par.Nab))
     
    # list of pointers
    ast.p_v = [None]*par.Nz # value function
    ast.p_g = [None]*par.Nz # distribution
    ast.p_RHS_HJB = [None]*par.Nz # RHS in HJB eq. sys
    ast.p_RHS_KFE = [None]*par.Nz # RHS in KF eq. sys
    ast.p_Wi = [None]*par.Nz # working memory for UMFPACK
    ast.p_W = [None]*par.Nz # working memory for UMFPACK
 
    for iz,i0,i1 in [(iz,iz*par.Nab,(iz+1)*par.Nab) for iz in range(par.Nz)]:
 
        ast.p_RHS_HJB[iz] = np.ctypeslib.as_ctypes(ast.RHS_HJB[i0:i1])
        ast.p_RHS_KFE[iz] = np.ctypeslib.as_ctypes(ast.RHS_KFE[i0:i1])
        ast.p_v[iz] = np.ctypeslib.as_ctypes(sol.v[iz].ravel())
        ast.p_g[iz] = np.ctypeslib.as_ctypes(sol.g[iz].ravel())
        ast.p_Wi[iz] = np.ctypeslib.as_ctypes(ast.Wi[iz])
        ast.p_W[iz] = np.ctypeslib.as_ctypes(ast.W[iz])  
     
    # pointers to pointers
    ast.pp_RHS_HJB = (ct.POINTER(ct.c_double)*par.Nz)(*ast.p_RHS_HJB,)
    ast.pp_RHS_KFE = (ct.POINTER(ct.c_double)*par.Nz)(*ast.p_RHS_KFE,)
    ast.pp_v = (ct.POINTER(ct.c_double)*par.Nz)(*ast.p_v,)
    ast.pp_g = (ct.POINTER(ct.c_double)*par.Nz)(*ast.p_g,)
    ast.pp_Wi = (ct.POINTER(ct.c_long)*par.Nz)(*ast.p_Wi,)
    ast.pp_W = (ct.POINTER(ct.c_double)*par.Nz)(*ast.p_W,)
 
    # precomputed symbolic matrices in UMFPACK
    ast.pp_symbolics = (ct.c_void_p*par.Nz)(*[None for _ in range(par.Nz)])
 
    return ast
 
################
# 3. solve HJB #
################
 
@njit(parallel=True,fastmath=True)
def upwind(par,sol):
    """ apply upwind scheme """
 
    # unpack
    s = sol.s
    h = sol.h
    c = sol.c
    d = sol.d
    d_adj = sol.d_adj
    h_B = sol.h_B
    h_F = sol.h_F
    c_B = sol.c_B
    c_F = sol.c_F
    Hc_B = sol.Hc_B
    Hc_F = sol.Hc_F
    sbc_B = sol.sbc_B
    sbc_F = sol.sbc_F
    daBbB = sol.daBbB
    daFbB = sol.daFbB
    daBbF = sol.daBbF
    HdaFbB = sol.HdaFbB
    HdaBbF = sol.HdaBbF
    HdaBbB = sol.HdaBbB
    daBbF_adj = sol.daBbF_adj
    daBbB_adj = sol.daBbB_adj
 
    # loop in parallel
    for iz in prange(par.Nz):
        for ia in range(par.Na):
            for ib in range(par.Nb):
                 
                a = par.grid_a[ia]
                b = par.grid_b[ib]
                z = par.grid_z[iz]
                index = (iz,ia,ib)
 
                # a. consumption and liquid savings from foc
                c_F[index],h_F[index],sbc_F[index],Hc_F[index] = modelfuncs.optimal_consumption(par,sol.vbF[index],z,b,par.Rb[index],par.w) # forwards        
                c_B[index],h_B[index],sbc_B[index],Hc_B[index] = modelfuncs.optimal_consumption(par,sol.vbB[index],z,b,par.Rb[index],par.w) # backwards
                c_0,h_0,sbc_0,Hc_0 = modelfuncs.optimal_consumption(par,-999.9,z,b,par.Rb[index],par.w) # stationary
                 
                if ib == par.Nb-1:
                    sbc_F[index] = 0
                    Hc_F[index] = -1e12               
                 
                # i. conditions
                validF = sbc_F[index] > 0
                validB = sbc_B[index] < 0
 
                # ii. consumption and liquid savings decision
                if validF and (~validB or Hc_F[index] >= Hc_B[index]) and Hc_F[index] >= Hc_0: # forward
                    c[index] = c_F[index]
                    h[index] = h_F[index]
                    s[index] = sbc_F[index]
 
                if validB and (~validF or Hc_B[index] >= Hc_F[index]) and Hc_B[index] >= Hc_0: # backwards
                    c[index] = c_B[index]
                    h[index] = h_B[index]
                    s[index] = sbc_B[index] 
                
                if ~validF and ~validB: # stationary
                    c[index] = c_0
                    s[index] = sbc_0     
                    h[index] = h_0                  
 
                # b. deposits from foc's
                daFbB[index] = modelfuncs.transaction_cost_foc(sol.vaF[index],sol.vbB[index],a,par) # a forward, b backward
                daBbF[index] = modelfuncs.transaction_cost_foc(sol.vaB[index],sol.vbF[index],a,par) # a backward, b forward
                daBbB[index] = modelfuncs.transaction_cost_foc(sol.vaB[index],sol.vbB[index],a,par) # a backward, b forward
                 
                HdaFbB[index] = sol.vaF[index]*daFbB[index] - sol.vbB[index]*(daFbB[index] + modelfuncs.transaction_cost(daFbB[index],a,par))
                daBbF_adj[index] = daBbF[index] + modelfuncs.transaction_cost(daBbF[index],a,par)
                HdaBbF[index] = sol.vaB[index]*daBbF[index] - sol.vbF[index]*daBbF_adj[index]
                daBbB_adj[index] = daBbB[index] + modelfuncs.transaction_cost(daBbB[index],a,par)
                HdaBbB[index] = sol.vaB[index]*daBbB[index] - sol.vbB[index]*daBbB_adj[index]
 
                # i. correct boundaries
                if ia == 0:
                    HdaBbF[index] = -1e12
                    HdaBbB[index] = -1e12
                if ia == par.Na-1: HdaFbB[index] = -1e12
                if ib == 0: HdaFbB[index] = -1e12
 
                # ii. conditions
                validFB = daFbB[index] > 0 and HdaFbB[index] > 0
                validBF = daBbF_adj[index] <= 0 and HdaBbF[index] > 0
                validBB = daBbB_adj[index] > 0 and daBbB[index] <= 0 and HdaBbB[index] > 0
                
                # c. find d
                if validFB and (~validBF or HdaFbB[index]>=HdaBbF[index]) and (~validBB or HdaFbB[index]>=HdaBbB[index]): d[index] = daFbB[index]
                if validBF and (~validFB or HdaBbF[index]>=HdaFbB[index]) and (~validBB or HdaBbF[index]>=HdaBbB[index]): d[index] = daBbF[index]
                if validBB and (~validFB or HdaBbB[index]>=HdaFbB[index]) and (~validBF or HdaBbB[index]>=HdaBbF[index]): d[index] = daBbB[index]
                if (~validFB and ~validBF and ~validBB): d[index] = 0
 
                # d. find d_adj
                d_adj[index] = d[index] + modelfuncs.transaction_cost(d[index],a,par)

def create_RHS_HJB(par,sol,ast,v_prev):
    """ create RHS of HJB """

    # a. utility
    u = modelfuncs.util(par,sol.c,sol.h)
    u = u.ravel()
 
    # d. total value
    v = v_prev.ravel()
    ast.RHS_HJB[:] = par.DeltaHJB*u + v + par.DeltaHJB*ast.switch_off@v
         
@njit(parallel=True,fastmath=True)
def create_diags_HJB(par,sol):
    """ create diagonals """
 
    # unpack
    centdiag = sol.centdiag
 
    a_lowdiag = sol.a_lowdiag
    a_updiag = sol.a_updiag
 
    b_lowdiag = sol.b_lowdiag
    b_updiag = sol.b_updiag
 
    # generate ltau0
    ltau0 = (par.ra+par.eta)*(par.a_max*0.999)**(1-par.ltau)
 
    # parallel loop
    for iz in prange(par.Nz):
        for ia in range(par.Na):
            for ib in range(par.Nb):
                 
                index = (iz,ia,ib)
 
                # a. set mechanical drift in a
                a = par.grid_a[ia]
                adrift = (par.ra + par.eta)*a - ltau0*a**par.ltau + par.xi*par.w
 
                # b. find diagonals in a and b space
 
                a_up = np.fmax(sol.d[index],0) + np.fmax(adrift,0)
                a_up /= par.daaaf[index]
 
                a_low = -np.fmin(sol.d[index],0) - np.fmin(adrift,0)
                a_low /= par.daaab[index]
 
                b_up = np.fmax(-sol.d_adj[index],0) + np.fmax(sol.s[index],0)
                b_up /= par.dbbbf[index]
 
                b_low = -np.fmin(-sol.d_adj[index],0) - np.fmin(sol.s[index],0)
                b_low /= par.dbbbb[index]
 
                # c. update
                i = ia*par.Nb + ib
 
                a_centdiag = a_low + a_up
                b_centdiag = b_low + b_up
                centdiag[iz,i] = 1 + par.DeltaHJB*(a_centdiag + b_centdiag + par.rho + par.eta - par.switch_diag[iz])
 
                if ia < par.Na-1: a_updiag[iz,i+par.Nb] = -par.DeltaHJB*a_up
                if ia > 0: a_lowdiag[iz,i-par.Nb] = -par.DeltaHJB*a_low
 
                if ib < par.Nb-1: b_updiag[iz,i+1] = -par.DeltaHJB*b_up
                if ib > 0: b_lowdiag[iz,i-1] = -par.DeltaHJB*b_low
 
def create_Q(par,sol,ast,solmethod):
    """ create Q matrix """

    if solmethod == 'scipy':
        create_Q_scipy(par,sol,ast,solmethod)
    elif solmethod == 'UMFPACK':
        create_Q_UMFPACK(par,sol)
        
        # equivalent:
        
        # create_Q_scipy(par,sol,ast,solmethod)

        # sol.Qps[:] = 0
        # sol.Qis[:] = 0
        # sol.Qxs[:] = 0     
           
        # for iz in range(par.Nz):
            
        #     Qz = ast.Qs[iz]
        #     N = Qz.data.size

        #     sol.Qps[iz,:] = Qz.indptr
        #     sol.Qis[iz,:N] = Qz.indices
        #     sol.Qxs[iz,:N] = Qz.data

    else:

        raise('unkwon solution method')

def create_Q_scipy(par,sol,ast,solmethod):
    """ create Q for use with scipy """

    def remove_small(x):
        I = np.abs(x) < eps_low
        y = x.copy()
        y[I] = 0
        return y

    for iz in range(par.Nz):

        #order of diagionals is important to getsorted indices
        ast.Qs[iz] = diags( diagonals=[                                        
                                        remove_small(sol.a_updiag[iz,par.Nb:]),
                                        remove_small(sol.b_updiag[iz,1:]),
                                        remove_small(sol.centdiag[iz,:]),
                                        remove_small(sol.b_lowdiag[iz,:-1]),
                                        remove_small(sol.a_lowdiag[iz,:-par.Nb]),
                                    ],
                    offsets=[par.Nb,1,0,-1,-par.Nb],
                    shape=(par.Nab,par.Nab),format='csc')

@njit(parallel=True,fastmath=True)                   
def create_Q_UMFPACK(par,sol):
    """ create Q matrix for use in UMFPACK """
    
    # unpack
    Qps = sol.Qps
    Qis = sol.Qis
    Qxs = sol.Qxs
 
    Qps[:] = 0
    Qis[:] = 0
    Qxs[:] = 0
 
    # loop in parallel
    for iz in prange(par.Nz):
         
        k = 0 # number of elements (so far)
        for col in range(par.Nab):
         
            # a upper
            if col >= par.Nb:
                x = sol.a_updiag[iz,col]
                if not np.abs(x) < eps_low:
                    Qis[iz,k] = col - par.Nb # row
                    Qxs[iz,k] = x
                    k += 1
 
            # b upper
            if col >= 1:
                x = sol.b_updiag[iz,col]
                if not np.abs(x) < eps_low:
                    Qis[iz,k] = col - 1 # row
                    Qxs[iz,k] = x
                    k += 1
 
            # center
            x = sol.centdiag[iz,col]
            if not np.abs(x) < eps_low:
                Qis[iz,k] = col # row
                Qxs[iz,k] = x
                k += 1
 
            # b lower
            if col <= par.Nab-2:
                x = sol.b_lowdiag[iz,col]
                if not np.abs(x) < eps_low:
                    Qis[iz,k] = col + 1 # row
                    Qxs[iz,k] = x
                    k += 1
 
            # a lower
            if col <= par.Nab-par.Nb-1:
                x = sol.a_lowdiag[iz,col]
                if not np.abs(x) < eps_low:
                    Qis[iz,k] = col + par.Nb # row
                    Qxs[iz,k] = x
                    k += 1  
             
            # update total number of elements so far
            Qps[iz,col+1] = k

def solve_eq_sys_HJB(par,sol,ast,solmethod,cppfile):
    """ solve equation system for HJB """

    if solmethod == 'scipy':
        for iz,i0,i1 in [(iz,iz*par.Nab,(iz+1)*par.Nab) for iz in range(par.Nz)]:
            sol.v.ravel()[i0:i1] = spsolve(ast.Qs[iz],ast.RHS_HJB[i0:i1],permc_spec='NATURAL')
    elif solmethod == 'UMFPACK':
        cppfile.solve_many(par.Nab,par.Nz,ast.pp_Qps,ast.pp_Qis,ast.pp_Qxs,
                           ast.pp_RHS_HJB,ast.pp_v,ast.pp_symbolics,ast.pp_Wi,ast.pp_W,True,True,True,par.cppthreads)
    else:
        raise Exception('unkwon solution method')

def howard_improvement_steps(par,sol,ast,solmethod,cppfile):
    """ take howard improvement steps """

    for _ in range(par.maxiter_HIS):

        # a. create RHS
        v_prev_HIS = sol.v.copy()
        create_RHS_HJB(par,sol,ast,v_prev_HIS)
        
        # b. solve
        solve_eq_sys_HJB(par,sol,ast,solmethod,cppfile)
        
        # c. distance
        HIS_dist = np.max(np.abs(sol.v-v_prev_HIS))
        if HIS_dist < par.HIStol:
            break
 
def solve_HJB(model,do_print=True,print_freq=100,solmethod='UMFPACK'):
    """ solve HJB equation """

    t0 = time.time()

    # unpack
    par = model.par
    sol = model.sol
    ast = model.ast
    cppfile = model.cppfile
      
    # solve HJB
    it = 1
    while it < par.maxiter_HJB:
 
        v_prev = sol.v.copy()
 
        # i. derivatives
        derivatives(par,sol)
         
        # ii. upwind scheme
        upwind(par,sol)
 
        # iii. RHS
        create_RHS_HJB(par,sol,ast,v_prev)     
 
        # iv. diagonals
        create_diags_HJB(par,sol)
         
        # v. construct Q
        create_Q(par,sol,ast,solmethod)
 
        # vi. solve equation system
        solve_eq_sys_HJB(par,sol,ast,solmethod,cppfile)

        # viii. howard improvement step
        if it > par.start_HIS and dist > par.stop_HIS_fac*par.HJBtol:
            howard_improvement_steps(par,sol,ast,solmethod,cppfile)
 
        # viii. check convergence
        dist = np.max(np.abs(sol.v-v_prev))
        if dist < par.HJBtol:
            if do_print: print(f' converged in {elapsed(t0)} in iteration {it}')
            break
        else:
            if do_print and (it < 10 or it%print_freq == 0):
                print(f'{it:5d}: {dist:.16f}')
            it += 1
 
    # assert converged value function monotonicity (not always fulfilled with dense grids)
    #assert np.any(np.diff(sol.v,axis = 1)<-1e-8) == 0 # monotonicity in a dimension
    #assert np.any(np.diff(sol.v,axis = 2)<-1e-8) == 0 # monotonicity in b dimension
 
    return time.time()-t0
 
################
# 4. solve KFE #
################

@njit(parallel=True,fastmath=True) 
def create_diags_KFE(par,sol):
    """ create diagonals for KFE """

    # unpack
    a_lowdiag = sol.a_lowdiag
    a_updiag = sol.a_updiag
    b_lowdiag = sol.b_lowdiag
    b_updiag = sol.b_updiag
    centdiag = sol.centdiag
 
    for iz in prange(par.Nz):
        for ia in range(par.Na):
            for ib in range(par.Nb):

                a = par.grid_a[ia]
                adrift = (par.ra + par.eta)*a - par.ltau0*a**par.ltau + par.xi*par.w
             
                a_low = -np.fmin(sol.d[iz,ia,ib] + adrift,0)/par.dab[ia]
                a_up = np.fmax(sol.d[iz,ia,ib] + adrift,0)/par.daf[ia]
                b_low = -np.fmin(sol.s[iz,ia,ib] - sol.d_adj[iz,ia,ib],0)/par.dbb[ib]
                b_up = np.fmax(sol.s[iz,ia,ib] - sol.d_adj[iz,ia,ib],0)/par.dbf[ib]
             
                # correct boundaries
                if ib == par.Nb-1:
                    a_low = -np.fmin(sol.d[iz,ia,ib-1] + adrift,0)/par.dab[ia]
                    a_up = np.fmax(sol.d[iz,ia,ib-1] + adrift,0)/par.daf[ia]
                    b_low = -np.fmin(sol.s[iz,ia,ib] - sol.d_adj[iz,ia,ib-1],0)/par.dbb[ib]
                 
                # update
                i = ib*par.Na + ia
 
                a_centdiag = a_low + a_up
                b_centdiag = b_low + b_up
                centdiag[iz,i] = 1 + par.DeltaKFE*(a_centdiag + b_centdiag + par.eta - par.switch_diag[iz])
 
                a_updiag[iz,i] = -par.DeltaKFE*a_up*par.DAB_lowdiag1[i]
                a_lowdiag[iz,i] = -par.DeltaKFE*a_low*par.DAB_updiag1[i]
 
                b_updiag[iz,i] = -par.DeltaKFE*b_up*par.DAB_lowdiag2[i]
                b_lowdiag[iz,i] = -par.DeltaKFE*b_low*par.DAB_updiag2[i]
              
    return sol

def create_B(par,sol,ast,solmethod):
    """ create B matrix """

    # think of:
    #  Qps as Bps
    #  Qis as Bis
    #  Qxs as Bxs

    # a. initialize
    if solmethod == 'UMFPACK':
        sol.Qps[:] = 0
        sol.Qis[:] = 0
        sol.Qxs[:] = 0     

    # b. construct sparse matrices
    for iz in range(par.Nz):

        ast.Qs[iz] = diags( diagonals=[                                        
                                    sol.b_lowdiag[iz,par.Na:],
                                    sol.a_lowdiag[iz,1:],
                                    sol.centdiag[iz,:],
                                    sol.a_updiag[iz,:-1],                                             
                                    sol.b_updiag[iz,:-par.Na],
                                    ],
                    offsets=[par.Na,1,0,-1,-par.Na],
                    shape=(par.Nab,par.Nab),format='csc')
        
        # pack information for UMFPACK
        if solmethod == 'UMFPACK':
            
            Qz = ast.Qs[iz]
            N = Qz.data.size

            sol.Qps[iz,:] = Qz.indptr
            sol.Qis[iz,:N] = Qz.indices
            sol.Qxs[iz,:N] = Qz.data

def solve_eq_sys_KFE(par,sol,ast,g_prev,solmethod,cppfile):
    """ solve equation system for KFE """

    # a. update g
    sol.g[:] = (np.identity(par.Nz) + par.DeltaKFE*par.switch_off).T@g_prev
    
    index = par.Na*par.Nb_neg
    sol.g[:,index] = sol.g[:,index] + par.DeltaKFE*par.eta/par.dab_tilde[par.Nb_neg,0]*(par.dab_tilde.ravel()@g_prev.T)

    # b. solve
    if solmethod == 'scipy':
        for iz in range(par.Nz):
            sol.g[iz,:] = spsolve(ast.Qs[iz],sol.g[iz,:])
    elif solmethod == 'UMFPACK':
       ast.RHS_KFE[:] = sol.g.ravel() # copy to RHS
       cppfile.solve_many(par.Nab,par.Nz,ast.pp_Qps,ast.pp_Qis,ast.pp_Qxs,
                          ast.pp_RHS_KFE,ast.pp_g,ast.pp_symbolics,ast.pp_Wi,ast.pp_W,True,True,True,par.cppthreads)
    else:
       raise Exception('unkwon solution method')

def solve_KFE(model,do_print=True,print_freq=100,solmethod='UMFPACK'):
    """ solve Kolmogorov-Forward equation """

    t0 = time.time()
    
    # unpack
    par = model.par
    sol = model.sol
    ast = model.ast
    cppfile = model.cppfile

    # a. diagonals
    create_diags_KFE(par,sol)
 
    # b. iterate
    it = 1
    while it < par.maxiter_KFE:
        
        g_prev = sol.g.copy()

        # i. construct B
        create_B(par,sol,ast,solmethod)

        # ii. solve equation 
        solve_eq_sys_KFE(par,sol,ast,g_prev,solmethod,cppfile)
         
        # iii. check convergence
        dist = np.max(np.abs(g_prev.ravel()-sol.g.ravel()))
        if dist < par.KFEtol:
            if do_print:
                print(f' converged in {elapsed(t0)} secs in iteration {it}')
            break
        else:
            if do_print and (it < 10 or it%print_freq == 0):
                print(f'{it:5d}: {dist:.16f}')
            it += 1   
         
    return time.time()-t0

##########
# 4. MPC #
##########

@njit(parallel=True,fastmath=True) 
def create_diags_cumcon(par,sol):
    """ create diagonals for cumulative consumption """

    # unpack
    a_lowdiag = sol.a_lowdiag
    a_updiag = sol.a_updiag
    b_lowdiag = sol.b_lowdiag
    b_updiag = sol.b_updiag
    centdiag = sol.centdiag
 
    for iz in prange(par.Nz):
        for ia in range(par.Na):
            for ib in range(par.Nb):

                a = par.grid_a[ia]
                adrift = (par.ra + par.eta)*a - par.ltau0*a**par.ltau + par.xi*par.w
             
                a_low = -np.fmin(sol.d[iz,ia,ib] + adrift,0)/par.dab[ia]
                a_up = np.fmax(sol.d[iz,ia,ib] + adrift,0)/par.daf[ia]
                b_low = -np.fmin(sol.s[iz,ia,ib] - sol.d_adj[iz,ia,ib],0)/par.dbb[ib]
                b_up = np.fmax(sol.s[iz,ia,ib] - sol.d_adj[iz,ia,ib],0)/par.dbf[ib]
             
                # correct boundaries
                if ib == par.Nb-1:

                    a_low = -np.fmin(sol.d[iz,ia,ib-1] + adrift,0)/par.dab[ia]
                    a_up = np.fmax(sol.d[iz,ia,ib-1] + adrift,0)/par.daf[ia]
                    b_low = -np.fmin(sol.s[iz,ia,ib] - sol.d_adj[iz,ia,ib-1],0)/par.dbb[ib]
                 
                # update
                i = ib*par.Na + ia
 
                a_centdiag = a_low + a_up
                b_centdiag = b_low + b_up
                centdiag[iz,i] = 1 + par.DeltaCUMCON*(a_centdiag + b_centdiag - par.switch_diag[iz])
 
                a_updiag[iz,i] = -par.DeltaCUMCON*a_up
                a_lowdiag[iz,i] = -par.DeltaCUMCON*a_low
 
                b_updiag[iz,i] = -par.DeltaCUMCON*b_up
                b_lowdiag[iz,i] = -par.DeltaCUMCON*b_low

def cumulative_consumption(par,sol):

    # a. create diags for sparse matrix
    create_diags_cumcon(par,sol)

    # b. define variables and containers for solution
    nsteps = int(np.round(1/par.DeltaCUMCON)) # 1 quarter
    cdvec = (np.reshape(np.array([sol.c,sol.d]),(2,par.Nz,par.Nab),order='F').swapaxes(0,1)).swapaxes(0,2)
    cdcumvec = np.zeros((par.Nab,2,par.Nz))

    # c. solve
    for _ in range(nsteps):
        
        cdcumvec += par.DeltaCUMCON*(cdvec + np.reshape(cdcumvec.reshape(2*par.Nab,par.Nz)@par.switch_off.T,(par.Nab,2,par.Nz)))
        
        # sweep over z
        for iz in range(par.Nz):
            Bz = spdiags(data=[sol.centdiag[iz,:],
                      sol.a_updiag[iz,:],
                      sol.a_lowdiag[iz,:],
                      sol.b_updiag[iz,:],
                      sol.b_lowdiag[iz,:]],
                      diags=[0,-1,1,-par.Na,par.Na],
                      m=par.Nab, n=par.Nab,
                      format='csc')
            cdcumvec[:,:,iz] = spsolve(Bz.T,cdcumvec[:,:,iz])
    
    # d. calculate one quarter cumulative expected consumption
    ccum1 = cdcumvec[:,0,:].reshape(par.Na,par.Nb,par.Nz,order='F')
    ccum1 = (ccum1.swapaxes(0,2)).swapaxes(1,2) # change ordering so this becomes unneccessary

    return ccum1

def FeynmanKac_MPC(par,sol,moms):

    # a. solve PDE
    ccum = cumulative_consumption(par,sol)

    # b. calculate MPC's
    rebamount = (500/115_000)*(moms['Y']*4)
    lmpreb = np.zeros((par.Nz,par.Na,par.Nb))
    for ia in range(par.Na):
        for iz in range(par.Nz):
            f = interpolate.interp1d(par.grid_b,ccum[iz,ia,:],fill_value='extrapolate')
            lmpreb[iz,ia,:] = f(par.grid_b+rebamount)

    MPCs = (lmpreb-ccum)/rebamount
    return MPCs