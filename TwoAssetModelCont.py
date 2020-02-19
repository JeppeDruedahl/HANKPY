#-*- coding: utf-8 -*-
"""TwoAssetModelCont

"""

##############
# 1. imports #
##############

import time
import numpy as np
from numba import jitclass, njit, prange
from numba import double, int32, boolean
 
# local packages
import UMFPACK
from consav.misc import elapsed
from consav import ModelClass # baseline model classes

# local modules
import modelfuncs
import income_process
import solve
import equilibrium
import estimate

class TwoAssetModelContClass(ModelClass):
    
    #########
    # setup #
    #########

    def __init__(self,name='baseline',load=False,solmethod='UMFPACK',like_HANK=False,**kwargs):
        """ called when model is created
        
        name: used when saving
        load: True/False
        solmethod: UMFPACK/scipy
        like_HANK: use same setup as in HANK
        **kwargs: additional parameter changes

        """

        # a. name and additional specifications
        self.name = name 
        self.solmethod = solmethod 
        
        self.moms = {}
        self.timings = {}
        self.savelist = ['moms','timings']

        # savelist -> stuff in self saved beyond par, sol and sim
                
        # b. define subclasses
        parlist = [
            
            # a. equlibrium objects
            ('ra',double),
            ('rb',double),
            ('Pi',double),
            ('w',double),

            # b. firms and production
            ('alpha',double),
            ('delta',double),
            ('vareps',double),
            ('zeta',double),
            ('omega',double),
            
            # c. preferences
            ('rho',double),
            ('eta',double),
            ('gamma',double),
            ('chi',double),
            ('varphi',double),
            
            # d. income process
            ('lambda1',double),
            ('lambda2',double),
            ('beta1',double),
            ('beta2',double),
            ('sigma1',double),
            ('sigma2',double),
            ('rho1',double),
            ('rho2',double),
            ('dt',double),

            # e. borrowing
            ('kappa',double),
            ('b_min',double),

            # f. adjustment costs
            ('kappa0',double),
            ('kappa1',double),
            ('kappa2',double),
            ('kappa3',double),
            ('xi', double),
                        
            # g. government
            ('labtax',double),
            ('lumpsum_transfer',double),
            
            # h. grids

            # jump drif
            ('z1_width',double),
            ('z2_width',double),
            ('kz_1',double),
            ('kz_2',double),
            ('Nz1',int32),
            ('Nz2',int32),  

            # assets
            ('b_max',double),
            ('Nb_neg',int32),
            ('Nb_pos',int32),
            ('k_b_neg',double),
            ('k_b_pos',double),

            ('a_min',double),
            ('a_max',double),
            ('Na',int32),
            ('k_a',double),
            
            # i. misc
            ('DeltaIncome',double),            
            ('DeltaHJB',double),
            ('DeltaKFE',double),
            ('DeltaCUMCON',double),            
            ('maxiter_HJB',int32),
            ('maxiter_KFE',int32),
            ('maxiter_HIS',int32),
            ('start_HIS',double),
            ('stop_HIS_fac',double),
            ('HJBtol',double),
            ('KFEtol',double),
            ('HIStol',double),
            ('dmax',int32),
            ('cmin',int32),
            ('ltau',double),
            ('cppthreads',int32),

            # used in create_grids
            ('rb_borrow',double),
            ('wtilde',double),
            ('Piz',double),
            ('Nz',int32),
            ('Nb',int32),
            ('Nab',int32),
            ('Nzab',int32),
            ('ltau0',double),

            ('grid_a',double[:]),
            ('daf',double[:]),
            ('dab',double[:]),

            ('grid_b',double[:]),
            ('dbf',double[:]),
            ('dbb',double[:]),

            ('grid_z_log',double[:]),
            ('z_markov',double[:,:]),
            ('grid_z',double[:]),
            ('z_dist',double[:]),

            ('grid_z1',double[:]),
            ('grid_z2',double[:]),
            ('z1_markov',double[:,:]),
            ('z2_markov',double[:,:]),
            ('z1dist',double[:]),
            ('z2dist',double[:]),
            ('z_log_scale',double),

            ('zzz',double[:,:,:]),
            ('aaa',double[:,:,:]),
            ('bbb',double[:,:,:]),
            ('daaaf',double[:,:,:]),
            ('dbbbf',double[:,:,:]),
            ('daaab',double[:,:,:]),
            ('dbbbb',double[:,:,:]),
            ('Rb',double[:,:,:]),

            ('da_tilde',double[:]),
            ('db_tilde',double[:]),
            ('dab_tilde',double[:,:]),

            ('DAB_updiag1',double[:]),
            ('DAB_lowdiag1',double[:]),
            ('DAB_updiag2',double[:]),
            ('DAB_lowdiag2',double[:]),

            # used in solve
            ('switch_diag',double[:]),
            ('switch_off',double[:,:]),            
            
        ]

        sollist = [

            ('v',double[:,:,:]),
            ('g',double[:,:]),
            ('c_0',double[:,:,:]),
        
            ('vbB',double[:,:,:]),
            ('vbF',double[:,:,:]),
            ('vaB',double[:,:,:]),
            ('vaF',double[:,:,:]),
        
            ('c_B',double[:,:,:]),
            ('c_F',double[:,:,:]),
            ('h_B',double[:,:,:]),
            ('h_F',double[:,:,:]),
            ('Hc_B',double[:,:,:]),
            ('Hc_F',double[:,:,:]),
            ('sbc_B',double[:,:,:]),
            ('sbc_F',double[:,:,:]),
        
            ('daBbB',double[:,:,:]),
            ('daFbB',double[:,:,:]),
            ('daBbF',double[:,:,:]),
            ('HdaBbB',double[:,:,:]),
            ('HdaFbB',double[:,:,:]),
            ('HdaBbF',double[:,:,:]),
            ('daBbF_adj',double[:,:,:]),
            ('daBbB_adj',double[:,:,:]),  

            ('c',double[:,:,:]),        
            ('d',double[:,:,:]),
            ('d_adj',double[:,:,:]),
            ('s',double[:,:,:]),
            ('h',double[:,:,:]),
        
            ('Qps',int32[:,:]),
            ('Qis',int32[:,:]),
            ('Qxs',double[:,:]),
        
            ('a_updiag',double[:,:]),
            ('b_updiag',double[:,:]),
            ('centdiag',double[:,:]),
            ('b_lowdiag',double[:,:]),
            ('a_lowdiag',double[:,:]),
        
            ('a_updiag_trans',double[:,:,:]),
            ('b_updiag_trans',double[:,:,:]),
            ('centdiag_trans',double[:,:,:]),
            ('b_lowdiag_trans',double[:,:,:]),
            ('a_lowdiag_trans',double[:,:,:]),

        ]

        simlist = []

        # c. create subclasses
        self.par,self.sol,self.sim = self.create_subclasses(parlist,sollist,simlist)

        # d. load
        if like_HANK:
            self.setup_like_HANK()
        else:
            self.setup(**kwargs)

        if load:
            self.load()
            
        # e. setup UMFPACK
        self.setup_UMFPACK()

    def setup_UMFPACK(self,force_build=False,force_compile=False):
        """ compile UMFPACk """

        filename = 'UMFPACK'
        UMFPACK.build_cpp_project(filename,force=force_build)
        UMFPACK.compile_cpp(filename,force=force_compile)
        self.cppfile = UMFPACK.link(filename)
        
    def setup_like_HANK(self,**kwargs):
        """ choose parameters like in HANK """

        par = self.par

        # a. equilibrium objects
        par.rb = 0.005 # interest rate on bonds
        par.ra = 0.014228408698221 # interest rate on illiquid assets 
        par.Pi = 0.170927454266510 # total profits
        par.w = 1.814357327040704 # wage        
        
        # b. firms and production function
        par.alpha = 0.33 # Cobb-Douglas parameter
        par.delta = 0.07/4 # depreciation rate
        par.vareps = 10 # Dixit-Stiglitz elasticty
        par.zeta = 0.997047259125388 # labor efficiency scale
        par.omega = par.alpha # share of profits distributed to illiquid account

        # c. preferences
        par.rho = 0.012728925130000 # subjective time preference rate
        par.eta = 1/(4*45) # death rate
        par.gamma = 1 # inverse elasticity of intertemporal substitution
        par.chi = 2.243356333032124 # disutility of labor scale
        par.varphi = 1 # frisch elasticity 
    
        # d. income process
        par.lambda1 = 7.987000048160553E-002
        par.lambda2 = 6.560000125318766E-003
        par.beta1 = 0.761600017547607
        par.beta2 = 9.390000253915787E-003
        par.sigma1 = 1.73520004749298
        par.sigma2 = 1.52590000629425
        par.rho1 = 0
        par.rho2 = 0
        par.dt = 0.25 # quarterly income process
    
        # e. borrowing
        par.kappa = 0.0148846 # borrowing wedge liquid asset
        par.b_min = -1 # liquid asset borrowing constraint
            
        # f. adjustment costs
        par.kappa0 = 0.04383
        par.kappa2 = 0.40176
        par.kappa1 = ((1-par.kappa0)*(1+par.kappa2))**(-1/par.kappa2)
        par.kappa3 = 0.0219
        par.xi = 0 # automatic deposits

        # g. government
        par.labtax = 0.3 # labor income tax
        par.lumpsum_transfer = 0.104347826086957 # lump-sum transfer
        
        # h. grids

        # jump-drift 
        par.z1_width = 3.78503497352302 # width income process 1
        par.z2_width = 5.69800327154487 # width income process 2
        par.kz_1 = 0.792311684654520 # curvature symmetric power grid, income process 1
        par.kz_2 = 0.641191531992662 # curvature symmetric power grid, income process 2
        par.Nz1 = 3 # number of grid points income process 1, must be odd
        par.Nz2 = 11 # number of grid points income process 2, must be odd
        par.z_log_scale = 0.85 # log-productivity scale 

        # assets
        par.b_max = 40  # max liquid asset grid point
        par.Nb_neg = 10 # number of liquid asset grid points, b < 0
        par.Nb_pos = 40 # number of liquid asset grid points, b >= 0
        par.k_b_neg = 0.40 # curvature of liquid asset power spaced grid, b < 0.
        par.k_b_pos = 0.35 # curvature of liquid asset power spaced grid, b >= 0.
    
        par.a_min = 0 # min illiquid asset grid point
        par.a_max = 2000 # max illiquid asset grid point
        par.Na = 40 # number of illiquid asset grid points
        par.k_a = 0.15 # curvature of illiquid asset power spaced grid.
    
        # i. misc
        par.DeltaIncome = 1 # delta timestep in income process        
        par.DeltaHJB = 1e6 # delta timestep in HJB algorithm
        par.DeltaKFE = 1e6 # delta timestep in KF algorithm
        par.DeltaCUMCON = 0.01 # delta timestep in cumulative consumption iterations
        par.maxiter_HJB = 500 # max number of iterations in the HJB algorithm
        par.maxiter_KFE = 2000 # max number of iterations in the KF algorithm
        par.maxiter_HIS = 2 # max number of howard improvement step iterations
        par.HJBtol = 1e-8 # convergence criterion of inner HJB loop
        par.KFEtol = 1e-12 # convergence criterion of KFE loop
        par.HIStol = 1e-5 # tolerance for howard improvement step loop
        par.start_HIS = np.inf # number of HJB iterations before howard improvements
        par.stop_HIS_fac = 10 #  factor used to determined when to stop howard improvements

        par.dmax = 1e10  # maximum deposit, for numerical stability while converging
        par.cmin = 1e-5 # minimum consumption
        par.ltau = 15 # if ra>>rb, impose tax on ra*a at high a, otherwise some households accumulate infinite illiquid wealth
        par.cppthreads = 20 # number of threads used in C++ program

    def setup(self,**kwargs):
        """ choose baseline parameters """

        par = self.par

        # a. start from HANK
        self.setup_like_HANK()

        # b. changes
        par.start_HIS = 3
        par.HJBtol = 1e-5
        par.KFEtol = 1e-6  
    
        # c. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val

    def create_grids(self):
        """ create grids (automatically called with .solve()) """

        par = self.par
        assert par.varphi == 1, 'only implemented for varphi = 1'

        # a. composite parameters

        # model
        par.rb_borrow = par.rb + par.kappa # borrowing interest rate
        par.wtilde = ((1-par.xi)-par.labtax)*par.w
        par.Piz = ((1-par.xi)-par.labtax)*(1-par.omega)*par.Pi/par.zeta

        assert par.ra - 1/(par.kappa1**(-par.kappa2)/(1+par.kappa2)) <= 0

        # grids
        par.Nz = par.Nz1*par.Nz2
        par.Nb = par.Nb_pos + par.Nb_neg
        par.Nab = par.Nb*par.Na       
        par.Nzab = par.Nz*par.Nb*par.Na
        
        # misc
        par.ltau0 = (par.ra+par.eta)*(par.a_max*0.999)**(1-par.ltau)

        # b. construct grids

        # grid_a
        par.grid_a = modelfuncs.power_spaced_grid(par.Na,par.k_a,par.a_min,par.a_max)
        if par.Na > 10: # evenly spaced in the beginning
            par.grid_a[0:9] = np.array(range(0,8+1))*par.grid_a[9]/9

        par.daf = np.append(np.diff(par.grid_a),1) # forward step sizes
        par.dab = np.append(1,np.diff(par.grid_a)) # backward step sizes

        # grid_b
        assert par.Nb_neg%2 == 0
        
        nbl = -par.lumpsum_transfer/(par.rb_borrow + par.eta) # natural borrowing limit
        abl = np.fmax(nbl+par.cmin,par.b_min) # actual borrowing limit

        grid_b_neg = modelfuncs.power_spaced_grid(par.Nb_neg/2+1,par.k_b_neg,abl,abl/2) # negative part
        grid_b_neg_rev = grid_b_neg[1:-1]
        grid_b_neg = np.append(grid_b_neg,abl - grid_b_neg_rev[::-1])
        grid_b_pos = modelfuncs.power_spaced_grid(par.Nb_pos,par.k_b_pos,0,par.b_max) # positive part

        par.grid_b = np.append(grid_b_neg,grid_b_pos).T
        par.dbf = np.append(np.diff(par.grid_b),1) # forward step sizes
        par.dbb = np.append(1,np.diff(par.grid_b)) # backward step sizes

        # grid_z
        par.grid_z_log,_,par.z_markov,par.z_dist = income_process.construct_jump_drift(par)   
        par.grid_z_log = par.grid_z_log/(1+par.z_log_scale*par.varphi)
        par.grid_z = np.exp(par.grid_z_log)
        par.grid_z = par.zeta*par.grid_z/np.sum(par.z_dist*par.grid_z)
    
        # full grids
        par.zzz,par.aaa,par.bbb = np.meshgrid(par.grid_z,par.grid_a,par.grid_b,indexing='ij')
        _,par.daaaf,par.dbbbf = np.meshgrid(par.grid_z,par.daf,par.dbf,indexing='ij')
        _,par.daaab,par.dbbbb = np.meshgrid(par.grid_z,par.dab,par.dbb,indexing='ij')
        par.Rb = par.rb*(par.bbb>0) + par.rb_borrow*(par.bbb<=0)

        # tilde grids for Kolmogorov-Forward correction
        par.db_tilde = 0.5*(par.dbb + par.dbf)
        par.db_tilde[0] = 0.5*par.dbf[0] # must go forwards at min
        par.db_tilde[par.Nb-1] = 0.5*par.dbb[par.Nb-1] # must go backwards at max
        par.da_tilde = 0.5*(par.dab + par.daf)
        par.da_tilde[0] = 0.5*par.daf[0]
        par.da_tilde[par.Na-1] = 0.5*par.dab[par.Na-1]
    
        da_tilde = np.expand_dims(par.da_tilde,axis=0) # add dimension for np.dot
        db_tilde = np.expand_dims(par.db_tilde,axis=1) # add dimension for np.dot
        par.dab_tilde = np.dot(db_tilde,da_tilde)
        dab_tilde_vec = par.dab_tilde.ravel()
        
        DAB_tilde = dab_tilde_vec.reshape((1,par.Nab))/dab_tilde_vec.reshape((par.Nab,1))
        
        # relevant diagonals DAB_tilde
        par.DAB_updiag1 = np.append(0,np.diag(DAB_tilde,1))
        par.DAB_lowdiag1 = np.append(np.diag(DAB_tilde,-1),0)
        par.DAB_updiag2 = np.append(np.zeros(par.Na),np.diag(DAB_tilde,par.Na))
        par.DAB_lowdiag2 = np.append(np.diag(DAB_tilde,-par.Na),np.zeros(par.Na))

    def solve(self,do_print=True,print_freq=100,v0=None,g0=None,solmethod=None):
        """ solve model """

        par = self.par
        sol = self.sol
        
        # a. create grids
        t0 = time.time()
        self.create_grids()
        if do_print: 
            print(f'Grids created in {elapsed(t0)}')

        # b. set solution method
        if solmethod is None: 
            solmethod = self.solmethod

        # c. prep and choose initial values
        t0 = time.time()
        self.ast = solve.prep(par,sol,solmethod)
        if do_print: 
            print(f'Solution prepared in {elapsed(t0)}')

        # value function
        if v0 is None:
            h_guess = 1/3
            c = par.wtilde*h_guess*par.zzz + par.lumpsum_transfer + (par.rb+par.eta)*par.bbb
            sol.v[:] = modelfuncs.util(par,c,h_guess)/(par.rho+par.eta)
        else:
            sol.v[:] = v0

        # steady state distribution
        if g0 is None:
            index = par.Nb_neg*par.Na + 0
            sol.g[:] = 0
            sol.g[:,index] = par.z_dist/par.dab_tilde[par.Nb_neg,0]
        else:
            sol.g[:] = g0

        # d. solve HJB
        if do_print:
            print('Solving HJB:')
        
        self.timings['HJB'] = solve.solve_HJB(self,do_print,print_freq,solmethod)

        if do_print:
            print('')

        # e. solve KFE
        if do_print:
            print('Solving KFE:')

        self.timings['KFE'] = solve.solve_KFE(self,do_print,print_freq,solmethod)

        # f. calculate moments
        self.calculate_moments()

    def calculate_moments(self,do_MPC=False):
        """ calculate moments """
        
        par = self.par
        sol = self.sol
        moms = self.moms = {}

        # a. marginal and cumulative distributions 
        
        # i. joint asset distributions
        _g = sol.g.reshape(par.Nz,par.Na,par.Nb,order='F')
        moms['margdist'] = _g*par.dab_tilde.T
        moms['ab_margdist'] = np.sum(moms['margdist'],axis=0)

        moms['ab_margcum'] = np.zeros(moms['ab_margdist'].shape)
        for i_a in range(par.Na):
            for i_b in range(par.Nb):
                moms['ab_margcum'][i_a,i_b] = np.sum(moms['ab_margdist'][:i_a+1,:i_b+1])

        # ii. single asset distributions        
        moms['b_margdist'] = np.sum(moms['ab_margdist'],axis=0)
        moms['b_margcum'] = np.cumsum(moms['b_margdist'])
        moms['b_margcumfrac'] = np.cumsum(moms['b_margdist']*par.grid_b) 
        moms['b_margcumfrac'] /= moms['b_margcumfrac'][-1]
        
        moms['a_margdist'] = np.sum(moms['ab_margdist'],axis=1)
        moms['a_margcum'] = np.cumsum(moms['a_margdist']) 
        moms['a_margcumfrac'] = np.cumsum(moms['a_margdist']*par.grid_a) 
        moms['a_margcumfrac'] /= moms['a_margcumfrac'][-1]

        # iii. other distributions
        order_c = sol.c.ravel().argsort(axis=0)
        moms['grid_c'] = np.sort(sol.c.ravel())
        moms['c_margdist'] = moms['margdist'].ravel()[order_c]
        moms['c_margcum'] = np.cumsum(moms['c_margdist'])
        
        dcf = np.diff(moms['grid_c'])
        dcb = np.diff(moms['grid_c'])
        dcf = np.append(dcf,dcf[-1])
        dcb = np.append(dcb[0],dcb)
        dc_tilde = 0.5*(dcb + dcf)
        dc_tilde[0] = 0.5*dcf[0]
        dc_tilde[-1] = 0.5*dcb[-1]

        order_v = sol.v.ravel().argsort(axis=0)
        moms['grid_v'] = np.sort(sol.v.ravel())
        moms['v_margdist'] = moms['margdist'].ravel()[order_v]
        moms['v_margcum'] = np.cumsum(moms['v_margdist'])

        # b. household moments
        
        # sums
        moms['N_supply'] = np.sum(moms['margdist']*sol.h*par.zzz)
        moms['WageIncome'] = par.w*moms['N_supply']        
        a_supply_dist = moms['margdist']*par.grid_a.reshape(1,par.Na,1)
        moms['A_supply'] = np.sum(a_supply_dist)
        b_demand_dist = moms['margdist']*par.grid_b.reshape(1,1,par.Nb)
        moms['B_demand'] = np.sum(b_demand_dist)
        moms['c'] = np.sum(moms['margdist']*sol.c)
        moms['v'] = np.sum(moms['margdist']*sol.v)

        # fractions
        moms["frac_b0_a0"] = np.sum(moms['ab_margdist'][0,par.Nb_neg]) # poor HtM
        moms["frac_b0_apos"] = np.sum(moms['ab_margdist'][1:,par.Nb_neg]) # wealthy HtM
        moms["frac_bneg"] = np.sum(moms['b_margdist'][0:(par.Nb_neg)]) # borrowers

        # robust fractions
        for j in ['baseline',0.001,0.01,0.05]:

            if j == 'baseline':
                Ia_pos = par.aaa >= 0.012627193499002001
                Ib_neg = par.bbb <= -0.008944271909999135
                Ib_pos = par.bbb >= 0.0011380493998883773
            else:
                Ia_pos = par.aaa >= j*par.w
                Ib_neg = par.bbb <= -j*par.w
                Ib_pos = par.bbb >= j*par.w

            Ia_zero = ~Ia_pos
            Ib_zero = (~Ib_pos) & (~Ib_neg)

            moms[("frac_b0_a0",j)] = np.sum(moms['margdist'][Ib_zero & Ia_zero]) # poor HtM
            moms[("frac_b0_apos",j)] = np.sum(moms['margdist'][Ib_zero & Ia_pos]) # wealthy HtM
            moms[("frac_bneg",j)] = np.sum(moms['margdist'][Ib_neg]) # borrowers        

        # percentiles
        pvec = [0.001,0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99,0.999]
        for p in pvec:

            # illiquid
            if p <= moms['a_margcum'][0]:
                moms[('a',p)] = par.grid_a[0]
            else:
                try:
                    moms[('a',p)] = np.interp(p,moms['a_margcum'],par.grid_a)
                except:
                    moms[('a',p)] = np.nan

            # liquid
            if p <= moms['b_margcum'][0]:
                moms[('b',p)] = par.grid_b[0]
            else:
                try:
                    moms[('b',p)] = np.interp(p,moms['b_margcum'],par.grid_b)
                except:
                    moms[('a',p)] = np.nan

        # gini
        moms['a_gini'] = np.sum(moms['a_margcum']*(1-moms['a_margcum'])*par.da_tilde)/moms["A_supply"]
        moms['b_gini'] = np.sum(moms['b_margcum']*(1-moms['b_margcum'])*par.db_tilde)/moms["B_demand"]
        moms['c_gini'] = np.sum(moms['c_margcum']*(1-moms['c_margcum'])*dc_tilde) / moms['c']

        # top and bottom shares
        for p in [0.001,0.01,0.10]:
            moms[('a','top',p)] = 1-np.interp(1-p,moms['a_margcum'],moms['a_margcumfrac'])
            moms[('b','top',p)] = 1-np.interp(1-p,moms['b_margcum'],moms['b_margcumfrac'])

        for p in [0.50,0.25]:
            moms[('a','bottom',p)] = np.interp(p,moms['a_margcum'],moms['a_margcumfrac'])
            moms[('b','bottom',p)] = np.interp(p,moms['b_margcum'],moms['b_margcumfrac'])

        # c. aggregate
        moms['q'] = par.omega*par.Pi/par.ra
        moms['K_supply'] = moms['A_supply'] - moms['q']         

        moms['Y'] = moms['K_supply']**par.alpha*moms['N_supply']**(1-par.alpha)
        moms['KN'] = moms['K_supply']/moms['N_supply']
        moms['AY'] = moms['A_supply'] / (4*moms['Y'])
        moms['KY'] = moms['K_supply'] / (4*moms['Y'])
        moms['BY'] = moms['B_demand'] / (4*moms['Y'])
        moms['Pi'] = moms['Y']/par.vareps
        moms['Pi_discrepancy'] = (moms['Pi']-par.Pi)/par.Pi

        # d. MPC
        if do_MPC:

            moms['MPCs'] = solve.FeynmanKac_MPC(par,sol,moms)
            moms['MPC'] = np.sum(moms['MPCs']*moms['margdist'])

            order_MPC = moms['MPCs'].ravel().argsort(axis=0)
            moms['grid_MPC'] = np.sort(moms['MPCs'].ravel())
            moms['MPC_margdist'] = moms['margdist'].ravel()[order_MPC]
            moms['MPC_margcum'] = np.cumsum(moms['MPC_margdist'])

        # e. compatibility with firms
        moms['KN_firms'] = par.alpha/(1-par.alpha)*par.w/(par.ra+par.delta)
        moms['KN_discrepancy'] = (moms['KN_firms']-moms['KN'])/moms['KN']

        # e. timings
        moms['HJB'] = self.timings['HJB']
        moms['KFE'] = self.timings['KFE']

    def show_moments(self):

        moms = self.moms

        print('Equilibrium objects:')
        print(f" rb: {self.par.rb:.4f}")
        print(f" ra: {self.par.ra:.4f}")
        print(f" w: {self.par.w:.3f}")
        print(f" Pi: {self.par.Pi:.3f}")
        print(f" capital-labor discrepancy: {moms['KN_discrepancy']:.8f}")
        print(f" profit discrepancy: {moms['Pi_discrepancy']:.8f}")
        print('')
        print('Aggregates:')
        print(f" GDP: {moms['Y']:.3f}")
        print(f" capital-labor ratio: {moms['KN']:.3f}")
        print(f" capital-output ratio: {moms['KY']:.3f}")
        print(f" bond-output ratio: {moms['BY']:.3f}")
        print('')
        print(f'Fractions:')
        print(f" poor HtM: {moms['frac_b0_a0']:.3f} [{moms[('frac_b0_a0','baseline')]:.3f}, {moms[('frac_b0_a0',0.01)]:.3f}]")
        print(f" wealthy HtM: {moms['frac_b0_apos']:.3f} [{moms[('frac_b0_apos','baseline')]:.3f}, {moms[('frac_b0_apos',0.01)]:.3f}]")
        print(f" borrowers: {moms['frac_bneg']:.3f} [{moms[('frac_bneg','baseline')]:.3f}, {moms[('frac_bneg',0.01)]:.3f}]")
        print('')
        print(f'Iliquid wealth:')
        print(f" top   0.1: {moms[('a','top',0.001)]:.3f}")
        print(f" top     1: {moms[('a','top',0.01)]:.3f}")
        print(f" top    10: {moms[('a','top',0.10)]:.3f}")
        print(f" bottom 50: {moms[('a','bottom',0.50)]:.3f}")
        print(f" bottom 25: {moms[('a','bottom',0.25)]:.3f}")        
        print(f" gini: {moms['a_gini']:.3f}")
        print('')
        print(f'Liquid wealth:')
        print(f" top   0.1: {moms[('b','top',0.001)]:.3f}")
        print(f" top     1: {moms[('b','top',0.01)]:.3f}")
        print(f" top    10: {moms[('b','top',0.10)]:.3f}")
        print(f" bottom 50: {moms[('b','bottom',0.50)]:.3f}")
        print(f" bottom 25: {moms[('b','bottom',0.25)]:.3f}")        
        print(f" gini: {moms['b_gini']:.3f}")   
        print('')
        if 'MPC' in moms:
            print(f"MPC: {moms['MPC']:.3f}")
        print(f"Consumption: avarage = {moms['c']:.3f}, gini = {moms['c_gini']:.3f}")
        print(f"Value: avarage = {moms['v']:.3f}")

    def find_ra(self,KN0,Pi0,use_prev_sol=False,do_print=True,step_size=0.1,tol=1e-8):

        equilibrium.find_ra(self,KN0=KN0,Pi0=Pi0,use_prev_sol=use_prev_sol,do_print=do_print,step_size=step_size,tol=tol)

    def calibrate(self,use_prev_sol=False,do_print=True,tol=1e-3):

        estimate.run(self,use_prev_sol=use_prev_sol,tol=tol,do_print=True)