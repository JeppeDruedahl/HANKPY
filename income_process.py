import numpy as np
from scipy.stats import norm
from scipy.sparse import diags
from numba import njit, prange

###################
# 1. construction #
###################

def find_lin_prob(xi,x):
    """ takes in xi, and finds two points in either side of xi and
    return the indices of them, y, and associated probabilites p """
    
    # a. pre-allocate output containers
    p = np.zeros(2)
    y = np.zeros(2)
    n = x.shape
    
    # b. calculate indices and probabilities
    if n == 1:

        p[0] = 1
        p[1] = 0
        y[0] = 0 
        y[1] = 0
        
    LocL = np.argmax(x[xi>x])
    
    if xi <= x[0]:
        y[0] = 1
        y[1] = 2
        p[0] = 1
        p[1] = 0
    elif LocL >= n:
        LocL = n-2
        y[0] = n-2
        y[1] = n-1
        p[0] = 0
        p[1] = 1
    elif x[LocL+1] == x[LocL]:
        y[0] = LocL
        y[1] = LocL+1
        p[0] = 0.5
        p[1] = 0.5
    else:
        y[0] = LocL
        y[1] = LocL+1
        p[1] = (xi - x[LocL])/np.real(x[LocL+1]-x[LocL])
        p[0] = 1-p[1]

    return y,p

def symmetric_power_grid(n,k,width,center):
    """ gives a grid spaced between center-width and center+width based on the interval [-1,1] 
    with a function x^(1/k) on either side k = 1 is linear, k = 0 is L-shaped """
    
    # a. pre-allocate solution vectors
    x = np.linspace(-1,1,n)
    z = np.zeros(n)
    
    # b. generate grid
    if n < 2: 
        print('n must be at least 2 to make grids')
        return

    if n == 2:    
        y = [center-width,center+width]
        return
    
    for i in range(n):
        if x[i] > 0:
            z[i] = x[i]**(1.0/k)
        elif x[i] == 0:
            z[i] = 0.0
        elif x[i] < 0:
            z[i] = -((-x[i])**(1.0/k))
    
    y = center + width*z
    
    return y

def income_grid(Nz,k,z_width):

    zgrid = symmetric_power_grid(Nz,k,z_width/2,0)
    dzgrid = np.diff(zgrid)

    return zgrid, dzgrid

def jump_matrix(Nz,zgrid,dzgrid,lambda_z,rho,sigma):
    
    # a. pre-allocate solution containers
    jump = np.zeros((Nz,Nz))
    eye = np.identity(Nz)
    
    # b. insert into jump matrix
    for i in range(Nz):
        for j in range(Nz):
            if j==0:
                jump[i,j] = norm.cdf(zgrid[j]+0.5*dzgrid[j],rho*zgrid[i],sigma)
            elif j > 0 and j < Nz-1:
                jump[i,j] = norm.cdf(zgrid[j]+0.5*dzgrid[j],rho*zgrid[i],sigma) - norm.cdf(zgrid[j]-0.5*dzgrid[j-1],rho*zgrid[i],sigma)
            elif j == Nz-1:
                jump[i,j] = 1.0 - norm.cdf(zgrid[j]-0.5*dzgrid[j-1],rho*zgrid[i],sigma)
        jump[i,:] = jump[i,:]/np.sum(jump[i,:])

    jump = lambda_z*(jump-eye)
    
    return jump

def drift_matrix(Nz,zgrid,beta,delta,DriftPointsVersion=2):
    
    # a. pre-allocate container for solution
    drift = np.zeros((Nz,Nz))
    
    # b. insert into drift matrix
    for i in range(Nz):
        if zgrid[i] != 0:
             
            ii,_ = find_lin_prob((1.0-beta*delta)*zgrid[i],zgrid)
                         
            drift[i,i] = -1
        
            if zgrid[i] < 0:
                drift[i,i] = drift[i,i] + (zgrid[int(ii[1])] - (1.0-beta)*zgrid[i])/(zgrid[int(ii[1])]-zgrid[i])
                drift[i,int(ii[1])] = (-zgrid[i] + (1.0-beta)*zgrid[i])/(zgrid[int(ii[1])]-zgrid[i])
        
            elif zgrid[i] > 0.0:
                drift[i,int(ii[0])] = (zgrid[i] - (1.0-beta)*zgrid[i])/(zgrid[i]-zgrid[int(ii[0])])
                drift[i,i] = drift[i,i] + (-zgrid[int(ii[0])] + (1.0-beta)*zgrid[i])/(zgrid[i]-zgrid[int(ii[0])])
    
    return drift

def ergodic_dist(Nz,z_markov,delta=1):
    """ find the ergodic distribution of a single income process component """
    
    # a. allocate containers
    zdist = np.zeros(Nz)
    zdist[int((Nz+1)/2)-1] = 1
    eye  = np.eye(Nz)
    
    # b. prepare solution matrices
    mat = eye - delta*z_markov
    matinv = np.linalg.inv(mat)
    
    # c. find ergodic distribution by iteration
    it = 1
    lerr = 1
    
    while lerr > 1e-14 and it < 1000:

        zdist_update = np.matmul(zdist,matinv)
        zdist_update[abs(zdist_update)<1.0e-20_8] = 0.0_8
        zdist_update = zdist_update/np.sum(zdist_update)
        lerr = np.amax(abs(zdist_update-zdist))
        zdist = zdist_update
        it = it + 1
    
    return zdist

def combined_process(Nz1,Nz2,z1grid,z2grid,z1dist,z2dist,z1_markov,z2_markov,dt,beta=100):

    # a. allocate containers for solution
    eye1 = np.identity(Nz1)
    eye2 = np.identity(Nz2)
    eye = np.identity(Nz1*Nz2)

    lzgrid_combined = np.zeros(Nz1*Nz2)
    lztrans_dt_combined = np.zeros((Nz1*Nz2,Nz1*Nz2))
    lzmarkov_combined = np.zeros((Nz1*Nz2,Nz1*Nz2))
    lzdist_combined = np.zeros(Nz1*Nz2)
    
    zgrid_combined = np.zeros(Nz1*Nz2)
    ztrans_dt_combined = np.zeros((Nz1*Nz2,Nz1*Nz2))
    zmarkov_combined = np.zeros((Nz1*Nz2,Nz1*Nz2))
    zdist_combined = np.zeros(Nz1*Nz2)
    
    # b. combined process function
    z1trans = dt*z1_markov + eye1
    z2trans = dt*z2_markov + eye2    
    
    # c. transition matrix corresponding to dt (e.g. dt = 0.25 yields a quarterly matrix)
    
    # i. pre-allocate solution
    z1trans_dt = z1trans.copy()
    z2trans_dt = z2trans.copy()
    
    # ii. normalize trans_dt according to dt
    for i in range(0,int(np.floor(1.0/dt))-1):
        z1trans_dt = np.matmul(z1trans_dt,z1trans)
        z2trans_dt = np.matmul(z2trans_dt,z2trans)
    
    # d. get combined grid and transition matrix
    i = -1
    for i1 in range(Nz1):
        for i2 in range(Nz2):
            i = i + 1
            lzgrid_combined[i] = z1grid[i1] + z2grid[i2]
            lzdist_combined[i] = z1dist[i1]*z2dist[i2]
            
            j = -1
            for j1 in range(Nz1):
                for j2 in range(Nz2):
                    j = j + 1
                    lztrans_dt_combined[i,j] = z1trans_dt[i1,j1]*z2trans_dt[i2,j2]
                    if i1==j1 and i2==j2: lzmarkov_combined[i,j] = z1_markov[i1,j1] + z2_markov[i2,j2]
                    if i1==j1 and i2!=j2: lzmarkov_combined[i,j] = z2_markov[i2,j2]
                    if i1!=j1 and i2==j2: lzmarkov_combined[i,j] = z1_markov[i1,j1]
                    if i1!=j1 and i2!=j2: lzmarkov_combined[i,j] = 0
    
    # e. sort into ascending order
    
    # i. generate sorted grid indices
    iorder = np.argsort(lzgrid_combined)
    
    # ii. sort combined grid, ytrans_dt_combined and combined markov matrix
    for i in range(Nz1*Nz2):
        zgrid_combined[i] = lzgrid_combined[iorder[i]] 
        zdist_combined[i] = lzdist_combined[iorder[i]]
        for j in range(Nz1*Nz2):
            ztrans_dt_combined[i,j] = lztrans_dt_combined[iorder[i],iorder[j]]
            zmarkov_combined[i,j] = lzmarkov_combined[iorder[i],iorder[j]]
    
    # f. fix up rounding in markov matrix
    zmarkov_combined = zmarkov_combined - np.diag(np.sum(zmarkov_combined,axis=1))
    
    # g. find ergodic distribution by iteration
    
    # i. prepare solution matrices
    z1dist_ = z1dist.copy()
    z1dist_[0] = 0.1
    ii = ((Nz1+1)/2-1)*Nz2 + (Nz2+1)/2
    zdist_combined[int(ii)] = 0.9
    mat = eye - beta*zmarkov_combined
    matinv = np.linalg.inv(mat)
    
    # ii. compute ergodic distribution
    it = 1
    err = 1
    
    while err>1e-15 and it<1000:
        zdist_combined_update = np.matmul(zdist_combined,matinv)
        zdist_combined_update[abs(zdist_combined_update)<1.0e-20_8] = 0.0_8
        zdist_combined_update = zdist_combined_update/np.sum(zdist_combined_update)
        err = np.amax(abs(zdist_combined_update-zdist_combined))
        zdist_combined = zdist_combined_update
        it = it + 1

    # iii. fix up rounding in ergodic distribution
    zdist_combined = zdist_combined/np.sum(zdist_combined)

    return zgrid_combined, ztrans_dt_combined, zmarkov_combined, zdist_combined

def construct_jump_drift(par):

    # a. income process component grids
    par.grid_z1, dz1grid = income_grid(par.Nz1,par.kz_1,par.z1_width)
    par.grid_z2, dz2grid = income_grid(par.Nz2,par.kz_2,par.z2_width)

    # b. jump matrices
    z1_jump = jump_matrix(par.Nz1,par.grid_z1,dz1grid,par.lambda1,par.rho1,par.sigma1)
    z2_jump = jump_matrix(par.Nz2,par.grid_z2,dz2grid,par.lambda2,par.rho2,par.sigma2)
    
    # c. drift (decay) matrices
    z1_drift = drift_matrix(par.Nz1,par.grid_z1,par.beta1,par.DeltaIncome)
    z2_drift = drift_matrix(par.Nz2,par.grid_z2,par.beta2,par.DeltaIncome)

    # d. add jumps and drift
    par.z1_markov = z1_jump + z1_drift
    par.z2_markov = z2_jump + z2_drift

    # e. ergodic distributions
    par.z1dist = ergodic_dist(par.Nz1,par.z1_markov)
    par.z2dist = ergodic_dist(par.Nz2,par.z2_markov)

    # f. combined process
    zgrid_combined, ztrans_dt_combined, zmarkov_combined, zdist_combined = combined_process(
        par.Nz1,par.Nz2,
        par.grid_z1,par.grid_z2,
        par.z1dist,par.z2dist,
        par.z1_markov,par.z2_markov,
        par.dt)

    return zgrid_combined, ztrans_dt_combined, zmarkov_combined, zdist_combined

def split_markov_matrix(par,zmarkov_combined):

    # a. get center diagonal
    switch_diag = np.diag(zmarkov_combined,0)

    # b. get off-diagonals
    switch_off_ = zmarkov_combined.copy()
    np.fill_diagonal(switch_off_, 0)

    # c. create off diagonal matrix
    
    # i. preallocate diagonal and offset lists
    diagonals = []
    offsets = []
    
    # ii. get upper diagonals
    it = par.Nab
    for iz in range(1,par.Nz):
        diagonals.append(np.repeat(np.diag(zmarkov_combined,iz),par.Nab))
        offsets.append(it)
        it = it + par.Nab

    # iii. get lower diagonals 
    it = par.Nab
    for iz in range(1,par.Nz):
        diagonals.append(np.repeat(np.diag(zmarkov_combined,-iz),par.Nab))
        offsets.append(-it)
        it = it + par.Nab
    
    # iv. generate sparse matrix for off diagonals
    switch_off = diags(diagonals = diagonals,
                       offsets   = offsets,
                       shape     = (par.Nzab,par.Nzab),format='csc')
    
    return switch_diag, switch_off, switch_off_

def stretch_markov_matrix(Nab,Nzab,Nz,zmarkov_combined):

    # a. preallocate diagonal lists
    diagonals = []
    offsets = []
    
    # b. get diagonals and offsets
    # i. center diagonal
    diagonals.append(np.repeat(np.diag(zmarkov_combined,0),Nab))
    offsets.append(0)
    
    # ii. upper diagonals
    it = Nab
    for iz in range(1,Nz): # parallel
        diagonals.append(np.repeat(np.diag(zmarkov_combined,iz),Nab))
        offsets.append(it)
        it = it + Nab
    
    # iii. lower diagonals
    it = Nab    
    for iz in range(1,Nz): # parallel
        diagonals.append(np.repeat(np.diag(zmarkov_combined,-iz),Nab))
        offsets.append(-it)
        it = it + Nab
    
    # c. generate sparse switch matrix
    switch = diags(diagonals = diagonals,
                   offsets   = offsets,
                   shape     = (Nzab,Nzab),format='csc')
    
    return switch

#################
# 2. simulation #
#################

@njit
def choice(p,r): 

    i = 0
    while r > p[i]:
        i = i + 1
        
    return i

@njit(parallel=True,fastmath=True)
def sim_est(par,nsim,seed=2019):
    
    np.random.seed(seed) # set seed    
    
    # a. define simulation parameters and pre-allocate solution containers
    Tburn = np.floor(100.0/par.dt)+1		
    Tsim = int(Tburn + np.floor(24.0/par.dt)+1)	# 24 quarters
    Tann = int((Tsim-Tburn)*par.dt/4)
    
    z1rand = np.random.normal(0,1,size=(nsim,Tsim))
    z2rand = np.random.normal(0,1,size=(nsim,Tsim))
    z1jumpI = np.random.poisson(par.lambda1*par.dt,size=(nsim,Tsim))
    z2jumpI = np.random.poisson(par.lambda2*par.dt,size=(nsim,Tsim))
    
    z1sim = np.zeros((nsim,Tsim))
    z2sim = np.zeros((nsim,Tsim))
    zsim = np.zeros((nsim,Tsim))

    zannsim = np.zeros((nsim,Tann))
    zlevsim = np.zeros((nsim,Tsim))
    zannlevsim = np.zeros((nsim,Tann))
    
    # b. get variance of each process for initial distribution
    if par.rho1 != 1.0:
        if par.beta1 == 0.0: lssvar1 = (par.sigma1**2) / (1.0 - par.rho1**2)
        if par.beta1 != 0.0: lssvar1 = par.lambda1*(par.sigma1**2) / (2.0*par.beta1 + par.lambda1*(1.0 - par.rho1**2))
    elif par.rho1 == 1.0:	
        lssvar1 = (par.sigma1**2) / (1.0 - 0.99**2)

    if par.beta2 == 0.0: lssvar2 = (par.sigma2**2) / (1.0 - par.rho2**2)
    if par.beta2 != 0.0: lssvar2 = par.lambda2*(par.sigma2**2) / (2.0*par.beta2 + par.lambda2*(1.0 - par.rho2**2))

    # c. simulate n income paths in dt increments 
    for i_n in prange(nsim):	
        
        # i. draw initial distribution from normal distribution with same mean and variance
        z1sim[i_n,0] = np.sqrt(lssvar1)*z1rand[i_n,0]
        z2sim[i_n,0] = np.sqrt(lssvar2)*z2rand[i_n,0]
        
        zsim[i_n,0] = z1sim[i_n,0] + z2sim[i_n,0]
        
	    # ii. loop over time
        for i_t in range(Tsim-1):
        
            if z1jumpI[i_n,i_t] == 1: 
                z1sim[i_n,i_t+1] = par.rho1*z1sim[i_n,i_t] + par.sigma1*z1rand[i_n,i_t+1]
            else:
                 z1sim[i_n,i_t+1] = (1 - par.dt*par.beta1)*z1sim[i_n,i_t]
        
            if z2jumpI[i_n,i_t] == 1:
                z2sim[i_n,i_t+1] = par.rho2*z2sim[i_n,i_t] + par.sigma2*z2rand[i_n,i_t+1]
            else:
                z2sim[i_n,i_t+1] = (1 - par.dt*par.beta2)*z2sim[i_n,i_t]
        	    
            zsim[i_n,i_t+1] = z1sim[i_n,i_t+1] + z2sim[i_n,i_t+1]
	
        zlevsim[i_n,:] = np.exp(zsim[i_n,:])

		# iii. aggregate to annual income
        for i_t in range(Tann):

            step = np.floor(4.0/par.dt)
            it1 = int(Tburn + step*i_t)
            itN = int(it1 + step)

            zannlevsim[i_n,i_t] = np.sum(zlevsim[i_n,it1:itN+1])
	
        zannsim[i_n,:] = np.log(zannlevsim[i_n,:])
    
    return zannsim, zannlevsim

@njit(parallel=True,fastmath=True)
def sim_disc(par,nsim,seed=2019):
    
    np.random.seed(seed) # set seed
    
    # a. define simulation parameters and pre-allocate solution containers
    Tsim = int(np.floor(24.0/par.dt))+1  # 24 quarters
    Tann = int(Tsim*par.dt/4)

    zsim = np.zeros((nsim,Tsim))
    zlevsim = np.zeros((nsim,Tsim))
    
    zannsim = np.zeros((nsim,Tann))
    zannlevsim = np.zeros((nsim,Tann))
    z1simI = np.zeros((nsim,Tsim))
    z2simI = np.zeros((nsim,Tsim))
    
    eye1 = np.identity(np.int64(par.Nz1))
    eye2 = np.identity(np.int64(par.Nz2))

    z1trans = par.dt*par.z1_markov + eye1
    z2trans = par.dt*par.z2_markov + eye2  
    z1distcum = np.cumsum(par.z1dist)
    z2distcum = np.cumsum(par.z2dist)

    z1transcum = np.zeros(z1trans.shape)
    z2transcum = np.zeros(z2trans.shape)

    for i in range(z1trans.shape[0]):
        z1transcum[i,:] = np.cumsum(z1trans[i,:])

    for i in range(z2trans.shape[0]):
        z2transcum[i,:] = np.cumsum(z2trans[i,:])

    # b. generate random numbers
    z1rand = np.random.uniform(0,1,size=(nsim,Tsim))
    z2rand = np.random.uniform(0,1,size=(nsim,Tsim))
    
    # c. generate quarterly transition matrix
    z1trans_qu = z1trans
    z2trans_qu = z2trans

    for _ in range(int(np.floor(1.0/par.dt))-1):
        z1trans_qu = z1trans_qu@z1trans
        z2trans_qu = z2trans_qu@z2trans
        
    # d. simulate n income paths in dt increments   
    for i_n in prange(nsim):
        
        # i. initialize distribution from ergodic distribution
        z1simI[i_n,0] = choice(z1distcum,z1rand[i_n,0])
        z2simI[i_n,0] = choice(z2distcum,z2rand[i_n,0])
        zsim[i_n,0] = par.grid_z1[int(z1simI[i_n,0])] + par.grid_z2[int(z2simI[i_n,0])]
	    
        # ii. loop over time
        for i_t in range(1,Tsim):
            z1simI[i_n,i_t] = choice(z1transcum[int(z1simI[i_n,i_t-1]),:],z1rand[i_n,i_t])
            z2simI[i_n,i_t] = choice(z2transcum[int(z2simI[i_n,i_t-1]),:],z2rand[i_n,i_t]) 
            zsim[i_n,i_t] = par.grid_z1[int(z1simI[i_n,i_t])] + par.grid_z2[int(z2simI[i_n,i_t])]
            
        # iii. aggregate to annual income
        zlevsim[i_n,:] = np.exp(zsim[i_n,:])
        
        for i_t in range(Tann):

            step = np.floor(4.0/par.dt)
            it1 = int(step*i_t)
            itN = int(it1 + step)

            zannlevsim[i_n,i_t] = np.sum(zlevsim[i_n,it1:itN+1])
        
        zannsim[i_n,:] = np.log(zannlevsim[i_n,:])
        
    return zannsim, zannlevsim

def compute_moments(nsim,zannsim,zannlevsim):
    
    # a. pre-allocate empty dictionary for output
    moms = {}
    
    # b. central moms: logs
    moms["muz"] = np.mean(zannsim)
    moms["mu2z"] = np.mean((zannsim-moms["muz"])**2)
    moms["mu3z"] = np.mean((zannsim-moms["muz"])**3)
    moms["mu4z"] = np.mean((zannsim-moms["muz"])**4)

    # c. standardised moms: logs
    if moms["mu2z"] > 0.0:
        moms["gam3z"] = moms["mu3z"]/(moms["mu2z"]**1.5)
        moms["gam4z"] = moms["mu4z"]/(moms["mu2z"]**2)
    else:
        moms["gam3z"] = 0.0
        moms["gam4z"] = 0.0


    # d. central moms: logs
    moms["muzlev"] = np.mean(zannlevsim)
    moms["mu2zlev"] = np.mean((zannlevsim-moms["muzlev"])**2)
    moms["mu3zlev"] = np.mean((zannlevsim-moms["muzlev"])**3)
    moms["mu4zlev"] = np.mean((zannlevsim-moms["muzlev"])**4)
    
    # e. standardised moms: logs
    if moms["mu2zlev"] > 0.0:
        moms["gam3zlev"] = moms["mu3zlev"]/(moms["mu2zlev"]**1.5)
        moms["gam4zlev"] = moms["mu4zlev"]/(moms["mu2zlev"]**2)
    else:
        moms["gam3zlev"] = 0.0
        moms["gam4zlev"] = 0.0

    # f. central moms: 1 year log changes
    moms["mudz1"] = np.mean(zannsim[:,1:]-zannsim[:,:-1])
    moms["mu2dz1"] = np.mean((zannsim[:,1:]-zannsim[:,:-1]-moms["mudz1"])**2)
    moms["mu3dz1"] = np.mean((zannsim[:,1:]-zannsim[:,:-1]-moms["mudz1"])**3)
    moms["mu4dz1"] = np.mean((zannsim[:,1:]-zannsim[:,:-1]-moms["mudz1"])**4)

    # g. standardised moms: 1 year log changes
    if moms["mu2dz1"] > 0.0:
        moms["gam3dz1"] = moms["mu3dz1"]/(moms["mu2dz1"]**1.5)
        moms["gam4dz1"] = moms["mu4dz1"]/(moms["mu2dz1"]**2)
    else:
        moms["gam3dz1"] = 0.0
        moms["gam4dz1"] = 0.0

    # h. central moms: 5 year log changes
    moms["mudz5"] = np.mean(zannsim[:,5:]-zannsim[:,:-5])
    moms["mu2dz5"] = np.mean((zannsim[:,5:]-zannsim[:,:-5]-moms["mudz5"])**2)
    moms["mu3dz5"] = np.mean((zannsim[:,5:]-zannsim[:,:-5]-moms["mudz5"])**3)
    moms["mu4dz5"] = np.mean((zannsim[:,5:]-zannsim[:,:-5]-moms["mudz5"])**4)

    # i. standardised moms: 5 year log changes
    if moms["mu2dz5"] > 0.0:
        moms["gam3dz5"] = moms["mu3dz5"]/(moms["mu2dz5"]**1.5)
        moms["gam4dz5"] = moms["mu4dz5"]/(moms["mu2dz5"]**2)
    else:
        moms["gam3dz5"] = 0.0
        moms["gam4dz5"] = 0.0

    # j. fraction 1 year log changes in ranges
    moms["fracdz1less5"] = np.mean(np.abs(zannsim[:,1:]-zannsim[:,:-1]) < 0.05)
    moms["fracdz1less10"] = np.mean(np.abs(zannsim[:,1:]-zannsim[:,:-1]) < 0.1)
    moms["fracdz1less20"] = np.mean(np.abs(zannsim[:,1:]-zannsim[:,:-1]) < 0.2)
    moms["fracdz1less50"] = np.mean(np.abs(zannsim[:,1:]-zannsim[:,:-1]) < 0.5)
    
    return moms


