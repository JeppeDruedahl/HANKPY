import time
import numpy as np
from scipy import optimize
import modelfuncs

from consav.misc import elapsed

def calc_obj(model):

    # a. targets
    sum_sq_diff = 0
    for target,value in model.targets.items():
        sum_sq_diff += (model.moms[target]-value)**2

    # b. equilibrium conditions
    discrepancy = 0
    discrepancy += np.abs(model.moms['KN_discrepancy'])
    discrepancy += np.abs(model.moms['Pi_discrepancy'])

    return sum_sq_diff + discrepancy

def update(x,model):
    
    par = model.par
    names = model.names

    for i,name in enumerate(names):
        setattr(par,name,x[i])

    if hasattr(model,'extra_update'):
        if model.extra_update is callable:
            model.extra_update(par)

def bound(x,model):

    lower = model.lower
    upper = model.upper

    penalty = 0
    x_clipped = x.copy()
    for i in range(x.size):
        x_clipped[i] = np.clip(x_clipped[i],lower[i],upper[i])
        penalty += 10_000*(x[i]-x_clipped[i])**2

    return x_clipped,penalty

def obj(x,model,use_prev_sol):
    
    global it, funcevals

    # i. bound parameters
    x,penalty = bound(x,model)

    # ii. update parameters
    update(x,model)

    # iii. solve
    if use_prev_sol:
        if it == 0:
            model.solve(do_print=False)
        else: # use result from last as starting value
            model.solve(v0=model.sol.v.copy(),g0=model.sol.g.copy(),do_print=False)        
    else:
        model.solve(do_print=False)
    
    funcevals += 1

    # iv. calculate objective    
    return calc_obj(model) + penalty

def progress(x,model):
        
    global it, funcevals, t0

    par = model.par
    names = model.names

    print(f'iteration: {it}')

    str_par = 'parameters: '
    for i,name in enumerate(names):
        if i > 0: str_par += ', '
        str_par += f'{name} = {getattr(par,name):.3f}'
    print(str_par)

    str_targets = ' targets: '
    for i,(target,value) in enumerate(model.targets.items()):
        if i > 0: str_targets += ', '
        str_targets += f'{target} = {model.moms[target]:.3f} [{value:.3f}]'
    print(str_targets)
    print(f' equilibrium: KN = {model.moms["KN"]:.3f} [{model.moms["KN_firms"] :.3f}], Pi = {model.par.Pi:.3f} [{model.moms["Pi"] :.3f}]')
    print(f' objective = {calc_obj(model):.8f}')
    print(f' time: {elapsed(t0)}  [functional evaluations: {funcevals}]')
    t0 = time.time()
    funcevals = 0
    print('')

    it += 1

def run(model,tol,use_prev_sol,do_print):
    
    global it, funcevals, t0
    it = 0
    funcevals = 0
    t0_outer = time.time()
    t0 = time.time()

    names = model.names
    specs = model.specs

    # a. unpack
    x = np.array([specs[name]['initial'] for name in names])
    model.lower = np.array([specs[name]['lower'] for name in names])
    model.upper = np.array([specs[name]['upper'] for name in names])

    # b. initial
    obj(x,model,use_prev_sol)
    progress(x,model)

    # c. run optimizer
    res = optimize.minimize(obj,x,
        args=(model,use_prev_sol),
        callback=lambda x: progress(x,model),
        method='Nelder-Mead',
        options={'disp':True,'maxiter':500,'xatol': tol,'fatol': tol})

    # d. update with result
    update(res.x,model)
    model.calculate_moments(do_MPC=True)
    
    if do_print:

        print('')
        secs = time.time()-t0_outer
        mins = secs//60
        secs -= 60*mins
        print(f'calibration done in {mins:.0f} min. and {secs:.1f} secs')