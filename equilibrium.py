import time
import numpy as np

from consav.misc import elapsed 

def find_ra(model,Pi0,KN0,use_prev_sol,do_print,step_size,tol):

    # unpack    
    par = model.par

    # initial
    m = (par.vareps-1)/par.vareps     
    Pi = Pi0
    KN = KN0
    it = 1
    
    # loop
    t0_outer = time.time()
    while True:
        
        t0 = time.time()

        if do_print: print(f'{it:3d}: KN = {KN:.8f}, Pi = {Pi:.8f}')

        # a. factor prices   
        par.ra = par.alpha*m/(KN**(1-par.alpha)) - par.delta
        par.w = (1-par.alpha)*m*KN**par.alpha
        par.Pi = Pi

        if do_print: 
            print(f'    implied ra = {par.ra:.4f}')
            print(f'    implied w = {par.w:.4f}')

        # b. solve model
        model.create_grids()
        if use_prev_sol and it > 1:
            model.solve(v0=model.sol.v.copy(),g0=model.sol.g.copy(),do_print=False)
        else:
            model.solve(do_print=False)
        model.calculate_moments()
        moms = model.moms

        # c. implied KN and Pi0
        if do_print: 
            print(f'    implied KN = {moms["KN"]:.4f} [diff.: {moms["KN_discrepancy"]:.8f}]')
            print(f'    implied Pi = {moms["Pi"]:.4f} [diff.: {moms["Pi_discrepancy"]:.8f}]')
            print(f'    time: {elapsed(t0)}')
        
        # d. check for convergence
        if abs(moms['KN_discrepancy']) < tol and abs(moms['Pi_discrepancy']) < tol:
            break

        # e. updates
        KN = step_size*moms["KN"] + (1-step_size)*KN
        Pi = step_size*moms["Pi"] + (1-step_size)*Pi

        it += 1   

    model.calculate_moments(do_MPC=True)
    if do_print:

        print('')
        print(f'equilibrium found in {elapsed(t0_outer)}')
        print(f' ra = {par.ra:.8f}')
        print(f' w = {par.w:.8f}')
        print(f' Pi = {par.Pi:.8f}')