#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
April 28, 2020
"""

import numpy as np
import time
from forward_model import numerical_model
from scipy.interpolate import interp1d

mult = np.matmul
inv = np.linalg.inv
tr = np.transpose

z_data,T_data = np.load('data/input_data.npy')

def norm2(m,z_data=z_data,T_data=T_data):
    T_interp = interp1d(m.z,m.T)
    T_mod = T_interp(z_data)
    norm = np.sum((T_data-T_mod)**2.)
    return norm

def cost(modeled_Temp, T_param, t_param, measured_Temp=T_data, regularization=0.0):
    """
    """
    # Return the cost function given the measured temperature profile and our model output
    if regularization > 0:
        reg = np.nansum(np.gradient(np.gradient(T_param,t_param),t_param) ** 2.0)
    else:
        reg = 0.0
    unreg_cost = np.sqrt(np.nansum((modeled_Temp - measured_Temp) ** 2.0))
    return unreg_cost + regularization * reg, unreg_cost

def weakly_nonlinear(f,norm,zdata,Tdata,C,m_init,mstep,Niter=10,
                     solution_tolerance=1e-5,verbose=True):
    """

    """

    # Set up matrices for output
    m_out = np.zeros((Niter,len(m_init)))
    m_out[0,:] = m_init  # model at each iteration
    d_out = np.zeros((Niter,len(Tdata)))
    d_out[0,:]=f(m_init)     # modeled data vector at each iteration
    if verbose:
        print('Initial model')
        print('Data residual norm:',norm(Tdata-f(m_init)))

    # Initialize the Jacobian to be filled in
    A = np.zeros((len(Tdata),len(m_init)))

    # Start iterating
    m = m_init.copy()
    for i in range(1,Niter):
        print('Iteration #', i)

        # Calculate the data residual vector
        dd = Tdata - f(m)
        ddw = mult(C,dd)

        # Calculate the Jacobian matrix.
        f0 = f(m)
        for j in range(len(m)):
            mJ = m.copy()
            mJ[j] += mstep[j]
            A[:,j] = (f(mJ)-f0)/mstep[j];
        Aw = mult(C,A)

        # Linearize equation to solve for the model step, dm
        dm = mult(inv(mult(tr(Aw),Aw)),mult(tr(Aw),ddw))

        while norm(Tdata-f(m+dm)) > norm(dd) or (m[2]+dm[2])<0:
            dm /= 2.

        # Update the model
        m += dm

        # Print and store the iteration metrics
        if verbose:
            print('Data residual norm:',norm(Tdata-f(m)))
            print('Norm of model step, dm:',norm(dm))
        m_out[i,:] = m
        d_out[i,:]=f(m)

        # If convergence criteria is met, break out of loop
        if i>1 and norm(dm)<solution_tolerance:
            return m_out[:i+1,:],d_out[:i+1,:]

def monte_carlo(params,Tdata,zdata,
                    tinit=-50000.,dt_var=5000.,
                    n_iterates=500,stepsize=0.2,regularization=0.):
    """
    Perform a suite of runs to get our distribution. We start with an initial guess, and move around.
    """

    # this is just to check how long it took
    now = time.time()
    i = 0

    # fencepost; this is so we don't have errors comparing to previous state
    t_var = np.arange(tinit,0.+dt_var,dt_var)
    t_var *= np.linspace(1,0,len(t_var))**3.
    Tmodel = numerical_model(params,t_var,tinit)

    # Preallocate locations for output--we want all our accepted models saved
    outputs = np.zeros((len(t_var), n_iterates))
    outputs[:, i] = params
    # misfits will have both a regularized and an unregularized row
    misfits = np.zeros((n_iterates, 2))
    misfits[i] = cost(Tmodel, params, t_var,regularization=regularization)

    # This is the real bulk of the work--keep trying to make steps downward
    while i < n_iterates - 1:
        # start by perturbing--first copy our vector to avoid modifying original in case we dont step
        params_pert = params.copy()
        # randint gives us a random component of the vector to perturb
        rand_ind = np.random.randint(0, len(t_var) - 1)
        # We use a normal distribution with variance "stepsize" for the perturbation
        #stepsize = max(10*(1.-float(i)/n_iterates),1)
        params_pert[rand_ind] = params[rand_ind] + np.random.normal(0., stepsize)

        # enforce that the temperature is within a reasonable range
        if params_pert[rand_ind] > 0. or params_pert[rand_ind] < -50.:
            # if not, we don't save the perturbed state and we just restart this iteration of the while loop
            continue

        #evaluate model
        Tmodel = numerical_model(params_pert,t_var,tinit)

        # see if model is any good---store cost function in vector, will be overwritten if not chosen
        misfits[i + 1, :] = cost(Tmodel, params_pert, t_var, regularization=regularization)

        # Decide whether to accept our new guess
        if i < 1 or np.random.rand() < min(1, np.exp(-(misfits[i + 1, 0] - misfits[i, 0]))):
            # We have accepted the model! We need to store it, and save this state as the new model to step from
            outputs[:, i + 1] = params_pert
            params = params_pert.copy()
            if i%100 == 0:
                print('Successful Run:',i,np.random.normal(0., stepsize),stepsize,misfits[i + 1, :])
            # increment only upon acceptance of the guess
            i += 1

    print('Run took {:5.1f}s'.format(time.time() - now))
    return outputs, misfits
