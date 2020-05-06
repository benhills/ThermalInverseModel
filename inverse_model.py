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

z_data,T_data = np.load('data/input_data.npy')


def norm2(m,z_data=z_data,T_data=T_data):
    T_interp = interp1d(m.z,m.T)
    T_mod = T_interp(z_data)
    norm = np.sum((T_data-T_mod)**2.)
    return norm


def cost(modeled_Temp, T_param, t_param, measured_Temp=T_data, regularization=0.0):
    # Return the cost function given the measured temperature profile and our model output
    if regularization > 0:
        reg = np.nansum(np.gradient(np.gradient(T_param,t_param),t_param) ** 2.0)
    else:
        reg = 0.0
    unreg_cost = np.sqrt(np.nansum((modeled_Temp - measured_Temp) ** 2.0))
    #plt.plot((modeled_Temp - measured_Temp) ** 2.0)
    return unreg_cost + regularization * reg, unreg_cost


def monte_carlo_run(params,Tdata,zdata,
                    tinit=-50000.,dt_var=5000.,
                    n_iterates=500,stepsize=0.2,regularization=0.):
    ###Perform a suite of runs to get our distribution. We start with an initial guess, and move around.

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
