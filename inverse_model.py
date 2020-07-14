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
from constants import constants
const = constants()

z_data,T_data,C_data = np.transpose(np.load('./data/icetemp_data.npy'))

# --------------------------------------------------------------------------------------------------------

def norm2(f,m,T_data=T_data):
    norm = np.sum((f(m)-T_data)**2.)
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

def reg(m,t_m,nu1=2.,nu2=1.5):
    """
    """

    # Smoothness of deformational terms, can't change too fast
    reg1 = nu1*np.nansum(np.gradient(np.gradient(m[2:2+len(t_m)],t_m/1e4),t_m/1e4)**2.)

    # Size of sliding terms, keep them close to zero
    reg2 = nu2*np.nansum(np.abs(m[-len(t_m):]))

    return reg1+reg2

# --------------------------------------------------------------------------------------------------------

mult = np.matmul
inv = np.linalg.inv
tr = np.transpose

def weakly_nonlinear(f,norm,zdata,Tdata,C,m_init,mstep,nu,L,Niter=10,
                     solution_tolerance=1e-5,verbose=True):
    """

    """

    # Set up matrices for output
    m_out = np.zeros((Niter+1,len(m_init)))
    m_out[0,:] = m_init  # model at each iteration
    d_out = np.zeros((Niter+1,len(Tdata)))
    d_out[0,:]=f(m_init)     # modeled data vector at each iteration
    if verbose:
        print('Initial model')
        print('Data residual norm:',norm(Tdata-d_out[0,:]))

    # Initialize the Jacobian to be filled in
    A = np.zeros((len(Tdata),len(m_init)))

    # Start iterating
    m = m_init.copy()
    for i in range(1,Niter+1):
        print('Iteration #', i)

        # Calculate the data residual vector
        f0 = f(m)
        dd = Tdata - f0
        ddw = mult(C,dd)

        # Calculate the Jacobian matrix.
        for j in range(len(m)):
            mJ = m.copy()
            mJ[j] += mstep[j]
            A[:,j] = (f(mJ)-f0)/mstep[j];
        Aw = mult(C,A)

        # Linearize equation to solve for the model step, dm
        try:
            dm = mult(inv(mult(tr(Aw),Aw)+nu**2.*mult(tr(L),L)),(mult(tr(Aw),ddw)-(nu**2.*mult(mult(tr(L),L),m))))
        except:
            print('Could not solve inversion, exiting.')
            print('Aw =', Aw)
            print('ddw =', ddw)
            return m_out[:i,:],d_out[:i,:]

        while np.any((m+dm)<0.):
            dm /= 2.

        fnew = f(m+dm)
        while norm(mult(C,Tdata-fnew)) + nu*np.dot(np.dot(tr(m+dm),tr(L)),np.dot(L,m+dm)) > norm(ddw) + nu*np.dot(np.dot(tr(m),tr(L)),np.dot(L,m)):
            dm /= 2.
            fnew = f(m+dm)

        # Update the model
        m += dm

        # Print and store the iteration metrics
        if verbose:
            print('Data residual norm:',norm(Tdata-f(m)))
            print('Norm of model step, dm:',norm(dm))
        m_out[i,:] = m
        d_out[i,:]=fnew

        # If convergence criteria is met, break out of loop
        if i>1 and norm(dm)<solution_tolerance:
            return m_out[:i+1,:],d_out[:i+1,:]
    print('Finished all iterations,',Niter,', without converging.')
    return m_out,d_out

# --------------------------------------------------------------------------------------------------------

def temp(k,kmax,a):
    """
    """
    return a/(k/kmax)


def P(cost,cost_new,T):
    """
    """
    if cost_new<cost:
        return 1.
    else:
        return np.exp(-(cost_new-cost)/T)


def simulated_annealing(f,reg,m,m_step,m_min,m_max,t_m,kmax=1000,a=2,cost=np.nan,T_data=T_data,save_names=['Models','Pred_Data','Cost'],restart=False):
    """
    """

    if restart:
        m_out = np.load(save_names[0]+'.npy')
        m = m_out[-1]
        Ts_out = np.load(save_names[1]+'.npy')
        cost_out = np.load(save_names[2]+'.npy')
        cost = cost_out[-1]

    else:
        # Compute cost of the new model
        print('Run the forward problem on initial model input.')
        predicted_data = f(m)
        cost = np.sum((predicted_data-T_data)**2.)
        if reg is not None:
            cost += reg(m,t_m)

        # Create arrays for outputs
        m_out = np.array([m])
        Ts_out = np.array([predicted_data])
        cost_out = np.array([cost])
        np.save(save_names[0],m_out)
        np.save(save_names[1],Ts_out)
        np.save(save_names[2],cost_out)

    k = 1 # Energy evaluation counter
    while k < kmax:

        # Update the model by some step of a randomly selected parameter
        rand_ind = np.random.randint(0,len(m))
        m_pert = m.copy()
        m_pert[rand_ind] = m[rand_ind] + np.random.normal(0,m_step[rand_ind])
        while m_pert[rand_ind] > m_max[rand_ind] or m_pert[rand_ind] < m_min[rand_ind]:
            m_pert[rand_ind] = m[rand_ind] + np.random.normal(0,m_step[rand_ind])

        # Compute cost of the new model
        predicted_data = f(m_pert)
        cost_pert = np.sum((predicted_data-T_data)**2.)
        if reg is not None:
            cost_pert += reg(m_pert,t_m)
        print('Finished with cost:',cost_pert)

        # Should we move to it?
        if P(cost, cost_pert, temp(k,kmax,a)) > np.random.rand():
            print('Moving to new model and saving output.')
            m = m_pert.copy()
            cost = cost_pert
            # Write model and cost to file
            m_out = np.append(m_out,[m],axis=0) # keep track of energy
            Ts_out = np.append(Ts_out,[predicted_data],axis=0) # keep track of energy
            cost_out = np.append(cost_out,cost) # keep track of energy
            np.save(save_names[0],m_out)
            np.save(save_names[1],Ts_out)
            np.save(save_names[2],cost_out)

        k += 1   # Iterate counter

    return m_out,cost_out

# --------------------------------------------------------------------------------------------------------

def monte_carlo(f,m,msteps,Tdata=T_data,zdata=z_data,norm=norm2,
                    n_iterates=500,regularization=0.):
    """
    Perform a suite of runs to get our distribution. We start with an initial guess, and move around.
    """

    # Initialize
    i = 0

    # Preallocate locations for output--we want all our accepted models saved
    m_out = np.zeros((len(m),n_iterates-1))
    m_out[:, i] = m

    Tmodel = f(m)

    # misfits will have both a regularized and an unregularized row
    misfits = np.zeros((n_iterates))
    misfits[i] = np.sum((Tdata-Tmodel)**2.)

    while i < n_iterates-1:
        # Start iterating
        m_pert = m.copy()
        rand_ind = np.random.randint(0,len(m))
        m_pert[rand_ind] = m[rand_ind] + np.random.normal(0.,msteps[rand_ind])

        Tmodel = f(m_pert)
        misfits[i+1] = np.sum((Tdata-Tmodel)**2.)

        if i < 1 or np.random.rand() < min(1,np.exp(-(misfits[i+1]-misfits[i]))):
            # accept the model
            m_out[:,i] = m_pert
            m = m_pert.copy()
            if i%100 == 0:
                print('Successful Run:',i,np.random.normal(0., msteps),msteps,misfits[i + 1])
            # increment only upon acceptance of the guess
            i += 1

        np.save('Models',m_out)
        np.save('Misfits',misfits)

    return m_out, misfits
