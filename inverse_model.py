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
from forward_model import numerical_model
from scipy.interpolate import interp1d
from constants import constants
const = constants()

mult = np.matmul
inv = np.linalg.inv
tr = np.transpose

z_data,T_data,C_data = np.transpose(np.load('./data/icetemp_data.npy'))
Spice_Accumulation = np.load('./data/SP_accumulation_interpolated.npy')
Spice_airTemp = np.load('./data/SP_airTemperature_interpolated.npy')

ts = Spice_Accumulation[0]
adot = Spice_Accumulation[1]
Tsurf = Spice_airTemp[1]
H = 2850.
t_m = np.arange(min(ts),max(ts)+10000,10000)

# Initial Guess
qgeo = .06
gamma = 1.
dH = np.sin(.05*np.pi/180.)
sr = .5
# Create model array and model step array
m_init = np.array([qgeo,gamma])
mstep = np.array([.001,.01])
m_init = np.append(m_init,dH*np.ones(7))
mstep = np.append(mstep,dH/10.*np.ones(7))
m_init = np.append(m_init,sr*np.ones(7))
mstep = np.append(mstep,.01*np.ones(7))

def f(m,H=H,ts=ts,adot=adot,Tsurf=Tsurf,zdata=z_data,tol=1e-4):
    print('Running Model')
    print('m:',m)
    fp = numerical_model()
    fp.ts = ts[:]*const.spy
    fp.adot = adot/const.spy
    fp.Ts = Tsurf
    fp.qgeo = m[0]
    fp.gamma = m[1]

    dH_interp = interp1d(t_m,m[2:9])
    fp.dHs = dH_interp(ts)
    fp.dH = fp.dHs[0]
    sr_interp = interp1d(t_m,m[9:])
    fp.srs = sr_interp(ts)
    fp.sliding_ratio = fp.srs[0]
    fp.initial_conditions(analytical='Rezvan')
    fp.source_terms()
    fp.stencil()
    fp.tol = tol
    fp.run(verbose=True)

    Tinterp = interp1d(fp.z,fp.Ts_out[-1])
    return Tinterp(H+zdata)


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


monte_carlo(f,m_init,mstep,n_iterates=1000)
