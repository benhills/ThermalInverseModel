#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
April 28, 2020
"""

# Import necessary libraries
import numpy as np
from scipy.optimize import least_squares
from scipy import sparse
from analytical_model import analytical_model

def numerical_model(Ts,qgeo,H,adot,const,dTs=[0.,0.],dH=[0.,0.],da=[0.,0.],v_surf=[0.,0.],
               eps_xy=0.,nz=101,steady=True,ts=[],conv_crit=1e-4,tol=1e-4,melt=True):
    """
    1-D finite difference model for ice temperature based on
    Weertman (1968)

    Assumptions:
        1) Initialize from Weertman's version of the Robin (1955) analytic solution.
            This has been checked against the Robin_T function and gives consistent
            results to 1e-14.
        2) Finite difference solution
        3) Horizontal velocity...
        4) Vertical velocity...
        5) Strain rates...

    Parameters
    ----------
    Ts:         float,      Surface Temperature                         [C]
    qgeo:       float,      Geothermal flux                             [W/m2]
    H:          float,      Ice thickness                               [m]
    adot:       float,      Accumulation rate                           [m/s]
    const:      class,      Constants
    dTs:        2x1 array,  Change in air temperature over distance x/y [C/m]
    dH:         2x1 array,  Thickness gradient in x/y directions, used for deformational flow calculation        [m/m]
    da:         2x1 array,  Accumulation gradient in x/y directions     [m/yr/m]
    v_surf:     2x1 array,  Surface velocity in x/y directions          [m/yr]
    eps_xy:     float,      Plane strain rate                           [m/m]
    nz:         int,        Number of layers in the ice column
    steady:     bool,       Decide to run to convergence or in transient
    ts:         array,      If not steady, input temperatures that are monotonically increasing (seconds)
    conv_crit:  float,      Convergence criteria, maximum difference between
                            temperatures at final time step

    Output
    ----------
    z:          1-D array,  Discretized height above bed through the ice column
    T_weertman: 1-D array,  Numerical solution for ice temperature
    T_analytical:    1-D array,  Analytic solution for ice temperature
    T_diff:     1-D array,  Convergence profile (i.e. difference between final and t-1 temperatures)
    M:          1-D array,  Melt rates through time (m/yr)

    """

    # Height above bed array
    z = np.linspace(0,H,nz)
    dz = np.mean(np.gradient(z))
    # pressure melting
    PMP = const.rho*const.g*(H-z)*const.beta

    ###########################################################################

    ### Start with the analytic 'Robin Solution' as an initial condition ###

    # Vertical Velocity
    v_z_surf = adot
    # Weertman has this extra term
    #v_z_surf += v_surf[0]*dH[0]+v_surf[1]*dH[1]
    if hasattr(v_z_surf,"__len__"):
        adv_analytical = v_z_surf[0]*const.spy
        Ts_analytical = Ts[0]
    else:
        adv_analytical = v_z_surf*const.spy
        Ts_analytical = Ts

    # Call the analytic solution from another script
    z_analytical,T_analytical = analytical_model(Ts_analytical,qgeo,H,adv_analytical,const,nz=nz)

    ###########################################################################

    ### Optimize the rate factor to fit the surface velocity ###

    # Shear Stress by Lamellar Flow (van der Veen section 4.2)
    tau_xz = const.rho*const.g*(H-z)*abs(dH[0])
    tau_yz = const.rho*const.g*(H-z)*abs(dH[1])

    # Function to Optimize
    def surfVelOpt(C):
        # Change the coefficient so that the least_squares function takes appropriate steps
        # TODO: (there is likely a better way to do this)
        C_opt = C*1e-13
        # Shear Strain Rate, Weertman (1968) eq. 7
        eps_xz = C_opt*np.exp(-const.Qminus/(const.R*(T_analytical+const.T0)))*tau_xz**const.n
        vx_opt = np.trapz(eps_xz,z)
        return abs(vx_opt-v_surf[0])*const.spy
    # Get the final coefficient value
    res = least_squares(surfVelOpt, 1)
    C_fin = res['x']*1e-13

    # Final Strain Rates, Weertman (1968) eq. 7
    eps_xz = C_fin*np.exp(-const.Qminus/(const.R*(T_analytical+const.T0)))*tau_xz**const.n
    eps_yz = C_fin*np.exp(-const.Qminus/(const.R*(T_analytical+const.T0)))*tau_yz**const.n
    # Horizontal Velocity (integrate the strain rate through the column)
    v_x = np.empty_like(z)
    v_y = np.empty_like(z)
    for i in range(len(z)):
        v_x[i] = np.trapz(eps_xz[:i+1],z[:i+1])
        v_y[i] = np.trapz(eps_yz[:i+1],z[:i+1])

    ###########################################################################

    ### Calculate Strain Heat Production and Advective Sources ###

    # effective stress and strain rate (van der Veen eq. 2.6/2.7)
    tau_e = np.sqrt((2.*tau_xz**2. + 2.*tau_yz**2.)/2.)
    eps_e = np.sqrt((2.*eps_xz**2. + 2.*eps_yz**2.)/2.)

    # strain heat term (K s-1)
    Q = (eps_e*tau_e)/(const.rho*const.Cp)

    # Horizontal Temperature Gradients, Weertman (1968) eq. 6b
    dTdx = dTs[0] + (T_analytical-np.mean(Ts))/2.*(1./H*dH[0]-(1/np.mean(adot))*da[0])
    dTdy = dTs[1] + (T_analytical-np.mean(Ts))/2.*(1./H*dH[1]-(1/np.mean(adot))*da[1])
    # Advection Rates (K s-1)
    Adv_x = -v_x*dTdx
    Adv_y = -v_y*dTdy

    # Final source term
    Sdot = Q + Adv_x + Adv_y

    ###########################################################################

    ### Finite Difference Scheme ###

    # Initial Condition from Robin Solution
    T = T_analytical.copy()
    dz = np.mean(np.gradient(z))
    Tgrad = -qgeo/const.k
    if hasattr(v_z_surf,"__len__"):
        v_z = v_z_surf[0]*z/H
    else:
        v_z = v_z_surf*z/H

    # Choose time step
    if steady:
        dt = 0.5*dz**2./(const.k/(const.rho*const.Cp))
    else:
        # Check if the time series is monotonically increasing
        if len(ts) == 0:
            raise ValueError("If steady=False, must input a time array.")
        if not np.all(np.gradient(np.gradient(ts))<tol):
            raise ValueError("Time series must monotonically increase.")
        dt = np.mean(np.gradient(ts))
    # Stability
    if max(v_z)*dt/dz > 1.:
        print(max(v_z)*dt/dz,dt,dz)
        raise ValueError("Numerically unstable, choose a smaller time step or a larger spatial step.")

    # Stencils
    diff = (const.k/(const.rho*const.Cp))*(dt/(dz**2.))
    A = sparse.lil_matrix((nz, nz))           # create a sparse Matrix
    A.setdiag((1.-2.*diff)*np.ones(nz))            #Set the diagonal
    A.setdiag((1.*diff)*np.ones(nz),k=-1)            #Set the diagonal
    A.setdiag((1.*diff)*np.ones(nz),k=1)            #Set the diagonal
    B = sparse.lil_matrix((nz, nz))           # create a sparse Matrix
    for i in range(len(z)):
        adv = (-v_z[i]*dt/dz)
        B[i,i] = adv
        B[i,i-1] = -adv

    # Boundary Conditions
    # Neumann at bed
    A[0,1] = 2.*diff
    B[0,:] = 0.
    # Dirichlet at surface
    A[-1,:] = 0.
    A[-1,-1] = 1.
    B[-1,:] = 0.

    # Source Term
    Sdot[0] = -2*dz*Tgrad*diff/dt
    Sdot[-1] = 0.

    ###########################################################################

    ### Iterations and Output ###

    if steady:
        # Iterate until convergence
        i = 0
        T_diff = 0.
        while i < 1 or (sum(abs(T_diff)) > conv_crit):
            T_new = A*T - B*T + dt*Sdot
            T_diff = T_new-T
            print('Convergence criteria:', sum(abs(T_diff)))
            T = T_new
            i += 1
            if np.any(T>PMP) and melt:
                Tplus = np.trapz(T[T>PMP]-PMP[T>PMP],z[T>PMP])
                T[T>PMP] = PMP[T>PMP]
                M = Tplus*const.rho*const.Cp/(const.rhow*const.L*dt/const.spy)
        T_weertman = T.copy()

    else:
        T_weertman = np.empty((0,len(z)))
        int_stencil = np.ones_like(z)
        int_stencil[[0,-1]] = 0.5
        Mrate = np.empty((0))
        Mcum = np.array([0])
        # iterate through all times
        for i in range(len(ts)):
            # Update to current time
            t = ts[i]
            if i%100 == 0 or t == ts[-1]:
                print('Start %.0f, current %.0f, end %.0f (years).'%(ts[0]/const.spy,t/const.spy,ts[-1]/const.spy))
                T_weertman = np.append(T_weertman,[T],axis=0)
            Tsurf = Ts[i]
            adv = v_z_surf[i]/v_z_surf[0]
            # set surface boundary condition
            T[-1] = Tsurf
            # solve
            T_new = A*T - adv*B*T + dt*Sdot
            T = T_new
            # If melting
            if np.any(T>PMP) and melt:
                Tplus = (T[T>PMP]-PMP[T>PMP])*int_stencil[T>PMP]*dz
                T[T>PMP] = PMP[T>PMP]
                if i%100 == 0 or t == ts[-1]:
                    Mrate = np.append(Mrate,Tplus*const.rho*const.Cp*const.spy/(const.rhow*const.L*dt))
            # If freezing
            elif Mcum[-1] > 0 and melt:
                Tminus = (T[0]-PMP[0])*0.5*dz
                T[0] = PMP[0]
                if i%100 == 0 or t == ts[-1]:
                    Mrate = np.append(Mrate,Tminus*const.rho*const.Cp*const.spy/(const.rhow*const.L*dt))
            else:
                T[T>PMP] = PMP[T>PMP]
                if i%100 == 0 or t == ts[-1]:
                    Mrate = np.append(Mrate,0.)

            # update the cumulative melt by the melt rate
            if i%100 == 0 or t == ts[-1]:
                # update the cumulative melt by the melt rate
                Mcum = np.append(Mcum,Mcum[-1]+Mrate[-1]*100*dt/const.spy)
                print('dt=',dt/const.spy,'melt=',np.round(Mrate[-1]*1000.,2),np.round(Mcum[-1],2))
        T_diff = T-T_analytical

    try:
        M
    except:
        M = False

    return z,T_weertman,T_analytical,T_diff,Mrate,Mcum
