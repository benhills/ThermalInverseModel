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
from scipy import sparse
from analytical_model import analytical_model, viscosity
from constants import constants
const = constants()

class numerical_model():
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
    """

    def __init__(self,const=const):
        """
        Initialize the
        """

        ### Boundary Constraints ###
        self.Ts             # Surface Temperature   [C]
        self.qgeo           # Geothermal flux       [W/m2]
        self.H              # Ice thickness         [m]
        self.adot           # Accumulation rate     [m/s]

        ### Gradients ###
        self.dTs=0.         # Change in air temperature over distance x/y [C/m]
        self.dH= 0.         # Thickness gradient in x/y directions, used for deformational flow calculation        [m/m]
        self.da = 0.        # Accumulation gradient in x/y directions     [m/yr/m]
        self.v_surf = 0.    # Surface velocity     [m/yr]
        self.nz=101         # Number of layers in the ice column
        self.ts=[]          # Times

        ### Convergence criteria, maximum difference between temperatures at final time step
        self.tol=1e-4

        ### Flags ###
        self.flags = ['melt']

        # Height above bed array
        self.z = np.linspace(0,self.H,self.nz)
        self.dz = np.mean(np.gradient(self.z))
        # pressure melting
        self.PMP = const.rho*const.g*(self.H-self.z)*const.beta

        ### Start with the analytic 'Robin Solution' as an initial condition ###

        # Vertical Velocity
        self.v_z_surf = self.adot
        # Weertman has this extra term
        #v_z_surf += v_surf[0]*dH[0]+v_surf[1]*dH[1]
        if hasattr(self.v_z_surf,"__len__"):
            self.adv_analytical = self.v_z_surf[0]*const.spy
            self.Ts_analytical = self.Ts[0]
        else:
            self.adv_analytical = self.v_z_surf*const.spy
            self.Ts_analytical = self.Ts

        # Call the analytic solution from another script
        self.z_analytical,self.T_analytical = analytical_model(self.Ts_analytical,
                self.qgeo,self.H,self.adv_analytical,const,nz=self.nz)

        # Initial Condition from Robin Solution
        # TODO: linear velocity profile should change eventually
        self.T = self.T_analytical.copy()
        self.Tgrad = -self.qgeo/const.k
        if hasattr(self.v_z_surf,"__len__"):
            self.v_z = self.v_z_surf[0]*self.z/self.H
        else:
            self.v_z = self.v_z_surf*self.z/self.H

        if 'melt' in self.flags:
            self.int_stencil = np.ones_like(self.z)
            self.int_stencil[[0,-1]] = 0.5
            self.Mrate = np.empty((0))
            self.Mcum = np.array([0])

    # ---------------------------------------------


    def source_terms(self,const=const):
        """
        """

        ### Optimize the rate factor to fit the surface velocity ###

        # Shear Stress by Lamellar Flow (van der Veen section 4.2)
        tau_xz = const.rho*const.g*(self.H-self.z)*abs(self.dH)

        A = viscosity(tau_xz)
        # Final Strain Rates, Weertman (1968) eq. 7
        eps_xz = A*tau_xz**const.n

        # strain heat term (K s-1) TODO: check that it is ok to use the xz terms instead of the effective terms
        Q = (eps_xz*tau_xz)/(const.rho*const.Cp)

        # Horizontal Velocity (integrate the strain rate through the column)
        v_x = np.empty_like(self.z)
        for i in range(len(self.z)):
            v_x[i] = np.trapz(eps_xz[:i+1],self.z[:i+1])

        # Horizontal Temperature Gradients, Weertman (1968) eq. 6b
        dTdx = self.dTs + (self.T_analytical-np.mean(self.Ts))/2.*(1./self.H*self.dH-(1/np.mean(self.adot))*self.da)
        # Advection Rates (K s-1)
        Adv_x = -v_x*dTdx

        # Final source term
        self.Sdot = Q + Adv_x

    # ---------------------------------------------

    def stencil(self,const=const):
        """
        Finite Difference Scheme
        """

        # Choose time step
        if 'steady' in self.flags:
            self.dt = 0.5*self.dz**2./(const.k/(const.rho*const.Cp))
        else:
            # Check if the time series is monotonically increasing
            if len(self.ts) == 0:
                raise ValueError("If steady=False, must input a time array.")
            if not np.all(np.gradient(np.gradient(self.ts))<self.tol):
                raise ValueError("Time series must monotonically increase.")
            self.dt = np.mean(np.gradient(self.ts))
        # Stability
        if max(self.v_z)*self.dt/self.dz > 1.:
            print(max(self.v_z)*self.dt/self.dz,self.dt,self.dz)
            raise ValueError("Numerically unstable, choose a smaller time step or a larger spatial step.")

        # Stencils
        diff = (const.k/(const.rho*const.Cp))*(self.dt/(self.dz**2.))
        self.A = sparse.lil_matrix((self.nz, self.nz))           # Create a sparse Matrix
        self.A.setdiag((1.-2.*diff)*np.ones(self.nz))            # Set the diagonal
        self.A.setdiag((1.*diff)*np.ones(self.nz),k=-1)          # Set the diagonal
        self.A.setdiag((1.*diff)*np.ones(self.nz),k=1)           # Set the diagonal
        self.B = sparse.lil_matrix((self.nz, self.nz))           # Create a sparse Matrix
        for i in range(len(self.z)):
            adv = (-self.v_z[i]*self.dt/self.dz)
            self.B[i,i] = adv
            self.B[i,i-1] = -adv

        # Boundary Conditions
        # Neumann at bed
        self.A[0,1] = 2.*diff
        self.B[0,:] = 0.
        # Dirichlet at surface
        self.A[-1,:] = 0.
        self.A[-1,-1] = 1.
        self.B[-1,:] = 0.

        # Source Term
        self.Sdot[0] = -2*self.dz*self.Tgrad*diff/self.dt
        self.Sdot[-1] = 0.

    # ---------------------------------------------

    def melt_output(self,i,const=const):
        """
        """
        # If melting
        if np.any(self.T>self.PMP):
            Tplus = (self.T[self.T>self.PMP]-self.PMP[self.T>self.PMP])*self.int_stencil[self.T>self.PMP]*self.dz
            if i%100 == 0 or self.ts[i] == self.ts[-1]:
                self.Mrate = np.append(self.Mrate,Tplus*const.rho*const.Cp*const.spy/(const.rhow*const.L*self.dt))
        # If freezing
        elif self.Mcum[-1] > 0:
            Tminus = (self.T[0]-self.PMP[0])*0.5*self.dz
            self.T[0] = self.PMP[0]
            if i%100 == 0 or self.ts[i] == self.ts[-1]:
                self.Mrate = np.append(self.Mrate,Tminus*const.rho*const.Cp*const.spy/(const.rhow*const.L*self.dt))
        else:
            if i%100 == 0 or self.ts[i] == self.ts[-1]:
                self.Mrate = np.append(self.Mrate,0.)
        # update the cumulative melt by the melt rate
        if i%100 == 0 or self.ts[i] == self.ts[-1]:
            # update the cumulative melt by the melt rate
            self.Mcum = np.append(self.Mcum,self.Mcum[-1]+self.Mrate[-1]*100*self.dt/const.spy)
            print('dt=',self.dt/const.spy,'melt=',np.round(self.Mrate[-1]*1000.,2),np.round(self.Mcum[-1],2))

    # ---------------------------------------------

    def run(self,const=const):
        """
        """
        # iterate through all times
        for i in range(len(self.ts)):
            # Update to current time
            Tsurf = self.Ts[i]
            # advection updates every time step TODO: this can be nonlinear
            adv = self.v_z_surf[i]/self.v_z_surf[0]
            # set surface boundary condition
            self.T[-1] = Tsurf
            # solve
            T_new = self.A*self.T - adv*self.B*self.T + self.dt*self.Sdot
            self.T = T_new

            if 'melt' in self.flags:
                self.melt_output(i,const=const)

            # reset temp to PMP
            self.T[self.T>self.PMP] = self.PMP[self.T>self.PMP]
