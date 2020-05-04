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
from scipy.integrate import cumtrapz
from supporting_functions import analytical_model, viscosity
from constants import constants
const = constants()

class numerical_model():
    """
    1-D finite difference model for ice temperature based on
    Weertman (1968)

    Assumptions:
        1) Initialize to 1-D Analytical temperature profile from Rezvanbehbahani et al. (2019)
        2) Finite difference solution
        3) Horizontal velocity...
        4) Vertical velocity...
        5) Strain rates...
    """

    def __init__(self,const=const):
        """
        Initialize the model with constant terms
        """

        ### Numerical Inputs ###
        self.nz=101         # Number of layers in the ice column
        self.tol=1e-4       # Convergence criteria

        ### Boundary Constraints ###
        self.Ts = -50.      # Surface Temperature   [C]
        self.qgeo = .050    # Geothermal flux       [W/m2]
        self.H = 2800.      # Ice thickness         [m]
        self.adot = .1      # Accumulation rate     [m/s]
        self.v_surf = 0.    # Surface velocity     [m/yr]

        ### Gradients ###
        self.dTs = 0.                   # Change in air temperature over distance x/y [C/m]
        self.dH = np.sin(.1*np.pi/180.)    # Thickness gradient in x/y directions, used for deformational flow calculation        [m/m]
        self.da = 0.                    # Accumulation gradient in x/y directions     [m/yr/m]

        ### Empty Time Array as Default ###
        self.ts=[]

        ### Flags ###
        self.flags = []

    # ---------------------------------------------

    def initial_conditions(self,const=const):
        """
        Define the initial ice column properties using an analytical solution
        with paramaters from at the beginning of the time series.
        """

        # TODO: Weertman has this extra term
        #v_z_surf = self.adot + v_surf*dH
        if hasattr(self.adot,"__len__"):
            self.z,self.T = analytical_model(self.Ts[0],self.qgeo,self.H,
                    self.adot[0],const=const,nz=self.nz)
            # TODO: linear velocity profile should change eventually
            self.v_z = self.adot[0]*self.z/self.H
        else:
            self.z,self.T = analytical_model(self.Ts,self.qgeo,self.H,
                    self.adot,const=const,nz=self.nz)
            # TODO: linear velocity profile should change eventually
            self.v_z = self.adot[0]*self.z/self.H

        ### Discretize the vertical coordinate ###
        self.dz = np.mean(np.gradient(self.z))      # Vertical step
        self.P = const.rho*const.g*(self.H-self.z)  # Pressure
        self.pmp = self.P*const.beta                # Pressure melting
        self.Tgrad = -self.qgeo/const.k             # Temperature gradient at bed


    def source_terms(self,const=const):
        """
        Heat sources from strain heating and downstream advection
        """

        ### Strain Heat Production ###
        # Shear Stress by Lamellar Flow (van der Veen section 4.2)
        tau_xz = const.rho*const.g*(self.H-self.z)*abs(self.dH)
        # Calculate the viscosity TODO: add option for optimization to surface velocity
        A = viscosity(self.T,self.z,const=const,tau_xz=tau_xz,v_surf=None)
        # Strain rate, Weertman (1968) eq. 7
        eps_xz = A*tau_xz**const.n
        # strain heat term (K s-1) TODO: check that it is ok to use the xz terms instead of the effective terms
        Q = (eps_xz*tau_xz)/(const.rho*const.Cp)

        ### Advection Term ###
        v_x = np.insert(cumtrapz(eps_xz,self.z),0,0)    # Horizontal velocity
        # Horizontal Temperature Gradients, Weertman (1968) eq. 6b
        dTdx = self.dTs + (self.T-np.mean(self.Ts))/2.*(1./self.H*self.dH-(1./np.mean(self.adot))*self.da)

        ### Final Source Term ###
        self.Sdot = Q - v_x*dTdx

    # ---------------------------------------------

    def stencil(self,const=const):
        """
        Finite Difference Scheme for 1-d advection diffusion
        Surface boundary is fixed (air temperature)
        Bed boundary is gradient (geothermal flux)
        """

        # Choose time step
        if 'steady' in self.flags:
            # TODO: what is this?
            self.dt = 0.5*self.dz**2./(const.k/(const.rho*const.Cp))
        else:
            # Check if the time series is monotonically increasing
            if len(self.ts) == 0:
                raise ValueError("If not steady, must input a time array.")
            if not np.all(np.gradient(np.gradient(self.ts))<self.tol):
                raise ValueError("Time series must monotonically increase.")
            self.dt = np.mean(np.gradient(self.ts))
        # Stability, check the CFL
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

        if 'melt' in self.flags:
            self.int_stencil = np.ones_like(self.z)
            self.int_stencil[[0,-1]] = 0.5


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
        Run the finite-difference model as it has been set up through the other functions.
        """
        if 'melt' in self.flags:
            self.Mrate = np.empty((0))
            self.Mcum = np.array([0])

        # iterate through all times
        for i in range(len(self.ts)):
            if i%1000 == 0:
                print(int(self.ts[i]/const.spy),end=',')
            # Update to current time
            self.T[-1] = self.Ts[i]  # set surface boundary condition
            adot_scale = self.adot[i]/self.adot[0]  # advection updates every time step
            # Solve
            T_new = self.A*self.T - adot_scale*self.B*self.T + self.dt*self.Sdot
            self.T = T_new
            # Output basal melting or freezeing
            if 'melt' in self.flags:
                self.melt_output(i,const=const)
            # reset temp to PMP
            self.T[self.T>self.pmp] = self.pmp[self.T>self.pmp]

