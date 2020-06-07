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
from supporting_functions import analytical_model, viscosity, Robin_T
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
        self.H = 2857.      # Ice thickness         [m]
        self.adot = .1/const.spy      # Accumulation rate     [m/s]
        self.gamma = 1.532
        self.p = 0.      # Lliboutry shape factor for vertical velocity (large p is ~linear)

        ### Gradients ###
        self.dTs = 0.                   # Change in air temperature over distance x/y [C/m]
        self.dH = np.sin(.2*np.pi/180.)    # Thickness gradient in x/y directions, used for deformational flow calculation        [m/m]
        self.da = 0.                    # Accumulation gradient in x/y directions     [m/yr/m]

        ### Velocity Terms ###
        self.Udef, self.Uslide = 0., 0.

        ### Melting Conditions ###
        self.Mrate = 0.
        self.Mcum = 0.

        ### Empty Time Array as Default ###
        self.ts=[]

        ### Flags ###
        self.flags = ['verbose']

    # ---------------------------------------------

    def initial_conditions(self,const=const,analytical='Robin'):
        """
        Define the initial ice column properties using an analytical solution
        with paramaters from at the beginning of the time series.
        """

        if len(self.ts)>0 and 'Udefs' not in vars(self):
            if 'verbose' in self.flags:
                print('No velocity arrays set, setting to constant value.')
            self.Udefs, self.Uslides = self.Udef*np.ones_like(self.ts), self.Uslide*np.ones_like(self.ts)

        # Weertman (1968) has this extra term to add to the vertical velocity
        if hasattr(self.adot,"__len__"):
            v_z_surf = self.adot[0] + self.Udef*self.dH
            T_surf = self.Ts[0]
        else:
            v_z_surf = self.adot + self.Udef*self.dH
            T_surf = self.Ts

        # initial temperature from analytical solution
        if analytical == 'Robin':
            self.z,self.T = Robin_T(T_surf,self.qgeo,self.H,
                    v_z_surf,const=const,nz=self.nz)
        elif analytical == 'Rezvan':
            self.z,self.T = analytical_model(T_surf,self.qgeo,self.H,
                    v_z_surf,const=const,nz=self.nz,gamma=self.gamma,gamma_plus=False)
        # vertical velocity
        if self.p == 0.:
            # by exponent, gamma
            self.v_z = v_z_surf*(self.z/self.H)**self.gamma
        else:
            # by shape factor, p
            zeta = (1.-(self.z/self.H))
            self.v_z = v_z_surf*(1.-((self.p+2.)/(self.p+1.))*zeta+(1./(self.p+1.))*zeta**(self.p+2.))

        ### Discretize the vertical coordinate ###
        self.dz = np.mean(np.gradient(self.z))      # Vertical step
        self.P = const.rho*const.g*(self.H-self.z)  # Pressure
        self.pmp = self.P*const.beta                # Pressure melting


    def source_terms(self,const=const):
        """
        Heat sources from strain heating and downstream advection
        """

        ### Strain Heat Production ###
        # Shear Stress by Lamellar Flow (van der Veen section 4.2)
        tau_xz = const.rho*const.g*(self.H-self.z)*abs(self.dH)
        # Calculate the viscosity
        A = viscosity(self.T,self.z,const=const,tau_xz=tau_xz,v_surf=self.Udef*const.spy)
        # Strain rate, Weertman (1968) eq. 7
        eps_xz = (A*tau_xz**const.n)/const.spy
        # strain heat term (K s-1)
        Q = 2.*(eps_xz*tau_xz)/(const.rho*const.Cp)
        # Sliding friction heat production
        self.tau_b = tau_xz[0]
        self.q_b = self.tau_b*self.Uslide

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
            # set time step with CFL
            self.dt = 0.5*self.dz/np.max(self.v_z)
        else:
            # Check if the time series is monotonically increasing
            if len(self.ts) == 0:
                raise ValueError("If not steady, must input a time array.")
            if not np.all(np.gradient(np.gradient(self.ts))<self.tol):
                raise ValueError("Time series must monotonically increase.")
            self.dt = np.mean(np.gradient(self.ts))
        # Stability, check the CFL
        if np.max(self.v_z)*self.dt/self.dz > 1.:
            print('CFL = ',max(self.v_z)*self.dt/self.dz,'; cannot be > 1.')
            print('dt = ',self.dt/const.spy,' years')
            print('dz = ',self.dz,' meters')
            raise ValueError("Numerically unstable, choose a smaller time step or a larger spatial step.")

        # Stencils
        self.diff = (const.k/(const.rho*const.Cp))*(self.dt/(self.dz**2.))
        self.A = sparse.lil_matrix((self.nz, self.nz))           # Create a sparse Matrix
        self.A.setdiag((1.-2.*self.diff)*np.ones(self.nz))            # Set the diagonal
        self.A.setdiag((1.*self.diff)*np.ones(self.nz),k=-1)          # Set the diagonal
        self.A.setdiag((1.*self.diff)*np.ones(self.nz),k=1)           # Set the diagonal
        self.B = sparse.lil_matrix((self.nz, self.nz))           # Create a sparse Matrix
        for i in range(len(self.z)):
            adv = (-self.v_z[i]*self.dt/self.dz)
            self.B[i,i] = adv
            self.B[i,i-1] = -adv

        # Boundary Conditions
        # Neumann at bed
        self.A[0,1] = 2.*self.diff
        self.B[0,:] = 0.
        # Dirichlet at surface
        self.A[-1,:] = 0.
        self.A[-1,-1] = 1.
        self.B[-1,:] = 0.

        # Source Term
        self.Tgrad = -(self.qgeo+self.q_b)/const.k             # Temperature gradient at bed
        self.Sdot[0] = -2*self.dz*self.Tgrad*self.diff/self.dt
        self.Sdot[-1] = 0.

        # Integration stencil to calculate melt near the bottom of the profile
        self.int_stencil = np.ones_like(self.z)
        self.int_stencil[[0,-1]] = 0.5

    # ---------------------------------------------

    def run(self,const=const):
        """
        Run the finite-difference model as it has been set up through the other functions.
        """

        if 'save_all' in self.flags:
            self.Ts_out = np.empty((0,len(self.T)))
            self.Mrate_all = np.empty((0))
            self.Mcum_all = np.array([0])

        # Run the initial conditions until stable
        T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
        steady_iter = 0
        if 'verbose' in self.flags:
            print('Initializing',end='')
        while any(abs(self.T[1:]-T_new[1:])>self.tol):
            if 'verbose' in self.flags and steady_iter%1000==0:
                print('.',end='')
            self.T = T_new.copy()
            T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
            T_new[T_new>self.pmp] = self.pmp[T_new>self.pmp]
            steady_iter += 1
        self.T = T_new.copy()
        if 'verbose' in self.flags:
            print('')

        # If a steady state model is desired
        if 'steady' in self.flags:
            if 'verbose' in self.flags:
                print('Exiting model at steady state condition.')
            # Run one more time to see how much things are changing still
            T_steady = self.A*self.T - self.B*self.T + self.dt*self.Sdot
            T_steady[T_steady>self.pmp] = self.pmp[T_steady>self.pmp]
            self.Ts_out = np.append(self.Ts_out,[self.T],axis=0)
            self.Ts_out = np.append(self.Ts_out,[T_steady],axis=0)
            return

        # ---

        # If a non-steady model is desired iterate through all times
        for i in range(len(self.ts)):

            ### Print and output
            if i%1000 == 0 or self.ts[i] == self.ts[-1]:
                if 'verbose' in self.flags:
                    print('t =',int(self.ts[i]/const.spy),'; dt =',self.dt/const.spy,'; melt rate =',np.round(self.Mrate*1000.,2),'; melt cum = ',np.round(self.Mcum,2),'; q_b = ',self.q_b)
                if 'save_all' in self.flags:
                    self.Mrate_all = np.append(self.Mrate_all,self.Mrate)
                    self.Mcum_all = np.append(self.Mcum_all,self.Mrate)
                    self.Ts_out = np.append(self.Ts_out,[self.T],axis=0)

            ### Update to current time
            self.Udef,self.Uslide = self.Udefs[i],self.Uslides[i]   # update the velocity terms from input
            self.T[-1] = self.Ts[i]  # set surface temperature condition from input
            v_z_surf = self.adot[i] + self.Udef*self.dH # set vertical velocity from input terms (accumulation and surface velocity)
            if self.p == 0.: # by exponent, gamma
                self.v_z = self.Mrate + v_z_surf*(self.z/self.H)**self.gamma
            else: # by shape factor, p
                zeta = (1.-(self.z/self.H))
                self.v_z = v_z_surf*(1.-((self.p+2.)/(self.p+1.))*zeta+(1./(self.p+1.))*zeta**(self.p+2.))
            for i in range(len(self.z)):
                adv = (-self.v_z[i]*self.dt/self.dz)
                self.B[i,i] = adv
                self.B[i,i-1] = -adv
            # Boundary Conditions
            self.B[0,:] = 0.  # Neumann at bed
            self.B[-1,:] = 0. # Dirichlet at surface
            if i%1000 == 0: # Only update the deformational heat source periodically because it is computationally expensive
                self.source_terms()
            self.q_b = self.tau_b*self.Uslide # Update sliding heat flux
            self.Tgrad = -(self.qgeo+self.q_b)/const.k  # Temperature gradient at bed updated from sliding heat flux
            self.Sdot[0],self.Sdot[-1] = -2*self.dz*self.Tgrad*self.diff/self.dt, 0. # update boundaries on heat source vector

            ### Solve
            T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
            self.T = T_new

            ### Calculate the volume melted/frozen during the time step.
            if np.any(self.T>self.pmp): # If Melting
                Tplus = (self.T[self.T>self.pmp]-self.pmp[self.T>self.pmp])*self.int_stencil[self.T>self.pmp]*self.dz
                self.Mrate = Tplus*const.rho*const.Cp*const.spy/(const.rhow*const.L*self.dt)
            elif self.Mcum > 0: # If freezing
                Tminus = (self.T[0]-self.pmp[0])*0.5*self.dz
                self.T[0] = self.pmp[0]
                self.Mrate = Tminus*const.rho*const.Cp*const.spy/(const.rhow*const.L*self.dt)
            self.Mcum += self.Mrate*self.dt/const.spy # Update the cumulative melt by the melt rate

            ### reset temp to PMP
            self.T[self.T>self.pmp] = self.pmp[self.T>self.pmp]
