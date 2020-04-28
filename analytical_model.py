#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
April 28, 2020
"""

import numpy as np
from scipy.special import gamma as γ
from scipy.special import gammaincc as γincc
from constants import constants, rateFactor

def analytical_model(Ts,qgeo,H,adot,nz=101,
             const=constants(),
             rateFactor=rateFactor,
             T_ratefactor=-10.,
             dHdx=0.,tau_dx=0.,
             gamma=1.397,gamma_plus=True,
             verbose=False):
    """
    1-D Analytical temperature profile from Rezvanbehbahani et al. (2019)
    Main improvement from the Robin (1955) solution is the nonlinear velocity profile

    Assumptions:
        1) no horizontal advection
        2) vertical advection takes the form v=(z/H)**(gamma)
        3) firn column is treated as equivalent thickness of ice
        TODO: 4) If base is warmer than the melting temperature recalculate with new basal gradient
        5) strain heating is added to the geothermal flux

    Parameters
    ----------
    Ts:     float,  Surface Temperature (C)
    qgeo:   float,  Geothermal flux (W/m2)
    H:      float,  Ice thickness (m)
    adot:   float,  Accumulation rate (m/yr)
    nz:     int,    Number of layers in the ice column
    const:  class,  Constants
    rateFactor:     function, to calculate the rate factor from Glen's Flow Law
    T_ratefactor:   float, Temperature input to rate factor function (C)
    dHdx:       float, Surface slope to calculate tau_dx
    tau_dx:     float, driving stress input directly (Pa)
    gamma:      float, exponent on the vertical velocity
    gamma_plus: bool, optional to determine gama_plus from the logarithmic regression with Pe Number

    Output
    ----------
    z:      1-D array,  Discretized height above bed through the ice column
    T:      1-D array,  Analytic solution for ice temperature
    """

    # if the surface accumulation is input in m/yr convert to m/s
    if adot>1e-5:
        adot/=const.spy
    # Thermal diffusivity
    K = const.k/(const.rho*const.Cp)
    if gamma_plus:
        # Solve for gamma using the logarithmic regression with the Pe number
        Pe = adot*H/K
        if Pe < 5. and verbose:
            print('Pe:',Pe)
            print('The gamma_plus fit is not well-adjusted for low Pe numbers.')
        # Rezvanbehbahani (2019) eq. (19)
        gamma = 1.39+.044*np.log(Pe)
    if dHdx != 0. and tau_dx == 0.:
        # driving stress Nye (1952)
        tau_dx = const.rho*const.g*H*abs(dHdx)
    if tau_dx != 0:
        # Energy from strain heating is added to the geothermal flux
        A = rateFactor(np.array([T_ratefactor]),const)[0]
        # Rezvanbehbahani (2019) eq. (22)
        qgeo_s = (2./5.)*A*H*tau_dx**4.
        qgeo += qgeo_s
    # Rezvanbehbahani (2019) eq. (19)
    lamb = adot/(K*H**gamma)
    phi = -lamb/(gamma+1)
    z = np.linspace(0,H,nz)

    # Rezvanbehbahani (2019) eq. (17)
    Γ_1 = γincc(1/(1+gamma),-phi*z**(gamma+1))*γ(1/(1+gamma))
    Γ_2 = γincc(1/(1+gamma),-phi*H**(gamma+1))*γ(1/(1+gamma))
    term2 = Γ_1-Γ_2
    T = Ts + qgeo*(-phi)**(-1./(gamma+1.))/(const.k*(gamma+1))*term2
    return z,T
