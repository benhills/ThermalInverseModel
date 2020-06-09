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
from scipy.interpolate import interp1d
from forward_model import numerical_model
from constants import constants
const = constants()

# Numbers as input to forward model
z_data,T_data,C_data = np.transpose(np.load('./data/icetemp_data.npy'))
Spice_Accumulation = np.load('./data/SP_accumulation_interpolated.npy')
Spice_airTemp = np.load('./data/SP_airTemperature_interpolated.npy')
ts = Spice_Accumulation[0]
adot = Spice_Accumulation[1]
Tsurf = Spice_airTemp[1]
H = 2850.

# Use constant uncertainty
C = np.diag(1/.15*np.ones(len(C_data)))

# Times to interpolate on
t_m = 20000*np.log(np.linspace(0,np.exp(max(ts)/20000),5))
t_m[0] = 0.

m_init = np.array([6.148453, 1.32109605, 7.33058424, 6.31995724, 4.53084996, 3.58193804,
                   2.85807974, 1.94213404, 0.58012404, 0.1134321,  0.27526652, 3.08104113])
m_step = np.array([.1,0.1])
m_step = np.append(m_step,.1*np.ones_like(t_m))
m_step = np.append(m_step,.1*np.ones_like(t_m))

def f(m,H=H,ts=ts,adot=adot,Tsurf=Tsurf,zdata=z_data,tol=1e-4,t_m=t_m):
    print('Running Model')
    print('m:',m)
    fp = numerical_model()
    fp.ts = ts[:]*const.spy
    fp.adot = adot/const.spy
    fp.Ts = Tsurf
    fp.qgeo = m[0]/100.
    fp.gamma = m[1]
    Udef_interp = interp1d(t_m,m[2:2+len(t_m)]/const.spy)
    Uslide_interp = interp1d(t_m,m[-len(t_m):]/const.spy)
    fp.Udefs = Udef_interp(ts)
    fp.Uslides = Uslide_interp(ts)
    fp.initial_conditions()
    fp.source_terms()
    fp.stencil()
    fp.flags = []
    fp.run()
    Tinterp = interp1d(fp.z,fp.T)
    return Tinterp(H+zdata)

L = np.zeros((len(m_init),len(m_init)))
for i in np.arange(3,6):
    h1 = (t_m[i-1]-t_m[i-2])/10000.
    h2 = (t_m[i-2]-t_m[i-3])/10000.
    L[i,i] = -2.*(h1+h2)/(h1*h2*(h1+h2))
    L[i,i+1] = 2.*h1/(h1*h2*(h1+h2))
    L[i,i-1] = 2.*h2/(h1*h2*(h1+h2))
L[2,2] = -2./((t_m[1]-t_m[0])/10000.)
L[2,3] = +2./((t_m[1]-t_m[0])/10000.)
L[6,6] = -2./((t_m[-1]-t_m[-2])/10000.)
L[6,5] = +2./((t_m[-1]-t_m[-2])/10000.)
for i in np.arange(7,len(m_init)):
    L[i,i] = 1.

from inverse_model import weakly_nonlinear
def norm2(r):
    return np.nansum(r**2.)
nu = 0.
m_out,d_out = weakly_nonlinear(f,norm2,z_data,T_data,C,m_init,m_step,nu,L,Niter=10,
                               solution_tolerance=1e-5)
