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

# Times to interpolate on
t_m = 20000*np.log(np.linspace(0,np.exp(max(ts)/20000),5))
t_m[0] = 0.

# Initial Model Guess
q_geo = .06
gamma = 1.
Udef = 10.
Uslide = 0.
m_init = np.array([q_geo,gamma])
m_init = np.append(m_init,Udef*np.ones_like(t_m))
m_init = np.append(m_init,Uslide*np.ones_like(t_m))
# Model step
m_step = np.array([0.01,0.1])
m_step = np.append(m_step,np.ones_like(t_m))
m_step = np.append(m_step,np.ones_like(t_m))
# Model min/max
m_min = np.zeros_like(m_init)
m_max = np.array([.1,5.])
m_max = np.append(m_max,100.*np.ones(len(t_m)*2))

def f(m,H=H,ts=ts,adot=adot,Tsurf=Tsurf,zdata=z_data,tol=1e-4,t_m=t_m):
    print('Running Model')
    print('m:',m)
    fp = numerical_model()
    fp.ts = ts[:]*const.spy
    fp.adot = adot/const.spy
    fp.Ts = Tsurf
    fp.qgeo = m[0]
    fp.gamma = m[1]
    Udef_interp = interp1d(t_m,m_init[2:2+len(t_m)]/const.spy)
    Uslide_interp = interp1d(t_m,m_init[-len(t_m):]/const.spy)
    fp.Udefs = Udef_interp(ts)
    fp.Uslides = Uslide_interp(ts)
    fp.initial_conditions()
    fp.source_terms()
    fp.stencil()
    fp.flags = []
    fp.run()
    Tinterp = interp1d(fp.z,fp.T)
    return Tinterp(H+zdata)

from inverse_model import reg,simulated_annealing
simulated_annealing(f,reg,m_init,m_step,m_min,m_max,t_m,kmax=5)
