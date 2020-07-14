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
import multiprocessing as mp
from constants import constants
const = constants()

# Numbers as input to forward model
z_data,T_data,C_data = np.transpose(np.load('./data/icetemp_data.npy'))
Spice_Accumulation = np.load('./data/SP_accumulation_interpolated.npy')
Spice_airTemp = np.load('./data/SP_airTemperature_interpolated.npy')
ts = Spice_Accumulation[0]
adot = Spice_Accumulation[1]
Tsurf = Spice_airTemp[1]

# Model vector
q_geo = .07
p = 10.
udef = 5.
uslide0 = 0.
uslide1 = 0.
t0 = 40.
m_init = np.array([q_geo,p,udef,uslide0,uslide1,t0])
# Model step
m_step = np.array([0.01,0.1,1.,.5,.5,.5,1.])
# Model min/max
m_min = np.zeros_like(m_init)
m_max = np.array([.1,100.,20.,20.,20.,np.nanmax(ts)])


def f(m,H,cum_flag,ts,adot,Tsurf,z_data,tol):
    fp = numerical_model()
    fp.ts = ts[:]*const.spy
    fp.adot = adot/const.spy
    fp.Ts = Tsurf
    fp.H = H
    fp.qgeo = m[0]
    fp.p = m[1]
    fp.Udefs = m[2]/const.spy*np.ones_like(ts)
    Uslides = np.empty_like(ts)
    Uslides[ts<=m[5]*1000.] = m[3]/const.spy
    Uslides[ts>m[5]*1000.] = m[4]/const.spy
    fp.Uslides = Uslides
    fp.initial_conditions()
    fp.source_terms()
    fp.stencil()
    fp.flags = []
    if cum_flag:
        fp.flags.append('water_cum')
        print('Running model with melt flag on.')
    else:
        print('Running model with melt flag off.')
    print('m:',m)
    fp.run()
    return fp

def f_parallel(m,Hs=np.array([2810,2850]),cum_flags=np.array([False,True]),
               ts=ts,adot=adot,Tsurf=Tsurf,z_data=z_data,tol=1e-4):
    # parallelize
    pool = mp.Pool(2)
    results = pool.starmap(f, [(m,Hs[i],cum_flags[i],ts,adot,Tsurf,z_data,tol) for i in range(2)])
    pool.close()
    # interpolate the result onto the data locations
    Tinterp_1 = interp1d(results[0].z,results[0].T)
    Tinterp_2 = interp1d(results[1].z,results[1].T)
    T_out = Tinterp_1(Hs[0]+z_data[:-1])
    T_out = np.append(T_out,Tinterp_2(0))
    return T_out

from inverse_model import simulated_annealing
simulated_annealing(f_parallel,None,m_init,m_step,m_min,m_max,None,kmax=10000,a=2)
