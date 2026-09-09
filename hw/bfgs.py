#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:00:59 2026

@author: asamcnaughton
"""
from numpy import *
import scipy.optimize as opt
import scipy as sc
import matplotlib.pyplot as plt

def objf(x):
    return(sin(5*x) +0.1*x**2)

nx = 100
starts = linspace(-10,10,nx)

mins = []
for i in range(nx):
    mins = mins + [opt.minimize(objf,starts[i],method='BFGS').x]
    
    
plt.plot(starts,mins,label='Minima')
plt.plot(starts,objf(starts),label='Objective function',ls='dashed',color='red')
plt.title(r'Initial guess $x_0$ vs minimization')
plt.xlabel('Initial guess $x_0$')
plt.ylabel('Minimal value $x$')
plt.legend()