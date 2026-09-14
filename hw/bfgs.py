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

nx = 10
starts = linspace(-10,10,nx)
objx = linspace(-10,10,100)

mins = []
for i in range(nx):
    mins = mins + [opt.minimize(objf,starts[i],method='BFGS').x]

mins = array(mins)
plt.plot(objx,objf(objx),label='Objective function',ls='dashed',color='red')
plt.scatter(starts,objf(starts),label=r'Starting points',color='blue')
plt.scatter(mins,objf(mins),label=r'Objective fcn(minima)',marker='x',color='blue')

plt.title(r'Initial guess $x_0$ vs minimization')
plt.xlabel('Initial guess $x_0$')
plt.ylabel('Minimal value $x$')
plt.legend()