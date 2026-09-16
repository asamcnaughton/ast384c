#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:50:07 2026

@author: asamcnaughton
"""

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

rng = random.default_rng()

alpha1,alpha2, m_break = 1.3,2.3,0.5
def kroupa_shape(m):
    ''' unnormalized Kroupa IMF shape'''
    m = asarray(m, dtype=float)
    return where(m < m_break, (m / m_break)**(-alpha1), (m / m_break)**(-alpha2))

def samplef(x):
    return((x*1.8)**-2)

def invcdf(y):
    return(- 154321/(500000*y-192901))



nsamp = int(3e1)
pts = rng.uniform(log10(0.08),log10(100.0),nsamp)
propsamps = invcdf(10**pts)


kroupxs = logspace(log10(0.08),2,1000)
kroupys = kroupa_shape(kroupxs)
testys = samplef(kroupxs)

plt.loglog(kroupxs,kroupys,color='orange',lw=2,label='Analytic Kroupa dist')
plt.loglog(kroupxs,testys,color='blue',lw=2,label='Proposed dist')
plt.scatter(10**pts,propsamps,color='green',label='samples')
plt.title('Kroupa exact vs sampled')
plt.xlabel(r'Stellar mass ($M_\odot$)')
plt.ylabel(r'Probability, unnormalized')
plt.legend()
plt.show()









