#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:18:36 2026

@author: asamcnaughton
"""
import numpy as np
from numpy import *

rng = random.default_rng()

def piest(npts=10):
    
    pts = rng.uniform(-1.0,1.0,(2,npts))
    def rad(x,y):
      return(x**2 + y**2)
    radii = rad(*pts)
    outorin = ones(npts)
    piest = np.sum(outorin[radii<1.0])*4/npts
    return(piest)
    
    
trialnums = logspace(1,7,10).astype(int)
ests = []
for n in trialnums:
    ests = ests + [piest(n)]
ests = array(ests)
print(ests)

