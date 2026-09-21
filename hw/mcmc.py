#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:21:53 2026
@author: asamcnaughton
"""

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import scipy as sp

np.random.seed(1)
rng = random.default_rng()
_p0 = sp.stats.norm(loc=1, scale=6)
_p1 = sp.stats.norm(loc=-2, scale=1)
def p(x): 
    # this is your target distribution
    return(0.5*(_p0.pdf(x) + _p1.pdf(x)))


def proposal_sample(x, scale=1): 
    # sample proposal distribution (simple Gaussian distribution)
    # x'~q(x'|x) = N(x, scale)
    q = sp.stats.norm(loc=x, scale=scale)
    return(q.rvs())

def mcmc_metro(S=10000, x_0=0, scale=1.5):
    # S = number of steps
    # x_0 = starting position
    # scale = Sigma of the proposal
    
    samples = []
    accepted = [] 
    x = x_0 
    samples.append(x) 
    for i in range(1, S): 
        
        x1 = proposal_sample(x,scale=scale)
        # sample from proposal q(x'|x)
        r  = np.min((1.0,p(x1)/p(x)))
        # calculate acceptance ratio r(x'|x)
        u = rng.uniform(0,1)
        # sample u ~ Uniform(0,1)
        
        if r > u: #
            x = x1
            accepted.append(1)
        else: # reject
            
            accepted.append(0)
        samples.append(x)
        
    return(samples, np.mean(accepted))


mcxs, mcaccept= mcmc_metro(S=10000, x_0=0, scale=20)

print(r'Mean acceptance rate:',mcaccept)

nsamp = int(1e4)
realx = linspace(-20,20,nsamp)
realy = p(realx)
plt.plot(realx,realy,label=r'p(x)',color='red',lw=2)
plt.hist(mcxs,bins=40,density=True,color='skyblue',label='Metro-Hastings samples')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Test MCMC Metro-Hastings algo samples')

plt.show()









