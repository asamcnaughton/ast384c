#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:15:06 2026

@author: asamcnaughton
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import scipy as sp
import emcee
import corner


a_true, b_true = 0, 1.75
x = np.array([1,1.5,2,2.5])
y = np.array([1.87640523, 2.59001572, 3.4978738 , 4.47408932])
yerr = 0.1

# plt.figure(figsize=(4,3))
# plt.errorbar(x, y, yerr=yerr, fmt='.k')
# plt.xlabel('x'); plt.ylabel('y')

def model(theta, x):
    a, b = theta
    return a + b * x

def log_likelihood(theta, x, y, yerr):
    m = model(theta, x)
    return -0.5 * np.sum((y - m) ** 2 / yerr**2)

def log_prior(theta):
    a, b = theta
    if -10. < a < 10. and -10.0 < b < 10.0:
        return 0.0
    return -np.inf

def log_posterior(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

_p0 = sp.stats.norm(loc=1, scale=6)
_p1 = sp.stats.norm(loc=-2, scale=1)
def logp(x): # emcee takes in logp as input
    return np.log(0.5*(_p0.pdf(x) + _p1.pdf(x)))
ndim = 2 # p(theta | X) is 2D
nwalkers = 32

theta_guess = [1., 1.] # initial guess 
def neg_log_likelihood(theta, x, y, yerr): 
    return -1 * log_likelihood(theta, x, y, yerr)
theta_est = sp.optimize.fmin(neg_log_likelihood, 
                             theta_guess, 
                             args=(x, y, yerr))
print('theta_MLE =', theta_est)

# create a small ball around the MLE the initialize each walker 
pos = theta_est + 1e-4 * np.random.randn(nwalkers, ndim)


sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                log_posterior, 
                                args=(x, y, yerr))
_ = sampler.run_mcmc(pos, 200, progress=True)

fig, axes = plt.subplots(ndim, sharex=True, figsize=(8,2))
samples = sampler.get_chain()
labels = ["a", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3, rasterized=True)
    ax.set_xlim(0, 200); ax.set_ylabel(labels[i])
axes[-1].set_xlabel("step number")

burnin = 100 
flat_samples = sampler.get_chain(discard=100, flat=True, thin=15)

fig = plt.figure(figsize=(4,4))
fig = corner.corner(flat_samples, labels=labels, 
                    truths=[a_true, b_true], fig=fig)

#%%
# tau = sampler.get_autocorr_time(discard=burnin)
# print(tau)
N_iters, taus = [], []
iters = linspace(2000,10000,50).astype(int)

for N in iters: # loop over different number of iteractions
    # run emcee for N steps
    _ = sampler.run_mcmc(pos, N, progress=True)
    # calculate autocorrelation time
    
    _tau = sampler.get_autocorr_time(discard=burnin)

    N_iters.append(N)
    taus.append(_tau)
#%%

fig = plt.figure(figsize=(8,3))

sub = fig.add_subplot(121)
sub.plot(N_iters, np.array(taus)[:,0])
sub.set_xlabel("$N$", fontsize=14)
sub.set_xscale('log')
sub.set_ylabel(r"$\tau_a$", fontsize=14)
sub = fig.add_subplot(122)
sub.plot(N_iters, np.array(taus)[:,1])
sub.set_xlabel("$N$", fontsize=14)
sub.set_xscale('log')
sub.set_ylabel(r"$\tau_b$", fontsize=14)
plt.suptitle(r'Autocorrelation times $\tau_a$, $\tau_b$ per $N$ iterations',fontsize=14)
plt.tight_layout(pad=1.0)
plt.show()

taua, taub = array(taus)[:,0], array(taus)[:,1]
percenta, percentb = diff(taua)/taua[:-1]*100, diff(taub)/taub[:-1]*100

line1 = np.ones(len(taua))
line2 = -line1 

fig = plt.figure(figsize=(8,3))

sub = fig.add_subplot(121)
sub.plot(N_iters[1:], percenta)
sub.plot(N_iters,line1,ls='dashed',color='red',label=r'$\pm 1 \%$ ')
sub.plot(N_iters,line2,ls='dashed',color='red')
sub.set_xlabel("$N$", fontsize=14)
sub.set_xscale('log')
sub.set_ylabel(r"$\%$ change in $\tau_a$", fontsize=14)
sub = fig.add_subplot(122)
sub.plot(N_iters[1:], percentb)
sub.plot(N_iters,line1,ls='dashed',color='red')
sub.plot(N_iters,line2,ls='dashed',color='red')
sub.set_xlabel("$N$", fontsize=14)
sub.set_xscale('log')
sub.set_ylabel(r"$\%$ change in $\tau_a$", fontsize=14)
fig.legend()
plt.suptitle(r'Autocorrelation times $\tau_a$, $\tau_b$ $\%$ change from last $N$ step',fontsize=14)
plt.tight_layout(pad=1.0)

""" Appears that both tau_a and tau_b converge (< 1% change) after ~3500 iterations in this case."""
""" 4000 would be safe. """

#%%


converged_samples = sampler.get_chain(discard=100, flat=True, thin=15)
fig = plt.figure(figsize=(4,4))
fig = corner.corner(converged_samples, labels=labels, 
                    truths=[a_true, b_true], fig=fig)
tpoint = array([0,1.75])
corner.overplot_points(fig,tpoint[None,:],marker='*',color='red',markersize=12)


















