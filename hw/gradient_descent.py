#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:19:21 2026

@author: asamcnaughton
"""

from numpy import *
import scipy.optimize as opt
import scipy as sc
import matplotlib.pyplot as plt

def objf(x,y):
    """ objective function to minimize """
    return((x-3)**2+(y+1)**2)
    
def gradf(x,y):
    """ objective function's analytic derivative"""
    return(2*(x-3)+2*(y+1))

def gradloop(nloops=10,lrate=0.1,start=array([0,0]),keep='no'):
    """
    implementation of gradient descent optimization for 2-dim objective function f(x,y)= (x-3)**2 + (y+1)**2
    Parameters
    ----------
    nloops : # of loops to iterate for optimization. The default is 10.
    lrate : learning rate. The default is 0.1.
    start : coordinates to begin searching at. The default is array([0,0]).
    keep : option to keep full history of optimization path. 
           anything but 'yes' will be treated as 'no'. The default is 'no'.

    Returns: guess of coordinates(x,y), history of optimization path IFF keep=='yes'
    ------
    """
    
    guess = start
    nloops= int(nloops)
    if keep=='yes':
        hist = [start]
    for i in range(nloops):
        np1 = guess - lrate*gradf(*guess)
        if keep=='yes':
            hist = hist + [np1]
        guess = np1
    if keep=='yes':
        return(guess,array(hist).reshape(2,-1))
    else:
        return(guess)

nt = 21
firstgear= gradloop(nloops=nt,lrate=0.01,keep='yes')
secondgear = gradloop(nloops=nt,lrate=0.1,keep='yes')
thirdgear = gradloop(nloops=nt,lrate=0.5,keep='yes')
fourthgear = gradloop(nloops=nt,lrate=1.1,keep='yes')

upb = 7
lob = -4
xs,ys = linspace(lob,upb,nt),linspace(lob,upb,nt)
X,Y = meshgrid(xs,ys)
zs = objf(X,Y)

plt.figure(figsize=(8,6))
ax1 = plt.axes()
ax1.contour(xs,ys,zs,levels=15,cmap='plasma')
ax1.plot(*firstgear[1],ls='dashed',color='green',label=fr'$\lambda=0.01, \ N={nt}$')
ax1.plot(*secondgear[1],ls='dashed',color='red',label=fr'$\lambda=0.1, \ N={nt}$')
ax1.plot(*thirdgear[1],ls='dashed',color='purple',label=fr'$\lambda=0.5, \ N={nt}$')
ax1.plot(*fourthgear[1],ls='dashed',color='orange',label=fr'$\lambda=1.1, \ N={nt}$')
ax1.set_xlim(lob,upb)
ax1.set_ylim(lob,upb)
ax1.legend()
plt.show()















