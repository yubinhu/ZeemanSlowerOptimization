# -*- coding: utf-8 -*-
"""

Calculate the B fild of a coil

B fields are expressed as functions of x

Created on Sun Feb 21 19:11:05 2021

@author: yubin
"""

import scipy.constants as constant
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline

# TODO: pack it all into a class

TESTFLAG = False


def loop_B_field(I,R):
    """
    Calculates the B filed on the axis of a current loop
    @param I current in loop (A)
    @param R radius of loop (m)
    @param x distance from center of loop (m)
    take the right hand direction of current as positive for x and B
    @return B(x) a function that takes an argument x in m and returns B in T
    """
    def B(x):
        r_squared = R**2+x**2
        if x == 0:
            theta = constant.pi/2
        else:
            theta = math.atan(R/abs(x))
        return constant.mu_0*I*R*math.sin(theta)/(2*r_squared)
    return B

def layer_B_field(I, R, d, n):
    """
    Calculate the B field for a layer of coil
    @param d diameter of wire (m)
    @param R radius of each loop (m)
    @param x distance from the center of the left most loop (m)
    @return B(x) a function that takes an argument x in m and returns B in T
    """
    B_loop = loop_B_field(I,R)
    def B(x):
        result = 0
        for i in range(n):
            offset = i*d
            result += B_loop(x-offset)
        return result
    return B

def profile_generator():
    # coil setup
    d = 1.6277*10**(-3) #awg14
    R = 2.9*10**(-2) #inner diameter
    n = 62 #winds
    B = layer_B_field(1, R, d, n)

    #takes in m, output in gauss
    def B_prof(x):
        return 10000*B(x)
    
    def B_prof_out(xl):
        return np.array([10000*B(x) for x in xl])

    def dBdx_out(xl):
        result = np.zeros(len(xl))
        for i in range(len(xl)):
            result[i] = derivative(B_prof,xl[i],0.001,n=1)
        return result
    
    def d2Bdx2_out(xl):
        result = np.zeros(len(xl))
        for i in range(len(xl)):
            result[i] = derivative(B_prof,xl[i],0.001,n=2)
        return result
    
    return B_prof_out, dBdx_out, d2Bdx2_out
    



def sample(N,xmin,xmax):
    """
    Return a list of B_prof(x)  in cm and gauss
    """
    
    # coil setup
    d = 1.6277*10**(-3) #awg14
    R = 2.9*10**(-2) #inner diameter
    n = 62 #winds
    B = layer_B_field(1, R, d, n)
        
    #takes in cm, output in gauss
    def B_prof(x):
        if TESTFLAG:
            return 0
        x_in_m = x/100
        return 10000*B(x_in_m)
    
    x = list(np.linspace(xmin,xmax,N))
    y = [B_prof(i) for i in x]
    return x,y
    

def main():
    # for testing
    test = 1
    if test==0:
        bluex,bluey = sample()
    if test==1:
        bluex = np.linspace(-1,1,2000)
        B_profile, dBdx, d2Bdx2 = profile_generator()
        bluey = [B_profile([i]) for i in bluex]
        firstDer = [dBdx([i]) for i in bluex]
        secondDer = [d2Bdx2([i]) for i in bluex]

    plt.plot(bluex,bluey)
    plt.plot(bluex,firstDer)
    plt.plot(bluex,secondDer)
    plt.show()

def testUS():
    N = 200
    xmin = -100
    xmax = 100
    x,y = sample(N,xmin,xmax)
    print(x)
    profile = UnivariateSpline(x,y,s=0.1)
    plt.plot(x,y)
    plt.plot(np.linspace(min(x),max(x),N),profile(np.linspace(min(x),max(x),N)))
    plt.show()

# testUS()

# main()