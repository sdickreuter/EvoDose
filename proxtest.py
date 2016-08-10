import numpy as np

import matplotlib as mpl

from plotsettings import *

import seaborn as sns
import matplotlib.pyplot as plt

from numba import jit,float64,int64

import math

import time

from scipy.stats import linregress
from scipy import integrate


alpha = 32.9 #nm
beta = 2610 #nm
gamma = 2*4.1 #nm
eta_1 = 1.66
eta_2 = 1.27

#http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
#@jit(float64(float64,float64,float64,float64,float64,float64),nopython=True)
def calc_prox(r,normalization,alpha,beta,gamma,eta_1,eta_2):
    return (normalization) * (1/(np.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*np.exp(-r**2/alpha**2) + (eta_1/beta**2)*np.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*np.exp(-np.sqrt(r/gamma)) )

# [return] = C/nm !!!
integrand = lambda x: 2*x*np.pi*(calc_prox(x,1,alpha,beta,gamma,eta_1,eta_2))
normalization, error = integrate.quad(integrand, 0, np.inf)
print('norm:'+str(normalization))
#normalization = 2.41701729505915
integrand = lambda x: 2*x*np.pi*(calc_prox(x,normalization,alpha,beta,gamma,eta_1,eta_2))
print(integrate.quad(integrand,0,np.inf))

x = np.linspace(0,100,500)
y = calc_prox(x,1, alpha, beta, gamma, eta_1, eta_2)

name = "proximity.png"
name_pgf = name[:-4] + ".pgf"
fig = newfig(0.9)
plt.semilogy(x,y)
plt.xlabel('r / nm')
plt.ylabel('Dosis [a.u.]')
plt.tight_layout(.5)
plt.savefig(name,dpi=300)
plt.savefig(name_pgf)
plt.close()
plt.show()

def normalized(x,alpha,beta,gamma,eta_1,eta_2):
    a = calc_prox(0,normalization, alpha, beta, gamma, eta_1, eta_2)
    return (calc_prox(x,normalization, alpha, beta, gamma, eta_1, eta_2))/a


n = 10

print('alpha:')
for i in range(n):
    y = normalized(x,alpha+(alpha*i*1.05),beta,gamma,eta_1,eta_2)
    print(alpha+(alpha*i*1.05))
    plt.plot(x,y)

plt.show()
print('---')

print('beta:')
for i in range(n):
    y = normalized(x,alpha,beta+(beta*i*1.05),gamma,eta_1,eta_2)
    print(beta+(beta*i*1.05))
    plt.plot(x,y)

plt.show()
print('---')

print('gamma:')
for i in range(n):
    y = normalized(x,alpha,beta,gamma+(gamma*i*1.05),eta_1,eta_2)
    print(gamma+(gamma*i*1.05))
    plt.plot(x,y)

plt.show()
print('---')

print('eta_1:')
for i in range(n):
    y = normalized(x,alpha,beta,gamma,eta_1+(eta_1*i*1.05),eta_2)
    print(eta_1+(eta_1*i*1.05))
    plt.plot(x,y)

plt.show()
print('---')

print('eta_2:')
for i in range(n):
    y = normalized(x,alpha,beta,gamma,eta_1,eta_2+(eta_2*i*1.05))
    print(eta_2+(eta_2*i*1.05))
    plt.plot(x,y)

plt.show()
print('---')