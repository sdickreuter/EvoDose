import numpy as np

#from plotsettings import *

import matplotlib.pyplot as plt
import seaborn as sns

from numba import jit,float64,int64

import math

import time

from scipy.stats import linregress
from scipy import integrate
from scipy.optimize import minimize, curve_fit, basinhopping



# alpha = 50 #nm
# beta = 5000 #nm
# gamma = 20 #nm
# eta_1 = 1
# eta_2 = 0.05


# normalization = 1.0
# #http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
# #@jit(float64(float64),nopython=True)
# def calc_prox(r,alpha,beta,gamma,eta_1,eta_2):
#     #return (1.0/normalization) * (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*math.exp(-r**2/alpha**2) + (eta_1/beta**2)*math.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*math.exp(-math.sqrt(r/gamma)) )
#     #return  (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*np.exp(-r**2/alpha**2) + (eta_1/beta**2)*np.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*np.exp(-np.sqrt(r/gamma)) )
#     return   ( (1/(alpha**2))*np.exp(-r**2/alpha**2) + (eta_1/beta**2)*np.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*np.exp(-np.sqrt(r/gamma)) )
#
# # [return] = C/nm !!!
# #normalization = integrate.quad(lambda x: 2*np.pi*x*calc_prox(x,alpha,beta,gamma,eta_1,eta_2), 0, np.inf)
# #print('norm:'+str(normalization))

def calc_prox(r,a,b,c,alpha,beta,gamma):
    #return (1.0/normalization) * (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*math.exp(-r**2/alpha**2) + (eta_1/beta**2)*math.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*math.exp(-math.sqrt(r/gamma)) )
    #return  (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*np.exp(-r**2/alpha**2) + (eta_1/beta**2)*np.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*np.exp(-np.sqrt(r/gamma)) )
    return   ( a*np.exp(-r**2/alpha**2) + b*np.exp(-r**2/beta**2) + c*np.exp(-np.sqrt(r/gamma)) )

dat = np.loadtxt('Messung17_11_2017_update.csv',delimiter=';',skiprows=1)
x_m = dat[:,4]
y_m = dat[:,1]
#y_m = y_m/y_m.max()
#y_m = 1/y_m#y_m.max()

print(x_m.shape)

plt.loglog(x_m,y_m)
plt.show()

#x_m = np.hstack((x_m,np.arange(1,15,1)))
#y_m = np.hstack((y_m,np.repeat(y_m.max(),14)))

#x_m = np.hstack((x_m,[10000,15000]))
#y_m = np.hstack((y_m,[1e-11,5e-18]))


# #def calc_prox(r,alpha,beta,gamma,eta_1,eta_2):
# fit_fun = lambda x, alpha,beta,gamma,eta_1,eta_2, a,c : np.log(a* (calc_prox(x,alpha,beta,gamma,eta_1,eta_2)) +c)
# plot_fun = lambda x, alpha,beta,gamma,eta_1,eta_2, a,c : a* (calc_prox(x,alpha,beta,gamma,eta_1,eta_2)) +c
# #fit_fun = lambda x, alpha,beta,gamma,eta_1,eta_2, a,c : a* (calc_prox(x,alpha,beta,gamma,eta_1,eta_2)) +c
#
# p0 = [20,2000,5,0.1,0.1,30,0]
# upper = (1e12,1e12,1e12,10,10,1e12,1e12)
# lower = (0,0,0,0,0,0,0)
# bounds = [lower, upper]
# popt, pcov = curve_fit(fit_fun, x_m, y_m, p0,bounds=bounds,**{'method':'trf','xtol':1e-30,'gtol':1e-20,'ftol':1e-20,'verbose':2,'loss':'arctan'})

fit_fun = lambda x, a,b,c,alpha,beta,gamma : (calc_prox(x,a,b,c,alpha,beta,gamma))

#p0 = [20,2000,5,0.1,0.1,30,0]
p0 = [1,0.01,0.1,10,100,10,1,0]
p0 = [  3.0e01,   1.0e-03,   1.0e00,   1.0e01,  3.7e03,   3.5e00]

#p0 = [  1.3e01,   2.0e02,   3.0e04,   1.3e01,  -5.4e01,   1.6e02,   1.4e01,   7.0e06]
upper = (1e12,1e12,1e12,1e12,1e12,1e12)
lower = (0,0,0,0,0,0)
bnds = []
for j in range(len(upper)):
    bnds.append((lower[j], upper[j]))

def err_fun(p):
    fit = fit_fun(x_m, *p)
    diff = (np.abs(np.log(y_m) - np.log(fit)) )**2
    return np.sum(diff)

minimizer_kwargs = {"method": "L-BFGS-B", "tol": 1e-12,"bounds": bnds}
#res = basinhopping(err_fun, p0, minimizer_kwargs=minimizer_kwargs, niter=200,disp=True)
#res = minimize(err_fun, p0, method='SLSQP', tol=1e-12, bounds=bnds, options={'disp': True, 'maxiter': 500})
#res = minimize(err_fun, p0, method='L-BFGS-B', jac=False, options={'disp': True, 'maxiter': 5000})
res = minimize(err_fun, p0, method='nelder-mead', options={'xtol': 1e-20, 'disp': True, 'maxiter': 50000})
popt = res.x


print(popt)

#x = np.linspace(x_m.min(),x_m.max(),500,dtype=np.float64)
x = np.linspace(1,x_m.max(),2000,dtype=np.float64)

y_fit = fit_fun(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
y = fit_fun(x,p0[0],p0[1],p0[2],p0[3],p0[4],p0[5])


name = "proximity.png"
name_pgf = name[:-4] + ".pgf"
#fig = newfig(0.9)
fig = plt.figure()

plt.loglog(x,y)
plt.loglog(x,y_fit)
plt.loglog(x_m,y_m,linewidth=0,marker='x')

#plt.plot(x,y)
#plt.plot(x_m,y_m,linewidth=0,marker='x')

plt.xlim((0,x_m.max()))
plt.xlabel('r / nm')
plt.ylabel('1/Dosis [a.u.]')
plt.tight_layout(.5)
#plt.savefig(name,dpi=300)
#plt.savefig(name_pgf)
#plt.close()
plt.show()

fig = plt.figure()

plt.plot(x,y_fit)
plt.plot(x_m,y_m,linewidth=0,marker='x')

plt.xlim((0,x_m.max()))
plt.xlabel('r / nm')
plt.ylabel('1/Dosis [a.u.]')
plt.tight_layout(.5)
#plt.savefig(name,dpi=300)
#plt.savefig(name_pgf)
#plt.close()
plt.show()

# #def calc_prox(r,a,b,c,alpha,beta,gamma):
# normalization0 = 17971.512008579837
# normalization = integrate.quad(lambda x: 2*np.pi*x*(1/normalization0)*calc_prox(x,popt[0],popt[1],popt[2],popt[3],popt[5],popt[6]), 0, np.inf)
# print('norm:'+str(normalization))


# a = 1.92520559e+01
# b = 1.40697462e-04
# c = 1.66089035e+00
# alpha = -1.70504796e+01
# beta = 3.74249694e+03
# gamma = 3.50332807e+00

a = 3.33818554e+01
b = 2.35068751e-04
c = 2.73254430e+00
alpha = 1.71120218e+01
beta = 3.94207696e+03
gamma = 3.59873819e+00

normalization0 = 44853.19174033405
def calc_prox(r):
    return   (1/normalization0)*( a*np.exp(-r**2/alpha**2) + b*np.exp(-r**2/beta**2) + c*np.exp(-np.sqrt(r/gamma)) )
normalization = integrate.quad(lambda x: 2*np.pi*x*calc_prox(x), 0, np.inf)
print('norm:'+str(normalization))

fig = plt.figure()
plt.loglog(x,(1/normalization0)*calc_prox(x))
plt.tight_layout(.5)
#plt.savefig(name,dpi=300)
#plt.savefig(name_pgf)
#plt.close()
plt.show()


# def double_gauss(r,alpha,beta,eta):
#     #return (1.0/normalization) * (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*math.exp(-r**2/alpha**2) + (eta_1/beta**2)*math.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*math.exp(-math.sqrt(r/gamma)) )
#     return  ( (1/(alpha**2))*np.exp(-r**2/alpha**2) + (eta/beta**2)*np.exp(-r**2/beta**2)  )
# # [return] = C/nm !!!
#
#
# dat = np.loadtxt('Messung20_09_2017.csv',delimiter=';',skiprows=1)
# x_m = dat[:,3]
# y_m = dat[:,1]
# y_m /= y_m.max()
#
#
# #def calc_prox(r,alpha,beta,gamma,eta_1,eta_2):
# fit_fun = lambda x, alpha, beta, eta, a : a* double_gauss(x,alpha,beta,eta)
#
# p0 = [100,100,1,5e3]
# upper = (1e12,1e12,10,1e12)
# lower = (1,1,0,0)
# bounds = [lower, upper]
#
# popt, pcov = curve_fit(fit_fun, x_m, y_m, p0,bounds=bounds,**{'xtol':1e-15,'gtol':1e-15,'ftol':1e-15})
#
# print(p0)
# print(popt)
#
# #x = np.linspace(x_m.min(),x_m.max(),500,dtype=np.float64)
# x = np.linspace(1,x_m.max(),500,dtype=np.float64)
#
# y = fit_fun(x,popt[0],popt[1],popt[2],popt[3])
# #y = fit_fun(x,p0[0],p0[1],p0[2],p0[3],p0[4])
#
#
# name = "proximity.png"
# name_pgf = name[:-4] + ".pgf"
# #fig = newfig(0.9)
# fig = plt.figure()
# plt.loglog(x,y)
# plt.loglog(x_m,y_m,linewidth=0,marker='x')
#
# # plt.plot(x,y)
# # plt.plot(x_m,y_m,linewidth=0,marker='x')
#
# plt.xlabel('r / nm')
# plt.ylabel('1/Dosis [a.u.]')
# plt.tight_layout(.5)
# #plt.savefig(name,dpi=300)
# #plt.savefig(name_pgf)
# #plt.close()
# plt.show()
