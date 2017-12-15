import numpy as np

from plotsettings import *

import matplotlib.pyplot as plt
import seaborn as sns

from numba import jit,float32,int32, njit, prange

import math

import time

from scipy.stats import linregress
from scipy import integrate

from progress.bar import Bar

try:
    import cPickle as pickle
except ImportError:
    import pickle



a = 1.92520559e+01
b = 1.40697462e-04
c = 1.66089035e+00
alpha = -1.70504796e+01
beta = 3.74249694e+03
gamma = 3.50332807e+00



def rot(alpha):
    return np.matrix( [[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]] )

def get_circle(r,n=12,inner_circle=False,centre_dot=False,dose_check_radius = 3):
    v = np.array( [r-dose_check_radius,0] )

    x = np.zeros(0)
    y = np.zeros(0)
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    if inner_circle:
        v = np.array( [(r-dose_check_radius)/2,0] )
        n = int(n/2)
        for i in range(n):
            x2, y2 = (v*rot(2*np.pi/n*i+2*np.pi/(2*n))).A1
            x = np.hstack( (x,x2) )
            y = np.hstack( (y,y2) )

    if centre_dot:
        x = np.hstack( (x,0) )
        y = np.hstack( (y,0) )


    v = np.array( [r,0] )
    cx = np.zeros(0)
    cy = np.zeros(0)
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        cx = np.hstack( (cx,x2) )
        cy = np.hstack( (cy,y2) )

    return x,y, cx, cy

def get_trimer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x = np.zeros(0)
    y = np.zeros(0)
    cx = np.zeros(0)
    cy = np.zeros(0)
    v = np.array( [0,0.5*(dist+2*r)/np.sin(np.pi/3)] )
    m = 3
    for i in range(m):
        x2, y2 = (v*rot(2*np.pi/m*i)).A1
        x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot, dose_check_radius)

        x = np.hstack( (x,x1+x2) )
        y = np.hstack( (y,y1+y2) )
        cx = np.hstack( (cx,cx1+x2) )
        cy = np.hstack( (cy,cy1+y2) )

    x += 500
    y += 500
    cx += 500
    cy += 500

    return x,y,cx,cy

def get_dimer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x2,y2,cx2,cy2 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x1 -= (r+dist/2)
    x2 += (r+dist/2)
    cx1 -= (r+dist/2)
    cx2 += (r+dist/2)

    x = np.concatenate((x1,x2))+500
    y = np.concatenate((y1,y2))+500
    cx = np.concatenate((cx1,cx2))+500
    cy = np.concatenate((cy1,cy2))+500
    # cy = np.concatenate((y1,y2))+500

    return x,y, cx, cy

def get_hexamer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x = np.zeros(0)
    y = np.zeros(0)
    cx = np.zeros(0)
    cy = np.zeros(0)
    v = np.array( [0.5*(dist+2*r)/np.sin(np.pi/6),0] )
    m = 6
    for i in range(m):
        x2, y2 = (v*rot(2*np.pi/m*i)).A1
        x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot)

        x = np.hstack( (x,x1+x2) )
        y = np.hstack( (y,y1+y2) )
        cx = np.hstack( (cx,cx1+x2) )
        cy = np.hstack( (cy,cy1+y2) )

    x += 500
    y += 500
    cx += 500
    cy += 500

    return x,y,cx,cy


def get_asymdimer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    r2 = 1.5*r
    x2,y2,cx2,cy2 = get_circle(r2,n,inner_circle, centre_dot,dose_check_radius)
    x1 -= r+dist/2
    x2 += r2+dist/2
    cx1 -= r+dist/2
    cx2 += r2+dist/2

    x = np.concatenate((x1,x2))+500
    y = np.concatenate((y1,y2))+500
    cx = np.concatenate((cx1,cx2))+500
    cy = np.concatenate((cy1,cy2))+500
    return x,y,cx,cy


def get_asymtrimer(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    r2 = 1.5*r
    x2,y2,cx2,cy2 = get_circle(r2,n,inner_circle, centre_dot,dose_check_radius)
    x1 += r+r2+dist
    cx1 += r+r2+dist
    #x2 += r2+dist/2

    r3 = 1.5*r2
    x3,y3,cx3,cy3 = get_circle(r3,n,inner_circle, centre_dot,dose_check_radius)
    x3 -= r2+r3+dist
    cx3 -= r2+r3+dist

    x = np.concatenate((x1,x2,x3))+500
    y = np.concatenate((y1,y2,y3))+500
    cx = np.concatenate((cx1,cx2,cx3))+500
    cy = np.concatenate((cy1,cy2,cy3))+500

    return x,y,cx,cy


def get_single(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1, y1, cx1, cy1 = get_circle(r, n, inner_circle, centre_dot,dose_check_radius)

    #if r >= 50:
    #    x1,y1 = get_circle(r,n=48,inner_circle=True,centre_dot=True)
    #else:
    #    x1, y1 = get_circle(r, n=32, inner_circle=False, centre_dot=True)

    x = x1+500
    y = y1+500
    cx = cx1+500
    cy = cy1+500

    return x,y,cx,cy



def get_triple(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x2,y2,cx2,cy2 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x3,y3,cx3,cy3 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x1 -= 2*r+dist
    x2 += 2*r+dist
    cx1 -= 2*r+dist
    cx2 += 2*r+dist

    x = np.concatenate((x1,x2,x3))+500
    y = np.concatenate((y1,y2,y3))+500
    cx = np.concatenate((cx1, cx2, cx3)) + 500
    cy = np.concatenate((cy1, cy2, cy3)) + 500

    return x,y,cx,cy



def get_triple_rotated(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3,alpha = 0):
    x1,y1,cx1,cy1 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x2,y2,cx2,cy2 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x3,y3,cx3,cy3 = get_circle(r,n,inner_circle, centre_dot,dose_check_radius)
    x1 -= 2*r+dist
    cx1 -= 2*r+dist

    #v = np.array([r - dose_check_radius, 0])
    v = np.array([2*r+dist, 0])
    x_rot, y_rot = (v * rot(alpha)).A1

    x2 += x_rot
    cx2 += x_rot
    y2 -= y_rot
    cy2 -= y_rot

    x = np.concatenate((x1,x2,x3))+500
    y = np.concatenate((y1,y2,y3))+500
    cx = np.concatenate((cx1, cx2, cx3)) + 500
    cy = np.concatenate((cy1, cy2, cy3)) + 500

    return x,y,cx,cy

def get_triple00(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    return get_triple_rotated(dist,r,n,inner_circle,centre_dot,dose_check_radius,0)

def get_triple30(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    return get_triple_rotated(dist,r,n,inner_circle,centre_dot,dose_check_radius,2*np.pi/12)

def get_triple60(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    return get_triple_rotated(dist,r,n,inner_circle,centre_dot,dose_check_radius,2*np.pi/6)

def get_triple90(dist,r,n, inner_circle=False, centre_dot = False,dose_check_radius = 3):
    return get_triple_rotated(dist,r,n,inner_circle,centre_dot,dose_check_radius,2*np.pi/4)



# def get_line(dist,r):
#     n = int(r/dist)
#     x = np.zeros(n)
#     y = np.zeros(n)
#     for i in range(n):
#         x[i] += i*dist
#
#     return x,y
#
# def get_hexgrid(dist,r,n):
#     x = np.zeros(0)
#     y = np.zeros(0)
#     x1 = np.zeros(0)
#     y1 = np.zeros(0)
#     x0, y0 = get_circle(r,9)
#     #v1 = np.array( [dist+2*r,0] )
#     #v2 = (v1*rot(2*np.pi/6)).A1
#     #v1 = (v1*rot(-2*np.pi/6)).A1
#     d = (dist+2*r)
#     a = np.sqrt(3)*d
#     v1 = a*np.array([np.sqrt(3)/2,1/2])
#     v2 = a*np.array([np.sqrt(3)/2,-1/2])
#
#     indices = np.arange(-50,50,1)
#
#     for i in indices:
#         for j in indices:
#             if np.sqrt(i**2+j**2) < 5:
#                 x = np.hstack( (x,x0+v1[0]*i+v2[0]*j) )
#                 y = np.hstack( (y,y0+v1[1]*i+v2[1]*j) )
#
#                 x1 = np.hstack( (x1,x0-v1[0]*i+v2[0]*j-d) )
#                 y1 = np.hstack( (y1,y0-v1[1]*i+v2[1]*j) )
#
#
#     x = np.hstack((x, x1))
#     y = np.hstack((y, y1))
#
#     x = x - np.min(x) + 500
#     y = y - np.min(y) + 500
#     return x,y

current = np.float32(100 * 1e-12) # A
dwell_time = np.float32(800 * 1e-9) #200 * 1e-9 # s
dose_check_radius = 5 # nm
target_dose = 600#70 # uC/cm^2


#outfilename = 'emre2.txt'
#outfilename = 'asymdimer.txt'
#outfilename = 'single.txt'
#outfilename = 'pillars_r'+str(radius[0])+'nm_dose'+str(target_dose)+'.txt'
#outfilename = 'test.txt'
#outfilename = 'hexgrid.txt'
#outfilename = 'anni_struct.txt'


# prefixes = ["pillar_dimer","pillar_trimer","pillar_hexamer","pillar_asymdimer","pillar_triple"]
# structures = [get_dimer,get_trimer,get_hexamer,get_asymdimer,get_triple]
# dists = [30]
# for i in range(30):
#     dists.append(dists[i]+2)
#for i in range(10):
#    dists.append(dists[i] + 5)

#prefixes = ["pillar_dimer","pillar_trimer","pillar_triple"]
#structures = [get_dimer,get_trimer,get_triple]
#dists = [10]
#for i in range(30):
#    dists.append(dists[i]+2)

# prefixes = ["pillar_dimer"]#,"pillar_trimer","pillar_hexamer"]#,"pillar_asymdimer","pillar_triple"]
# structures = [get_dimer]#,get_trimer,get_hexamer]#,get_asymdimer,get_triple]
# dists = [40]
# radius = 100
#prefixes = ["kreis"]
#structures = [get_single]#,get_trimer,get_hexamer,get_asymdimer,get_triple]
#dists = [1]

prefixes = ["pillar_single","pillar_dimer","pillar_trimer","pillar_hexamer","pillar_asymdimer","pillar_triple00","pillar_triple30","pillar_triple60","pillar_triple90"]
structures = [get_single,get_dimer,get_trimer,get_hexamer,get_asymdimer,get_triple00,get_triple30,get_triple60,get_triple90]

# prefixes = ["pillar_dimer"]
# structures = [get_dimer]
dists = [30]

# dists = [20]
# for i in range(30):
#     dists.append(dists[i]+2)
radius = [30]
n =      [32]
centre = [ 0]
inner =  [ 0]


#dists = [0]
#radius = [15,20,25,30,35,40,45,50,60,70,80,90,100,120,140,160,180,200]
#n =      [12,12,12,12,12,14,14,16,16,18,18,20, 22, 24, 26, 28, 30, 32]
#centre = [ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#inner =  [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


#outfilename = 'pillars_r'+str(radius[0])+'nm_dose'+str(target_dose)+'_800ns.txt'
outfilename = 'test.txt'



# normalization = 1
# #http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
# @jit(float32(float32),nopython=True)
# def calc_prox(r):
#     return (1/normalization) * (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*math.exp(-r**2/alpha**2) + (eta_1/beta**2)*math.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*math.exp(-math.sqrt(r/gamma)) )
# # [return] = C/nm !!!
# normalization = integrate.quad(lambda x: 2*np.pi*x*calc_prox(x), 0, np.inf)
# print('norm:'+str(normalization))
# #normalization = 2.41701729505915

normalization0 = 25311.230793201186
def calc_prox(r):
    return   (1/normalization0)*( a*np.exp(-r**2/alpha**2) + b*np.exp(-r**2/beta**2) + c*np.exp(-np.sqrt(r/gamma)) )
normalization = integrate.quad(lambda x: 2*np.pi*x*calc_prox(x), 0, np.inf)
print('norm:'+str(normalization))



@jit(float32(float32,float32,float32,float32),nopython=True)
def dist(x0,y0,x,y):
    return math.sqrt( (x0-x)*(x0-x)+(y0-y)*(y0-y) )


#@jit(float32[:](float32[:],float32[:],float32[:],float32[:],float32[:]),nopython=True)
@jit
def calc_map_2(x0,y0,doses,x,y):
    exposure = np.zeros(len(x),dtype=np.float32)
    pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
    for i in range(len(x)):
        for j in range(len(x0)):
            r= dist(x0[j],y0[j],x[i],y[i])
            exposure[i] += calc_prox(r)*doses[j]* pixel_area
    return exposure

@njit(float32[:](float32[:,:],float32[:]),parallel=False)
def calc_map(proximity,doses):
    exposure = np.zeros(proximity.shape[1],dtype=np.float32)
    for i in range(proximity.shape[1]):
        for j in range(proximity.shape[0]):
            exposure[i] += proximity[j,i]*doses[j]
    return exposure

@jit(float32[:, :](float32[:], float32[:]),nopython=True)
def recombine_arrays(arr1, arr2):
    res = np.zeros((len(arr1), 2), dtype=np.float32)
    res[:, 0] = arr1
    res[:, 1] = arr2
    n_crossover = int(len(arr1)/2)
    for i in range(n_crossover):
        k = np.random.randint(0, len(arr1) )
        alpha = np.random.random()
        #alpha = 0
        res[k, 0] = alpha * arr1[k] + (1 - alpha) * arr2[k]
        res[k, 1] = alpha * arr2[k] + (1 - alpha) * arr1[k]
    return res

# @jit(float32[:](float32[:],float32, float32),nopython=True)
# def mutate(arr,sigma,mutation_rate):
#     for i in range(arr.shape[0]):
#         if np.random.random() < mutation_rate:
#             mutation = np.random.normal()*sigma
#             if mutation > sigma*1.0:
#                 mutation = sigma
#             if mutation < -sigma*1.0:
#                 mutation = -sigma
#             arr[i] = arr[i] + mutation
#     return arr

@njit(float32[:](float32[:],float32, float32),parallel=True)
def mutate(arr,sigma,mutation_rate):
    for i in prange(arr.shape[0]):
        if np.random.random() < mutation_rate:
            mutation = (np.random.random()-0.5) * sigma
            arr[i] = arr[i] + mutation
    return arr

@jit(float32[:,:](float32[:,:]),nopython=True)
def reinit_population(population):
    n = population.shape[1]
    new_pop = np.zeros(population.shape,dtype=np.float32)
    mother = population[:,0]
    new_pop[:, 0] = mother
    sigma = np.mean(mother)*0.01
    i = 1
    while True:
        new_pop[:, i] = mother + (np.random.random(population.shape[0])-0.5)*sigma
        i += 1
        if i >= n:
             break

    return new_pop

@njit(float32[:](float32[:,:],float32[:,:]),parallel=False)
def calc_fitness(population,proximity):
    fitness = np.zeros(population.shape[1],dtype=np.float32)
    exposure = np.zeros(population.shape[1],dtype=np.float32)
    pixel_area =  np.float32(1) #nm^2 #pixel_area * 1e-14  # cm^2
    #repetitions = np.zeros(population.shape[0],dtype=np.float32)

    for p in range(population.shape[1]):
        #for i in range(population.shape[0]):
        #    repetitions[i] = round(population[i,p])
        exposure = calc_map(proximity[:,:],population[:, p] * current * dwell_time)
            #exposure = calc_map(proximity[:, j, :], repetitions * current * dwell_time)
        exposure = (exposure* np.float32(1e6))/(pixel_area*np.float32(1e-14) ) # uC/cm^2
        #sum = 0.0
        #for i in range(exposure.shape[0]):
        #    sum += math.fabs(target_dose-exposure[i])
        #fitness[p] = sum/exposure.shape[0]
        #fitness[p] = np.sum(np.abs(target_dose-exposure))/exposure.shape[0]
        fitness[p] = np.mean(np.abs(np.subtract(target_dose,exposure)))

    return fitness

# @jit(float32[:,:](float32[:,:]),nopython=True)
# def recombine_population(population):
#     #n_recombination = 6
#     n_recombination = int(population.shape[1]/3)
#     #n_recombination = int(population.shape[1]/2)
#
#     for i in range(int(n_recombination/2)):
#         k = 2*i
#         l = 2*i+1
#         r_rec = recombine_arrays(population[:, k],population[:, l])
#         population[:, -k] = r_rec[:, 0]
#         population[:, -l] = r_rec[:, 1]
#
#     return population

# @jit(float32[:,:](float32[:,:]),nopython=True)
# def recombine_population(population):
#     new_pop = np.zeros(population.shape,dtype=np.float32)
#     n = population.shape[1]
#
#     fit = np.arange(0,int(n/2))
#
#     i = 0
#
#     while True:
#         mother = np.random.randint(0,len(fit))
#         father = np.random.randint(0,len(fit))
#         r_rec = recombine_arrays(population[:, mother], population[:, father])
#         new_pop[:, i] =  r_rec[:, 0]
#         new_pop[:, i+1] = r_rec[:, 1]
#         i += 2
#         if i >= n:
#              break
#
#     return new_pop

@jit(int32(float32[:]),nopython=True)
def weighted_choice(weights):
    i = 0
    rnd = np.random.random() * np.sum(weights)
    for i in range(len(weights)):
        rnd -= weights[i]
        if rnd < 0:
            break
    return i

@jit(float32[:,:](float32[:,:],float32[:]),nopython=True)
def recombine_population(population,fitness):
    weights =  fitness.max()-fitness
    new_pop = np.zeros(population.shape,dtype=np.float32)
    n = population.shape[1]
    i = 0
    while True:
        father = weighted_choice(weights)
        mother = weighted_choice(weights)
        r_rec = recombine_arrays(population[:, mother], population[:, father])
        new_pop[:, i] =  r_rec[:, 0]
        new_pop[:, i+1] = r_rec[:, 1]
        i += 2
        if i >= n:
             break

    return new_pop

# @jit(float32[:,:](float32[:,:],float32, float32),nopython=True)
# def mutate_population(population,sigma,mutation_rate):
#
#     for i in range(population.shape[1]):
#         #if i < int(population.shape[1]/3):
#         if i < 2:
#             population[:, i] = mutate(population[:, i], sigma/10, mutation_rate)#
#         elif i < 6:
#             population[:, i] = mutate(population[:, i], sigma/2, mutation_rate)#
#         else:
#             population[:, i] = mutate(population[:, i], sigma, mutation_rate)  #
#     return population

@njit(float32[:,:](float32[:,:],float32, float32),parallel=True)
def mutate_population(population,sigma,mutation_rate):
    for i in range(int(population.shape[1]/2)):
        mutate(population[:, np.random.randint(0, population.shape[1])], sigma, mutation_rate)

    return population

@jit(float32[:,:](float32[:,:]),nopython=True)
def check_limits(population):
    for i in range(population.shape[1]):
        for j in range(population.shape[0]):
            if population[j, i] < 0.1:
                population[j, i] = 0
    return population


population_size = 100
#max_iter = 200000
max_iter = 500000
#max_iter = 50000

#@jit()#(float32(float32[:],float32[:],float32[:],float32[:],float32[:],float32[:]))
def iterate(x0,y0,repetitions,target):
    mutation_rate = 0.3
    logpoints = np.arange(500,max_iter,500)
    #logpoints = np.array([max_iter+1])
    checkpoints = np.arange(50,max_iter,50)

    population = np.zeros((len(x0),population_size),dtype=np.float32)
    fitness = np.zeros(population_size,dtype=np.float32)
    pixel_area =  1 #nm^2 #pixel_area * 1e-14  # cm^2


    proximity = np.zeros((population.shape[0],target.shape[0]),dtype=np.float32)
    convergence = np.zeros(max_iter)
    t = np.zeros(max_iter)

    i=0
    j=0
    for i in range(population.shape[0]):
        for j in range(target.shape[0]):
            proximity[i,j] = calc_prox(dist(x0[i],y0[i],target[j,0],target[j,1]))

    start = np.linspace(0,100,num=population_size)
    for i in range(population_size):
        #population[:, i] = repetitions + np.random.randint(-50, 50)
        population[:, i] = np.repeat(start[i],len(repetitions))
        for j in range(len(repetitions)):
            population[j, i] = population[j, i]+np.random.randint(-5,5)
            if population[j, i] < 1:
                population[j, i] = 1

    #print("Starting Iteration")
    sigma = 1
    std_err = 0.0
    slope = 0.0
    starttime = time.time()
    for i in range(max_iter):
        fitness = calc_fitness(population, proximity)
        sorted_ind = np.argsort(fitness)
        #sorted_ind = argsort1D(fitness)
        fitness = fitness[sorted_ind]
        population = population[:,sorted_ind]

        #sigma = fitness[0]/5

        if 100*fitness[0]/target_dose < 0.001:#0.01:
            break

        if i < 2000:
            sigma = 2#1
        elif i < 4000:
            sigma = 1#0.5
        else:
            if i in checkpoints:
                indices = np.arange(i - 50, i, step=1)
                slope, intercept, r_value, p_value, std_err = linregress(indices, convergence[indices])
                std_err = np.abs((np.std(convergence[indices]) * 100))/ np.mean(convergence[indices])
                #slope = slope / np.mean(convergence[indices])

                if i in checkpoints:

                    # if (std_err > 0.1) or (slope > 0):
                    # if (slope > 0):
                    if ((std_err > 0.1) and (slope < -1)) or (std_err > 0.05 and slope > -0.1):
                        #if sigma > 10:
                            sigma *= 0.995

                    if (std_err < 0.1) and (slope > 0) and (sigma < 1 / 2):
                        sigma *= 1.01
                        # sigma += sigma_start/20

                # if not i % 10000 and i > 10000:
                #     population = reinit_population(population)
                #     # sigma = np.mean(population[:,0]) / 10

        population = recombine_population(population,fitness)
        population = mutate_population(population,sigma,mutation_rate)
        population = check_limits(population)
        if i in logpoints:
            print("{0:7d}: fitness: {1:2.6f}, sigma: {2:5.5f}, std_err: {3:5.5f}, slope: {4:5.5f}".format(i, fitness[0], sigma,std_err,slope))
        convergence[i] = fitness[0]
        t[i] = time.time() - starttime

    print("Done -> Mean Error: {0:1.5f}%, sigma: {1:1.5f}".format(convergence[:i][-1] , sigma))

    return population[:,0], t[:i], convergence[:i]




Outputfile = open(outfilename,'w')

n_total = len(radius) * len(structures) * len(dists)

bar = Bar('Status', max=n_total)

for r in range(len(radius)):
    for l in range(len(structures)):
        for k in range(len(dists)):
            at = (r+1)*(l+1)*(k+1)
            print("structure #: {0:d}/{1:d} {4:s} ,dist #: {2:d}/{3:d}, iter #: {6:d}/{5:d}".format(l + 1, len(structures), k + 1, len(dists), prefixes[l], n_total, at))

            #Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11500, 11500, 5, 5" + '\n')
            #Outputfile.write('I 1' + '\n')
            #Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
            #Outputfile.write("FSIZE 20 micrometer" + '\n')
            #Outputfile.write("UNIT 1 micrometer" + '\n')

            Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11000, 11000, 5, 5" + '\n')
            #Outputfile.write('D ' + prefixes[l] + '-' + str(r) + ", 11000, 11000, 5, 5" + '\n')
            Outputfile.write('I 1' + '\n')
            Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
            Outputfile.write("FSIZE 15 micrometer" + '\n')
            Outputfile.write("UNIT 1 micrometer" + '\n')

            if isinstance(n, list) and isinstance(inner, list) and isinstance(centre, list):
                x0, y0, cx, cy = structures[l](dists[k], radius[r], n[r],inner[r],centre[r],dose_check_radius)
                # if radius[r] < 100:
                #     x0, y0 = structures[l](dists[k] + dose_check_radius * 2, radius[r] - dose_check_radius, n[r],inner[r],centre[r])
                #     cx, cy = structures[l](dists[k], radius[r], n[r])
                # else:
                #     x0, y0 = structures[l](dists[k],radius[r]*3/4, n[r], False, False)
                #     x01, y01 = structures[l](dists[k], radius[r] * 1 / 4, int(n[r]/2), False, False)
                #     x0 = np.concatenate((x0,x01))
                #     y0 = np.concatenate((y0, y01))
                #     cx, cy = structures[l](dists[k], radius[r], n[r])
                #     cx2, cy2 = structures[l](dists[k], int(radius[r]*2/4), int(n[r]*2/3))
                #     #cx3, cy3 = structures[l](dists[k], int(radius[r]*1/5), int(n[r]*1/3))
                #     cx3, cy3 = (np.array([500]),np.array([500]))
                #     #cx = np.concatenate((cx,cx2,cx3))
                #     #cy = np.concatenate((cy, cy2,cy3))
                #     cx = np.concatenate((cx,cx3))
                #     cy = np.concatenate((cy,cy3))
            else:
                x0,y0 = structures[l](dists[k]+dose_check_radius*2,radius[r]-dose_check_radius,n)
                cx, cy = structures[l](dists[k],radius[r],n)

            #plt.scatter(x0,y0)
            #plt.show()
            #plt.close()

            repetitions = np.ones(len(x0),dtype=np.float32)*300

            target = np.zeros((len(cx),2),dtype=np.float32)
            target[:,0] = cx
            target[:,1] = cy

            repetitions, t, convergence = iterate(x0, y0, repetitions,target)

            repetitions[np.where(repetitions < 1)] = 0
            repetitions = np.array(np.round(repetitions),dtype=np.int)

            x0 = x0 -500
            y0 = y0 -500
            target = target - 500

            #x = np.arange(np.min(x0)-50,np.max(x0)+50,step=0.2)
            #y = np.arange(np.min(y0)-50,np.max(y0)+50,step=0.2)
            x = np.arange(np.min(x0)-50,np.max(x0)+50,step=1)
            y = np.arange(np.min(y0)-50,np.max(y0)+50,step=1)
            x, y = np.meshgrid(x, y)
            orig_shape = x.shape
            x = x.ravel()
            y = y.ravel()
            exposure = calc_map_2(x0, y0, repetitions * dwell_time * current, x, y) # C
            exposure = exposure.reshape(orig_shape)
            exposure = exposure * 1e6 # uC
            pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
            pixel_area = pixel_area * 1e-14  # cm^2
            exposure = exposure/pixel_area # uC/cm^2

            name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+".png"
            name_pgf = "pics/" + prefixes[l] + "_" + str(dists[k]) + "_" + str(radius[r]) + ".pgf"
            #fig = newfig(0.9)
            cmap = sns.cubehelix_palette(light=1, as_cmap=True,reverse=False)
            plot = plt.imshow(np.flipud(exposure),cmap=cmap,extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
            cb = plt.colorbar()
            #cb.set_label('Dosis / uC/cm^2 ')
            cb.set_label(r'$Dosis\, / \, \frac{\mu C}{cm^2} ')
            plt.contour(x.reshape(orig_shape), y.reshape(orig_shape), exposure, [target_dose])#[290,300, 310])
            plt.xlabel(r'$x\,  /\,  nm')
            plt.ylabel(r'$y\,  /\,  nm')
            plt.tight_layout(.5)
            plt.savefig(name,dpi=200)
            #plt.savefig(name_pgf)
            plt.close()

            #x = np.arange(np.min(x0)-50,np.max(x0)+50,step=0.2)
            #y = np.arange(np.min(y0)-50,np.max(y0)+50,step=0.2)
            x = np.arange(np.min(x0)-50,np.max(x0)+50,step=1)
            y = np.arange(np.min(y0)-50,np.max(y0)+50,step=1)
            x, y = np.meshgrid(x, y)
            orig_shape = x.shape
            x = x.ravel()
            y = y.ravel()
            exposure = calc_map_2(x0, y0, repetitions * dwell_time * current, x, y) # C
            exposure = exposure.reshape(orig_shape)
            exposure = exposure * 1e6 # uC
            pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
            pixel_area = pixel_area * 1e-14  # cm^2
            exposure = exposure/pixel_area # uC/cm^2

            name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_expected.png"
            name_pgf = name[:-4]+".pgf"
            #fig = newfig(0.9)
            plot = plt.imshow(np.flipud(exposure >= target_dose),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
            plt.scatter(x0,y0,c="blue")
            plt.scatter(target[:,0].ravel(), target[:,1].ravel(), c="red")
            plt.axes().set_aspect('equal')
            plt.xlabel(r'$x\,  /\,  nm')
            plt.ylabel(r'$y\,  /\,  nm')
            plt.tight_layout()
            plt.savefig(name,dpi=200)
            #plt.savefig(name_pgf)
            plt.close()

            name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_convergence.png"
            name_pgf = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_convergence.pgf"
            #fig = newfig(0.9)
            print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
            plt.semilogy(t,convergence)
            plt.xlabel('time / s')
            plt.ylabel('Mean Error')
            plt.tight_layout()
            plt.savefig(name,dpi=200)
            #plt.savefig(name_pgf)
            plt.close()

            area = np.pi * (15*repetitions/np.max(repetitions))**2
            #area = np.pi * ( 0.005*(np.max(y)-np.min(y)) * repetitions / np.max(repetitions)) ** 2
            #fig = newfig(0.9)
            plt.scatter(x0, y0, s=area, alpha=0.5,edgecolors="black",linewidths=1)
            plt.axes().set_aspect('equal', 'datalim')
            name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_scatter.png"
            name_pgf = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_scatter.pgf"
            plt.xlabel(r'$x\,  /\,  nm')
            plt.ylabel(r'$y\,  /\,  nm')
            plt.tight_layout()
            plt.savefig(name,dpi=200)
            #plt.savefig(name_pgf)
            plt.close()
            x0 = x0/1000+0.5
            y0 = y0/1000+0.5
            print(repetitions)
            for j in range(len(x0)):
                if repetitions[j] >= 1:
                    Outputfile.write('RDOT '+str(x0[j]) + ', ' + str(y0[j]) + ', ' + str((repetitions[j])) + '\n')
            Outputfile.write('END' + '\n')
            Outputfile.write('\n')
            Outputfile.write('\n')

            bar.next()
            print("ETA: " + str(bar.eta_td))

bar.finish()
Outputfile.close()

