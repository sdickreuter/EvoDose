import numpy as np

#from plotsettings import *

import matplotlib.pyplot as plt
#import seaborn as sns

from numba import jit,float32,int32, void, njit, prange

import math

import time

from scipy.stats import linregress
from scipy import integrate
from scipy.signal import convolve,convolve2d
from scipy import linalg
from drawnow import drawnow, figure
from skimage.morphology import disk
from skimage.filters import rank
from scipy import ndimage, special

try:
   import cPickle as pickle
except:
   import pickle

alpha = 14*2 # nm
beta = 2810 # nm
eta = 0.92

current = 100 * 1e-12 # A
dwell_time = 800 * 1e-9#200 * 1e-9 # s
target_dose = 600#70 # uC/cm^2
nmperpixel = 10 # nm

def put_circle(x0,y0,r,exposure_indices=None):
    r = int(r/nmperpixel) # pixel
    x0 = int(x0/nmperpixel) # pixel
    y0 = int(y0/nmperpixel) # pixel
    x = np.linspace(-r,+r,2*r+1,dtype = np.int32)
    y = np.linspace(-r,+r,2*r+1,dtype = np.int32)
    x,y = np.meshgrid(x,y)
    x = x.ravel()+x0
    y = y.ravel()+y0

    if exposure_indices is None:
        exposure_indices = np.empty( shape=(0, 2) , dtype=np.int32)

    for i in range(x.shape[0]):
        if np.sqrt((x[i]-x0)**2+(y[i]-y0)**2) < r:
                exposure_indices = np.vstack((exposure_indices,np.array([x[i],y[i]],dtype=np.int32)))

    return exposure_indices



outfilename = 'test.txt'



@jit(float32(float32,float32,float32,float32),nopython=True)
def dist(x0,y0,x,y):
    return math.sqrt( (x0-x)*(x0-x)+(y0-y)*(y0-y) )

@jit(void(float32[:,:],int32[:,:],float32[:]),nopython=True,parallel= True)
def set_doses_field(field, exposure_indices, doses):
    for i in prange(doses.shape[0]):
        field[exposure_indices[i,0],exposure_indices[i,1]] = doses[i]

@jit(void(float32[:,:],int32[:,:],float32),nopython=True)
def set_target(target, exposure_indices, dose):
    for i in range(exposure_indices.shape[0]):
        target[exposure_indices[i,0],exposure_indices[i,1]] = dose

@njit(void(float32[:,:],float32[:,:],float32[:],float32[:]),parallel=True)
def convolve_with_vector(field,exposure,v,h):
    buf = np.zeros(field.shape,dtype=np.float32)

    for j in prange(field.shape[1]):
        for i in range(field.shape[0]):
            if field[i,j] > 0:
                for k in range(v.shape[0]):
                    fi = i+k-int((v.shape[0]-1)/2)
                    if fi >= 0 and fi < field.shape[0]:
                        buf[fi,j] += field[i,j]*v[k]

    for i in prange(field.shape[0]):
        for j in range(field.shape[1]):
            if buf[i,j] < 0:
                for k in range(h.shape[0]):
                    fj = j+k-int((h.shape[0]-1)/2)
                    if fj >= 0 and fj < field.shape[1]:
                        exposure[i,fj] += buf[i,j]*h[k]

@jit(float32[:,:](float32[:,:],float32[:],float32[:],float32[:],float32[:]),nopython=True)
def calc_exposure(field,v_alpha,h_alpha,v_beta,h_beta):
    exposure = np.zeros(field.shape,dtype=np.float32)
    convolve_with_vector(field, exposure, v_alpha, h_alpha)
    convolve_with_vector(field, exposure, v_beta, h_beta)
    return exposure

@jit(float32(float32,float32), nopython=True)
def calc_gauß_alpha(r,alpha):
    # return 1/(np.pi*(1+eta)) * (1/alpha**2)*math.exp(-r ** 2 / alpha ** 2)
    return (1 / alpha ** 2) * math.exp(-r ** 2 / alpha ** 2)

@jit(float32(float32,float32,float32), nopython=True)
def calc_gauß_beta(r,beta,eta):
    return 1 / (np.pi * (1 + eta)) * (eta / beta) * math.exp(-r ** 2 * eta / alpha ** 2)

@jit(float32[:](float32[:], float32[:], float32[:], float32[:], float32), nopython=True)
def calc_map_alpha(x0, y0, x, y, alpha):
    gauss = np.zeros(len(x), dtype=np.float32)
    for i in range(len(x)):
        for j in range(len(x0)):
            gauss[i] += calc_gauß_alpha(dist(x0[j], y0[j], x[i], y[i]),alpha)
    return gauss

@jit(float32[:](float32[:], float32[:], float32[:], float32[:], float32, float32), nopython=True)
def calc_map_beta(x0, y0, x, y, beta, eta):
    gauss = np.zeros(len(x), dtype=np.float32)
    for i in range(len(x)):
        for j in range(len(x0)):
            gauss[i] += calc_gauß_beta(dist(x0[j], y0[j], x[i], y[i]), beta, eta)
    return gauss

#@jit()
def generate_hv_vectors(alpha,beta,eta):
    #width = 5000 # nm
    width = 2000  # nm
    size = int(np.round(width/nmperpixel/2))
    size += size%2 # make sure size is even !
    x_psf = np.linspace(-size,size,size+1,dtype=np.float32)*nmperpixel
    y_psf = np.linspace(-size,size,size+1,dtype=np.float32)*nmperpixel

    x_psf, y_psf = np.meshgrid(x_psf, y_psf)

    gauß_alpha = calc_map_alpha(np.array([0.0],dtype=np.float32),np.array([0.0],dtype=np.float32),x_psf.ravel(),y_psf.ravel(),alpha)
    gauß_alpha = gauß_alpha.reshape(x_psf.shape)
    gauß_beta = calc_map_beta(np.array([0.0],dtype=np.float32),np.array([0.0],dtype=np.float32),x_psf.ravel(),y_psf.ravel(),beta,eta)
    gauß_beta = gauß_beta.reshape(x_psf.shape)

    U_alpha, S_alpha, V_alpha = np.linalg.svd(gauß_alpha, full_matrices=True)
    U_beta, S_beta, V_beta = np.linalg.svd(gauß_beta, full_matrices=True)

    v_alpha = U_alpha[:,0] * np.sqrt(S_alpha[0])
    v_alpha = np.reshape(v_alpha,(v_alpha.shape[0],1))
    h_alpha = V_alpha[0,:] * np.sqrt(S_alpha[0])
    h_alpha = np.reshape(h_alpha,(1,h_alpha.shape[0]))

    v_beta = U_beta[:,0] * np.sqrt(S_beta[0])
    v_beta = np.reshape(v_beta,(v_beta.shape[0],1))
    h_beta = V_beta[0,:] * np.sqrt(S_beta[0])
    h_beta = np.reshape(h_beta,(1,h_beta.shape[0]))

    return np.array(v_alpha.ravel(),dtype=np.float32), np.array(h_alpha.ravel(),dtype=np.float32), np.array(v_beta.ravel(),dtype=np.float32), np.array(h_beta.ravel(),dtype=np.float32)

def generate_empty_field_matrix(width):
    size = int(np.round(width/nmperpixel))
    return np.zeros((size, size),dtype=np.float32)

#---------------- EVO Stuff -----------------------------------------

@njit(float32[:, :](float32[:], float32[:]),parallel=True)
def recombine_arrays(arr1, arr2):
    res = np.zeros((len(arr1), 2), dtype=np.float32)
    res[:, 0] = arr1
    res[:, 1] = arr2
    n_crossover = int(len(arr1)/2)
    for i in prange(n_crossover):
        k = np.random.randint(0, len(arr1))
        alpha = np.random.random()
        #alpha = 1/3
        res[k, 0] = alpha * arr1[k] + (1 - alpha) * arr2[k]
        res[k, 1] = alpha * arr2[k] + (1 - alpha) * arr1[k]
    return res

# @jit(float32[:, :](float32[:], float32[:]),nopython=True)
# def recombine_arrays(arr1, arr2):
#     res = np.zeros((len(arr1), 2), dtype=np.float32)
#     res[:, 0] = arr1
#     res[:, 1] = arr2
#     for i in range(len(arr1)):
#         alpha = 1/3
#         #alpha = np.random.random()
#         res[i, 0] = alpha * arr1[i] + (1 - alpha) * arr2[i]
#         res[i, 1] = alpha * arr2[i] + (1 - alpha) * arr1[i]
#     return res


@njit(float32[:](float32[:],float32, float32),parallel=True)
def mutate(arr,sigma,mutation_rate):
    for i in prange(arr.shape[0]):
        if np.random.random() < mutation_rate:
            #mutation = np.random.normal()*sigma
            #if mutation > sigma*1.0:
            #    mutation = sigma
            #if mutation < -sigma*1.0:
            #    mutation = -sigma
            mutation = (np.random.random()-0.5) * sigma
            arr[i] = arr[i] + mutation
    return arr


@njit(float32[:](float32[:,:],int32[:,:],float32[:,:],float32[:],float32[:],float32[:],float32[:]),parallel=True)
def calc_fitness(population, exposure_indices, target, v_alpha, h_alpha, v_beta, h_beta):
    fitness = np.zeros(population.shape[1],dtype=np.float32)
    exposure = np.zeros(target.shape,dtype=np.float32)
    field = np.zeros(target.shape, dtype=np.float32)
    for p in range(population.shape[1]):
        field *= 0
        set_doses_field(field, exposure_indices, population[:, p])
        exposure = calc_exposure(field, v_alpha, h_alpha, v_beta, h_beta)
        #fitness[p] = np.sum(np.abs(np.subtract(target,exposure)))#/exposure_indices.shape[0]
        buf = 0
        for i in prange(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i,j] > 0:
                    buf += np.abs(exposure[i,j]-target[i,j])
        fitness[p] = fitness[p] + buf
        #fitness[p] += np.sum(population[:,p]>np.var(population[:,p]))#/exposure_indices.shape[0]
        # buf = 0
        # for i in range(field.shape[0]):
        #     for j in range(field.shape[1]):
        #         if field[i,j] > 0:
        #             neighbors = np.array([field[i+1,j],field[i-1,j],field[i,j+1],field[i,j-1],field[i+1,j+1],field[i+1,j-1],field[i-1,j+1],field[i-1,j-1]],dtype=np.float32)
        #             notzero = np.sum(neighbors>0)+1
        #             buf += np.abs(field[i,j]- (field[i,j] +field[i+1,j]+field[i,j+1]+field[i-1,j]+field[i,j-1]+field[i+1,j+1]+field[i+1,j-1]+field[i-1,j+1]+field[i-1,j-1])/notzero)/100
        # fitness[p] = fitness[p] + buf

    fitness /= exposure_indices.shape[0]
    return fitness


# @jit(float32[:,:](float32[:,:]),nopython=True)
# def recombine_population(population):
#     #n_recombination = 6
#     #n_recombination = int(population.shape[1]/3)
#     #n_recombination = int(population.shape[1]*3/4)
#     n_recombination = int(population.shape[1])
#
#     for i in range(int(n_recombination/2)):
#         k = 2*i
#         l = 2*i+1
#         r_rec = recombine_arrays(population[:, k],population[:, l])
#         population[:, -k] = r_rec[:, 0]
#         population[:, -l] = r_rec[:, 1]
#
#     return population

@jit(float32[:,:](float32[:,:]),nopython=True)
def recombine_population(population):
    new_pop = np.zeros(population.shape,dtype=np.float32)
    n = population.shape[1]

    fit = np.arange(0,int(n/2))

    i = 0

    while True:
        mother = np.random.randint(0,len(fit))
        father = np.random.randint(0,len(fit))
        r_rec = recombine_arrays(population[:, mother], population[:, father])
        new_pop[:, i] =  r_rec[:, 0]
        new_pop[:, i+1] = r_rec[:, 1]
        i += 2
        if i >= n:
             break

    return new_pop

@njit(float32[:,:](float32[:,:],float32, float32),parallel=True)
def mutate_population(population,sigma,mutation_rate):

    for i in range(population.shape[1]):
        #if i < int(population.shape[1]/3):
        if i < 2:
            population[:, i] = mutate(population[:, i], sigma/10, mutation_rate)#
        elif i < 4:
            population[:, i] = mutate(population[:, i], sigma/2, mutation_rate)#
        else:
            population[:, i] = mutate(population[:, i], sigma, mutation_rate)  #
        # mutate(population[:, i], sigma, mutation_rate)
    return population

@njit(float32[:,:](float32[:,:]),parallel=True)
def check_limits(population):
    for i in prange(population.shape[1]):
        for j in range(population.shape[0]):
            if population[j, i] < 1:
                population[j, i] = 0
    return population


#@jit(void(float32[:,:],float32[:,:],float32[:,:]),nopython=True)
@jit
def smooth_doses(target,population,exposure_indices):
    field = np.zeros(target.shape, dtype=np.float32)

    for p in range(population.shape[1]):
        set_doses_field(field, exposure_indices, population[:, p])
        field = ndimage.median_filter(field, footprint=disk(1), mode="mirror")
        for i in range(exposure_indices.shape[0]):
            population[i,p] = field[exposure_indices[i, 0], exposure_indices[i, 1]]

    return population

population_size = 150
max_iter = 3000

#@jit(float32(float32[:],float32[:],float32[:],float32[:],float32[:],float32[:]))
@jit()
def iterate(exposure_indices,target, v_alpha, h_alpha, v_beta, h_beta):
    doplot = True

    if doplot:
        field = np.zeros(target.shape, dtype=np.float32)

    mutation_rate = 1.0
    logpoints = np.arange(0,max_iter,2)
    #logpoints = np.array([max_iter+1])
    checkpoints = np.arange(11,max_iter,1)

    population = np.zeros((exposure_indices.shape[0],population_size),dtype=np.float32)
    fitness = np.zeros(population_size,dtype=np.float32)

    doses = np.zeros(exposure_indices.shape[0],dtype=np.float32)

    convergence = np.zeros(max_iter)
    t = np.zeros(max_iter)
    variance = 0

    i=0
    j=0
    start = np.linspace(5000, 200000, num=population_size, dtype=np.float32) # for 5nmperpixel
    #start = np.linspace(10000, 20000, num=population_size, dtype=np.float32) # for 5nmperpixel
    #start = np.linspace(10000,15000,num=population_size, dtype=np.float32) # for 3 nmperpixel
    for i in range(population_size):
        population[:, i] = np.repeat(start[i],population.shape[0])
        for j in range(population.shape[0]):
            population[j, i] = np.random.randint(-start.max()/50,start.max()/50)
    population = check_limits(population)
    sigma_start = start.max()/5
    sigma = sigma_start
    starttime = time.time()
    for i in range(max_iter):

        if i == 500:
            mutation_rate = 0.5
        #     population = smooth_doses(target,population,exposure_indices)
        # if i == 10:
        #     mutation_rate = 0.75
        #     sigma /= 20


        #def calc_fitness(population, exposure_indices, target, v_alpha, h_alpha, v_beta, h_beta):
        fitness = calc_fitness(population, exposure_indices, target, v_alpha, h_alpha, v_beta, h_beta)
        sorted_ind = np.argsort(fitness)
        fitness = fitness[sorted_ind]
        population = population[:,sorted_ind]
        #if fitness[sorted_ind][0] < 0.01:#0.01:
        #    break

        #if i < 50:
        #    sigma = sigma_start # 1
        #elif i < 100:
        #    sigma = sigma_start/2 # 0.5
        #else:
        std_err = 0.0
        slope = 0.0
        if i > 11:


            indices = np.arange(i-10,i,step=1)
            slope, intercept, r_value, p_value, std_err = linregress(indices,convergence[indices])
            std_err = np.abs((np.std(convergence[indices])*100)/intercept)
            variance = np.var(fitness)
            if i in checkpoints:

                #if (std_err > 0.1) or (slope > 0):
                #if (slope > 0):
                if i > 100:
                    if ((std_err > 5) and (slope < -5)) or (std_err > 1.0 and slope > -0.1):
                        if sigma > 10:
                            sigma *= 0.98

                if (std_err < 0.1) and (slope > 0) and (sigma < sigma_start/2):
                    sigma *= 1.02
                    #sigma += sigma_start/20

            if i > 1000 and len(fitness) > population_size/2:
                if np.random.random() < 0.5:
                    population = population[:, :-1]

            #if i > 1000 and len(fitness) > population_size/4:
            #    if np.random.random() < 0.5:
            #        population = population[:, :-1]

            # if i > 5000 and len(fitness) > population_size/4:
            #    if np.random.random() < 0.5:
            #        population = population[:, :-1]

            if i > 2000 and len(fitness) > population_size/3:
                if np.random.random() < 0.5:
                    population = population[:, :-1]

            # if i in [5000]:#i == 1000 or i == 5000:
            #     population = smooth_doses(target,population,exposure_indices)


            if doplot and not i%20:
                set_doses_field(field, exposure_indices, population[:,0])
                field[field==0] = np.nan
                plt.imshow(field)
                plt.colorbar()
                plt.savefig('current.png',dpi=300)
                plt.close()

        population = recombine_population(population)
        population = mutate_population(population,sigma,mutation_rate)
        population = check_limits(population)
        if i in logpoints:
            print("{0:7d}: fitness: {1:5.1f}, sigma: {2:5.5f}, std_err: {3:5.5f}, slope: {4:5.5f}, var: {5:5.5f}".format(i, fitness[0], sigma,std_err,slope,variance))
        convergence[i] = fitness[0]
        t[i] = time.time() - starttime

    print("Done -> Mean Error: {0:1.5f}, sigma: {1:3.5f}".format(convergence[:i][-1] , sigma))

    return population[:,0], t[:i], convergence[:i]


alpha_p = None
beta_p = None
eta_p = None
nmperpixel_p = None
try:
    with open('hv_vectors.obj', 'rb') as fp:
        alpha_p, beta_p, eta_p, nmperpixel_p, v_alpha, h_alpha, v_beta, h_beta = pickle.load(fp)
except:
    print('Error loading hv_vectors from file, generating h and v vectors')

if (alpha_p != alpha) or (beta_p != beta) or (eta_p != eta) or (nmperpixel_p != nmperpixel):
    v_alpha, h_alpha, v_beta, h_beta = generate_hv_vectors(alpha,beta,eta)
    with open('hv_vectors.obj', 'wb') as fp:
        pickle.dump((alpha, beta, eta, nmperpixel, v_alpha, h_alpha, v_beta, h_beta), fp)



#
# field = np.zeros(target.shape,dtype=np.float32)
# set_doses_field(field,exposure_indices,doses)
# exposure = calc_exposure(field,v_alpha,h_alpha,v_beta,h_beta)
# plt.imshow(exposure)
# plt.show()
#
# target = target*exposure.max()
# plt.imshow(target-exposure)
# plt.show()





# exposure_indices = put_circle(500,500-40,35)
# exposure_indices = put_circle(500,500+40,35,exposure_indices=exposure_indices)
#
# target = generate_empty_field_matrix(1000)
# set_target(target,exposure_indices,600.0)
#
# doses = np.repeat(1,exposure_indices.shape[0])
# doses = np.array(doses,dtype=np.float32)
# field = generate_empty_field_matrix(1000)
# set_doses_field(field,exposure_indices,doses)
#
# plt.imshow(target/target.max()-field)
# plt.show()


def make_grid(size,dist):
    size = int((size+size%2)/2)
    size += size%2
    print(size*2+1)
    x = np.linspace(-size,size,2*size+1,dtype=np.float32)
    x*=dist
    y = x
    x,y = np.meshgrid(x,y)
    return x.ravel(),y.ravel()


r = 60
dist = r+60
# x,y = make_grid(1,dist)
# x += 500
# y += 500
# exposure_indices = np.empty( shape=(0, 2) , dtype=np.int32)
# for i in range(x.shape[0]):
#     exposure_indices = put_circle(x[i], y[i], r, exposure_indices=exposure_indices)

exposure_indices = np.empty( shape=(0, 2) , dtype=np.int32)
exposure_indices = put_circle(500,500,r)
exposure_indices = put_circle(500+dist,500,r,exposure_indices=exposure_indices)
exposure_indices = put_circle(500-dist,500,r,exposure_indices=exposure_indices)
# exposure_indices = put_circle(500,500+dist,r,exposure_indices=exposure_indices)
# exposure_indices = put_circle(500,500-dist,r,exposure_indices=exposure_indices)
# exposure_indices = put_circle(500+dist,500+dist,r,exposure_indices=exposure_indices)
# exposure_indices = put_circle(500-dist,500-dist,r,exposure_indices=exposure_indices)
# exposure_indices = put_circle(500+dist,500-dist,r,exposure_indices=exposure_indices)
# exposure_indices = put_circle(500-dist,500+dist,r,exposure_indices=exposure_indices)

target = generate_empty_field_matrix(1000)
set_target(target,exposure_indices,600.0)

plt.imshow(target)
plt.show()


doses = np.linspace(90000,100000,exposure_indices.shape[0],dtype=np.float32)
field = generate_empty_field_matrix(1000)
set_doses_field(field,exposure_indices,doses)
exposure = calc_exposure(field,v_alpha,h_alpha,v_beta,h_beta)
print(np.sum(exposure-target))
plt.imshow(target-exposure)
plt.show()



#def iterate(x0,y0,repetitions,target):

print("Starting Iteration")
doses, t, convergence = iterate(exposure_indices,target, v_alpha, h_alpha, v_beta, h_beta)

plt.semilogy(t,convergence)
#plt.plot(t,convergence)
plt.xlabel('time / s')
plt.ylabel('Mean Error')
plt.tight_layout()
#plt.show()
plt.savefig('pics/convergence.png',dpi=600)
plt.close()



field = np.zeros(target.shape,dtype=np.float32)
set_doses_field(field,exposure_indices,doses)

exposure = calc_exposure(field,v_alpha,h_alpha,v_beta,h_beta)
plt.imshow((exposure-target))
plt.colorbar()
#plt.show()
plt.tight_layout()
plt.savefig('pics/exposure-target.png',dpi=1200)
plt.close()

plt.imshow(field)
plt.colorbar()
#plt.show()
plt.tight_layout()
plt.savefig('pics/doses.png',dpi=1200)
plt.close()

field = np.zeros(target.shape,dtype=np.float32)
set_doses_field(field,exposure_indices,doses)
exposure = calc_exposure(field,v_alpha,h_alpha,v_beta,h_beta)
plt.imshow(exposure)
plt.colorbar()
plt.contour(exposure, [590])  # [290,300, 310])
#plt.show()
plt.tight_layout()
plt.savefig('pics/exposure.png',dpi=1200)
plt.close()


# Outputfile = open(outfilename,'w')
#
# #Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11000, 11000, 5, 5" + '\n')
# #Outputfile.write('D ' + prefixes[l] + '-' + str(r) + ", 11000, 11000, 5, 5" + '\n')
# Outputfile.write('I 1' + '\n')
# Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
# Outputfile.write("FSIZE 15 micrometer" + '\n')
# Outputfile.write("UNIT 1 micrometer" + '\n')



#
#
# Outputfile = open(outfilename,'w')
#
# n_total = len(radius) * len(structures) * len(dists)
#
# bar = Bar('Status', max=n_total)
#
# for r in range(len(radius)):
#     for l in range(len(structures)):
#         for k in range(len(dists)):
#             at = (r+1)*(l+1)*(k+1)
#             print("structure #: {0:d}/{1:d} {4:s} ,dist #: {2:d}/{3:d}, iter #: {6:d}/{5:d}".format(l + 1, len(structures), k + 1, len(dists), prefixes[l], n_total, at))
#
#             #Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11500, 11500, 5, 5" + '\n')
#             #Outputfile.write('I 1' + '\n')
#             #Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
#             #Outputfile.write("FSIZE 20 micrometer" + '\n')
#             #Outputfile.write("UNIT 1 micrometer" + '\n')
#
#             Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11000, 11000, 5, 5" + '\n')
#             #Outputfile.write('D ' + prefixes[l] + '-' + str(r) + ", 11000, 11000, 5, 5" + '\n')
#             Outputfile.write('I 1' + '\n')
#             Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
#             Outputfile.write("FSIZE 15 micrometer" + '\n')
#             Outputfile.write("UNIT 1 micrometer" + '\n')
#
#             if isinstance(n, list) and isinstance(inner, list) and isinstance(centre, list):
#                 x0, y0, cx, cy = structures[l](dists[k], radius[r], n[r],inner[r],centre[r],dose_check_radius)
#                 # if radius[r] < 100:
#                 #     x0, y0 = structures[l](dists[k] + dose_check_radius * 2, radius[r] - dose_check_radius, n[r],inner[r],centre[r])
#                 #     cx, cy = structures[l](dists[k], radius[r], n[r])
#                 # else:
#                 #     x0, y0 = structures[l](dists[k],radius[r]*3/4, n[r], False, False)
#                 #     x01, y01 = structures[l](dists[k], radius[r] * 1 / 4, int(n[r]/2), False, False)
#                 #     x0 = np.concatenate((x0,x01))
#                 #     y0 = np.concatenate((y0, y01))
#                 #     cx, cy = structures[l](dists[k], radius[r], n[r])
#                 #     cx2, cy2 = structures[l](dists[k], int(radius[r]*2/4), int(n[r]*2/3))
#                 #     #cx3, cy3 = structures[l](dists[k], int(radius[r]*1/5), int(n[r]*1/3))
#                 #     cx3, cy3 = (np.array([500]),np.array([500]))
#                 #     #cx = np.concatenate((cx,cx2,cx3))
#                 #     #cy = np.concatenate((cy, cy2,cy3))
#                 #     cx = np.concatenate((cx,cx3))
#                 #     cy = np.concatenate((cy,cy3))
#             else:
#                 x0,y0 = structures[l](dists[k]+dose_check_radius*2,radius[r]-dose_check_radius,n)
#                 cx, cy = structures[l](dists[k],radius[r],n)
#
#             #plt.scatter(x0,y0)
#             #plt.show()
#             #plt.close()
#
#             repetitions = np.ones(len(x0),dtype=np.float32)*300
#
#             target = np.zeros((len(cx),2),dtype=np.float32)
#             target[:,0] = cx
#             target[:,1] = cy
#
#             repetitions, t, convergence = iterate(x0, y0, repetitions,target)
#
#             repetitions[np.where(repetitions < 1)] = 0
#             repetitions = np.array(np.round(repetitions),dtype=np.int)
#
#             x0 = x0 -500
#             y0 = y0 -500
#             target = target - 500
#
#             #x = np.arange(np.min(x0)-50,np.max(x0)+50,step=0.2)
#             #y = np.arange(np.min(y0)-50,np.max(y0)+50,step=0.2)
#             x = np.arange(np.min(x0)-50,np.max(x0)+50,step=1)
#             y = np.arange(np.min(y0)-50,np.max(y0)+50,step=1)
#             x, y = np.meshgrid(x, y)
#             orig_shape = x.shape
#             x = x.ravel()
#             y = y.ravel()
#             exposure = calc_map_2(x0, y0, repetitions * dwell_time * current, x, y) # C
#             exposure = exposure.reshape(orig_shape)
#             exposure = exposure * 1e6 # uC
#             pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
#             pixel_area = pixel_area * 1e-14  # cm^2
#             exposure = exposure/pixel_area # uC/cm^2
#
#             name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+".png"
#             name_pgf = "pics/" + prefixes[l] + "_" + str(dists[k]) + "_" + str(radius[r]) + ".pgf"
#             #fig = newfig(0.9)
#             cmap = sns.cubehelix_palette(light=1, as_cmap=True,reverse=False)
#             plot = plt.imshow(np.flipud(exposure),cmap=cmap,extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
#             cb = plt.colorbar()
#             #cb.set_label('Dosis / uC/cm^2 ')
#             cb.set_label(r'$Dosis\, / \, \frac{\mu C}{cm^2} ')
#             plt.contour(x.reshape(orig_shape), y.reshape(orig_shape), exposure, [target_dose])#[290,300, 310])
#             plt.xlabel(r'$x\,  /\,  nm')
#             plt.ylabel(r'$y\,  /\,  nm')
#             plt.tight_layout(.5)
#             plt.savefig(name,dpi=200)
#             #plt.savefig(name_pgf)
#             plt.close()
#
#             #x = np.arange(np.min(x0)-50,np.max(x0)+50,step=0.2)
#             #y = np.arange(np.min(y0)-50,np.max(y0)+50,step=0.2)
#             x = np.arange(np.min(x0)-50,np.max(x0)+50,step=1)
#             y = np.arange(np.min(y0)-50,np.max(y0)+50,step=1)
#             x, y = np.meshgrid(x, y)
#             orig_shape = x.shape
#             x = x.ravel()
#             y = y.ravel()
#             exposure = calc_map_2(x0, y0, repetitions * dwell_time * current, x, y) # C
#             exposure = exposure.reshape(orig_shape)
#             exposure = exposure * 1e6 # uC
#             pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
#             pixel_area = pixel_area * 1e-14  # cm^2
#             exposure = exposure/pixel_area # uC/cm^2
#
#             name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_expected.png"
#             name_pgf = name[:-4]+".pgf"
#             #fig = newfig(0.9)
#             plot = plt.imshow(np.flipud(exposure >= target_dose),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
#             plt.scatter(x0,y0,c="blue")
#             plt.scatter(target[:,0].ravel(), target[:,1].ravel(), c="red")
#             plt.axes().set_aspect('equal')
#             plt.xlabel(r'$x\,  /\,  nm')
#             plt.ylabel(r'$y\,  /\,  nm')
#             plt.tight_layout()
#             plt.savefig(name,dpi=200)
#             #plt.savefig(name_pgf)
#             plt.close()
#
#             name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_convergence.png"
#             name_pgf = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_convergence.pgf"
#             #fig = newfig(0.9)
#             print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
#             plt.semilogy(t,convergence)
#             plt.xlabel('time / s')
#             plt.ylabel('Mean Error')
#             plt.tight_layout()
#             plt.savefig(name,dpi=200)
#             #plt.savefig(name_pgf)
#             plt.close()
#
#             area = np.pi * (15*repetitions/np.max(repetitions))**2
#             #area = np.pi * ( 0.005*(np.max(y)-np.min(y)) * repetitions / np.max(repetitions)) ** 2
#             #fig = newfig(0.9)
#             plt.scatter(x0, y0, s=area, alpha=0.5,edgecolors="black",linewidths=1)
#             plt.axes().set_aspect('equal', 'datalim')
#             name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_scatter.png"
#             name_pgf = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(radius[r])+"_scatter.pgf"
#             plt.xlabel(r'$x\,  /\,  nm')
#             plt.ylabel(r'$y\,  /\,  nm')
#             plt.tight_layout()
#             plt.savefig(name,dpi=200)
#             #plt.savefig(name_pgf)
#             plt.close()
#             x0 = x0/1000+0.5
#             y0 = y0/1000+0.5
#             print(repetitions)
#             for j in range(len(x0)):
#                 if repetitions[j] >= 1:
#                     Outputfile.write('RDOT '+str(x0[j]) + ', ' + str(y0[j]) + ', ' + str((repetitions[j])) + '\n')
#             Outputfile.write('END' + '\n')
#             Outputfile.write('\n')
#             Outputfile.write('\n')
#
#             bar.next()
#             print("ETA: " + str(bar.eta_td))
#
# bar.finish()
# Outputfile.close()
#
