import numpy as np

import matplotlib.pyplot as plt

from numba import jit,float64,int64,njit, prange

import math

import time

from scipy.stats import linregress
from scipy import integrate

# Parameters for Exposure
current = 100 * 1e-12 # A
dwell_time = 800 * 1e-9 # s
target_dose = 900 # uC/cm^2

# Name for all saved stuff
outfilename = 'line.txt'

# Parameters for Genetic Algorithm
population_size = 50
max_iter = 1000000
target_fitness = 0.1


#---------- Functions for generating Structures -------------------

@njit()
def get_line(length, width, n):
    dist = length / n
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] += i * dist

    cx = np.zeros(n)
    cy = np.zeros(n) + width / 2
    for i in range(n):
        cx[i] += i * dist

    cx2 = np.zeros(n)
    cy2 = np.zeros(n) - width / 2
    for i in range(n):
        cx2[i] += i * dist

    cx = np.concatenate((cx,cx2))
    cy = np.concatenate((cy, cy2))

    return x,y,cx,cy




#--------- PSF --------------------

alpha = 32.9*1.5 #nm
beta = 2610 #nm
gamma = 4.1 * 1.5 #nm
eta_1 = 1.66
eta_2 = 1.27

normalization = 1
#http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
@njit(float64(float64))
def calc_prox(r):
    return (1/normalization) * (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*math.exp(-r**2/alpha**2) + (eta_1/beta**2)*math.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*math.exp(-math.sqrt(r/gamma)) )
# [return] = C/nm !!!
normalization = integrate.quad(lambda x: 2*np.pi*x*calc_prox(x), 0, np.inf)
print('norm:'+str(normalization))
#normalization = 2.41701729505915


#--------- Genetic Algorithm --------------------

@njit(float64(float64,float64,float64,float64))
def dist(x0,y0,x,y):
    return math.sqrt( (x0-x)*(x0-x)+(y0-y)*(y0-y) )


@njit(float64[:](float64[:],float64[:],float64[:],float64[:],float64[:]),parallel=True)
def calc_map_2(x0,y0,doses,x,y):
    exposure = np.zeros(len(x),dtype=np.float64)
    pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
    for i in range(len(x)):
        for j in range(len(x0)):
            r= dist(x0[j],y0[j],x[i],y[i])
            exposure[i] += calc_prox(r)*doses[j]* pixel_area
    return exposure

@njit(float64[:](float64[:,:],float64[:]),parallel=True)
def calc_map(proximity,doses):
    exposure = np.zeros(proximity.shape[1],dtype=np.float64)
    for i in range(proximity.shape[1]):
        for j in range(proximity.shape[0]):
            exposure[i] += proximity[j,i]*doses[j]
    return exposure

@njit(float64[:, :](float64[:], float64[:]),parallel=False)
def recombine_arrays(arr1, arr2):
    res = np.zeros((len(arr1), 2), dtype=np.float64)
    res[:, 0] = arr1
    res[:, 1] = arr2
    n_crossover = int(len(arr1)/3)
    for i in range(n_crossover):
        k = np.random.randint(0, len(arr1) - 1)
        alpha = np.random.random()
        res[k, 0] = alpha * arr1[k] + (1 - alpha) * arr2[k]
        res[k, 1] = alpha * arr2[k] + (1 - alpha) * arr1[k]
    return res

# @jit(float64[:, :](float64[:], float64[:]),nopython=True)#,parallel=True)
# def recombine_arrays(arr1, arr2):
#     res = np.zeros((len(arr1), 2), dtype=np.float64)
#     res[:, 0] = arr1
#     res[:, 1] = arr2
#     alpha = 0.25
#     for i in range(len(arr1)):
#         res[i, 0] = alpha * arr1[i] + (1 - alpha) * arr2[i]
#         res[i, 1] = alpha * arr2[i] + (1 - alpha) * arr1[i]
#     return res

# @jit(float64[:](float64[:], float64[:]),nopython=True)#,parallel=True)
# def recombine_arrays_simple(arr1, arr2):
#     res = np.zeros(len(arr1), dtype=np.float64)
#     n_crossover = int(len(arr1)/3)
#     res = arr1
#     for i in range(n_crossover):
#         k = np.random.randint(0, len(arr1) - 1)
#         alpha = np.random.random()
#         res[i] = alpha * arr1[k] + (1 - alpha) * arr2[k]
#     return res


@njit(float64[:](float64[:],float64, float64),parallel=True)
def mutate(arr,sigma,mutation_rate):
    for i in range(arr.shape[0]):
        if np.random.random() < mutation_rate:
            mutation = np.random.normal()*sigma
            if mutation > sigma*1.0:
                mutation = sigma
            if mutation < -sigma*1.0:
                mutation = -sigma
            arr[i] = arr[i] + mutation
    return arr


@njit(float64[:](float64[:,:],float64[:,:]),parallel=False)
def calc_fitness(population,proximity):
    fitness = np.zeros(population.shape[1],dtype=np.float64)
    #exposure = np.zeros(population.shape[1],dtype=np.float64)
    pixel_area =  1 #nm^2 #pixel_area * 1e-14  # cm^2
    #repetitions = np.zeros(population.shape[0],dtype=np.float64)

    for p in range(population.shape[1]):
        #for i in range(population.shape[0]):
        #    repetitions[i] = round(population[i,p])
        exposure = calc_map(proximity[:,:],population[:, p] * current * dwell_time)
            #exposure = calc_map(proximity[:, j, :], repetitions * current * dwell_time)
        exposure = (exposure* 1e6)/(pixel_area*1e-14 ) # uC/cm^2
        #sum = 0.0
        #for i in range(exposure.shape[0]):
        #    sum += math.fabs(target_dose-exposure[i])
        #fitness[p] = sum/exposure.shape[0]
        #fitness[p] = np.sum(np.abs(target_dose-exposure))/exposure.shape[0]
        fitness[p] = np.mean(np.abs(np.subtract(target_dose,exposure)))**2

    return fitness

@njit(float64[:,:](float64[:,:]),parallel=True)
def recombine_population(population):
    #n_recombination = 6
    #n_recombination = int(population.shape[1]/3)
    n_recombination = int(population.shape[1]/2)
    #n_recombination = int(population.shape[1])

    for i in prange(int(n_recombination/2)):
        k = 2*i
        l = 2*i+1
        r_rec = recombine_arrays(population[:, k],population[:, l])
        population[:, -k] = r_rec[:, 0]
        population[:, -l] = r_rec[:, 1]

    return population

# @jit(float64[:,:](float64[:,:]),nopython=True)
# def recombine_population(population):
#     new_pop = np.zeros(population.shape,dtype=np.float64)
#     n = population.shape[0]
#
#     unfit = np.arange(int(n/4),n)
#     fit = np.arange(0,int(n/2))
#
#     i = 0
#
#     while True:
#         mother = fit[np.random.randint(0,len(fit))]
#         father = fit[np.random.randint(0,len(fit))]
#         new_pop[unfit[i],:] = recombine_arrays_simple(population[mother, :], population[father, :])
#         i += 1
#         if i >= len(unfit):
#              break
#
#         # childs = recombine_arrays(population[mother, :], population[father, :])
#         # new_pop[unfit[i],:] = childs[:, 0]
#         # i += 1
#         # if i >= len(unfit):
#         #     break
#         # new_pop[unfit[i], :] = childs[:, 1]
#         # i += 1
#         # if i >= len(unfit):
#         #     break
#
#     rest = np.arange(0,unfit[0])
#
#     for i in rest:
#         new_pop[i, :] = unfit[np.random.randint(0,len(unfit))]
#
#     return new_pop


@njit(float64[:,:](float64[:,:],float64, float64),parallel=True)
def mutate_population(population,sigma,mutation_rate):
    for i in prange(population.shape[1]):
        population[:, i] = mutate(population[:, i], sigma, mutation_rate)

    # for i in range(population.shape[1]):
    #     #if i < int(population.shape[1]/3):
    #     if i < 4:
    #         population[:, i] = mutate(population[:, i], sigma/10, mutation_rate)#
    #     elif i < 10:
    #         population[:, i] = mutate(population[:, i], sigma/2, mutation_rate)#
    #     else:
    #         population[:, i] = mutate(population[:, i], sigma, mutation_rate)  #
    return population

@njit(float64[:,:](float64[:,:]),parallel=True)
def check_limits(population):
    for i in prange(population.shape[1]):
        for j in range(population.shape[0]):
            if population[j, i] < 0.1:
                population[j, i] = 0
    return population


@jit()#(float64(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
def iterate(x0,y0,repetitions,target):
    mutation_rate = 0.2
    logpoints = np.arange(500,max_iter,500)
    #logpoints = np.array([max_iter+1])
    checkpoints = np.arange(50,max_iter,50)

    population = np.zeros((len(x0),population_size),dtype=np.float64)
    fitness = np.zeros(population_size,dtype=np.float64)
    pixel_area =  1 #nm^2 #pixel_area * 1e-14  # cm^2


    proximity = np.zeros((population.shape[0],target.shape[0]),dtype=np.float64)
    convergence = np.zeros(max_iter)
    t = np.zeros(max_iter)

    i = 0
    j = 0
    for i in range(population.shape[0]):
        for j in range(target.shape[0]):
            proximity[i,j] = calc_prox(dist(x0[i],y0[i],target[j,0],target[j,1]))

    start = np.linspace(0,200,num=population_size)
    for i in range(population_size):
        #population[:, i] = repetitions + np.random.randint(-50, 50)
        population[:, i] = np.repeat(start[i],len(repetitions))
        for j in range(len(repetitions)):
            population[j, i] = population[j, i]+np.random.randint(-20,20)
            if population[j, i] < 1:
                population[j, i] = 1

    #print("Starting Iteration")
    sigma = 5.0
    slope = 0.0
    std_err = 0.0
    variance = 0.0
    starttime = time.time()

    for i in range(max_iter):
        fitness = calc_fitness(population, proximity)
        sorted_ind = np.argsort(fitness)
        #sorted_ind = argsort1D(fitness)
        fitness = fitness[sorted_ind]
        population = population[:,sorted_ind]

        if 100*np.sqrt(fitness[0])/target_dose < target_fitness:#0.01:
            break


        if i < 2000:
            sigma = 5#1
        elif i < 4000:
            sigma = 2#0.5
        else:
            if i in checkpoints:
                indices = np.arange(i-500,i,step=1)
                slope, intercept, r_value, p_value, std_err = linregress(t[indices],convergence[indices])
                variance = np.var(fitness) / fitness[0]
                # if (std_err > 0.001) or (slope > 0):
                #     sigma *= 0.99
                # if (std_err < 0.001):# and (slope > 0):
                #     sigma *= 1.03
                if slope > 0 and variance > 0.01:
                    if sigma > 0.001:
                        sigma *= 0.98


                if slope > 0 and variance < 0.01:
                    if sigma < 1:
                        sigma *= 1.02

        #sigma = np.sqrt(fitness[0])/4

        population = recombine_population(population)
        population = mutate_population(population,sigma,mutation_rate)

        population = check_limits(population)
        if i in logpoints:
            print("{0:7d}: fitness: {1:1.5f}%, sigma: {2:1.5f}, var: {3:1.5f}, slope: {4:1.5f}".format(i, 100*np.sqrt(fitness[0])/target_dose, sigma,variance,slope))

        convergence[i] = fitness[0]
        t[i] = time.time() - starttime

    print("Done -> Mean Error: {0:1.5f}%, sigma: {1:1.5f}".format(convergence[:i][-1] , sigma))

    return population[:,0], t[:i], convergence[:i]



#------------- Make Structures ---------------------

x0, y0, cx, cy = get_line(500,60,50)

x1, y1, cx1, cy1 = get_line(500,60,50)
y1 += 500
cy1 += 500

x0 = np.hstack((x0,x1))
y0 = np.hstack((y0,y1))
cx = np.hstack((cx,cx1))
cy = np.hstack((cy,cy1))


# plt.scatter(x0,y0)
# plt.scatter(cx,cy)
# plt.show()
# plt.close()

#----------- Run Iterations ------------------------

repetitions = np.ones(len(x0),dtype=np.float64)*300

target = np.zeros((len(cx),2),dtype=np.float64)
target[:,0] = cx
target[:,1] = cy

repetitions, t, convergence = iterate(x0, y0, repetitions,target)

repetitions[np.where(repetitions < 1)] = 0
repetitions = np.array(np.round(repetitions),dtype=np.int)

#------- adjust positions
x0 = x0 +500
y0 = y0 +500
target = target + 500

#----------- Prepare Data for Plots ----------------------------------

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

#----------- Make Plots ----------------------------------

name = "pics/"+outfilename+".pdf"
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

name = "pics/"+outfilename+"_expected.pdf"
plot = plt.imshow(np.flipud(exposure >= target_dose),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
plt.scatter(x0,y0,c="blue")
plt.scatter(target[:,0].ravel(), target[:,1].ravel(), c="red")
plt.axes().set_aspect('equal')
plt.xlabel(r'$x\,  /\,  nm')
plt.ylabel(r'$y\,  /\,  nm')
plt.tight_layout()
plt.savefig(name,dpi=200)
plt.close()

name = "pics/"+outfilename+"_convergence.pdf"
print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
plt.semilogy(t,convergence)
plt.xlabel('time / s')
plt.ylabel('Mean Error')
plt.tight_layout()
plt.savefig(name,dpi=200)
plt.close()

name = "pics/"+outfilename+"_scatter.pdf"
area = np.pi * (15*repetitions/np.max(repetitions))**2
#area = np.pi * ( 0.005*(np.max(y)-np.min(y)) * repetitions / np.max(repetitions)) ** 2
plt.scatter(x0, y0, s=area, alpha=0.5,edgecolors="black",linewidths=1)
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel(r'$x\,  /\,  nm')
plt.ylabel(r'$y\,  /\,  nm')
plt.tight_layout()
plt.savefig(name,dpi=200)
plt.close()
x0 = x0/1000+0.5
y0 = y0/1000+0.5


#--------------- Write Pattern -------------

Outputfile = open(outfilename,'w')

#Outputfile.write('D line_test, 11000, 11000, 5, 5' + '\n')
Outputfile.write('D line_test' + '\n')
Outputfile.write('I 1' + '\n')
Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
Outputfile.write("FSIZE 15 micrometer" + '\n')
Outputfile.write("UNIT 1 micrometer" + '\n')

print(repetitions)
for j in range(len(x0)):
    if repetitions[j] >= 1:
        Outputfile.write('RDOT '+str(x0[j]) + ', ' + str(y0[j]) + ', ' + str((repetitions[j])) + '\n')
Outputfile.write('END' + '\n')
Outputfile.write('\n')
Outputfile.write('\n')

Outputfile.close()

