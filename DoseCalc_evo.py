import numpy as np

from plotsettings import *

import matplotlib.pyplot as plt
import seaborn as sns

from numba import jit,float64,int64

import math

import time

from scipy.stats import linregress
from scipy import integrate

alpha = 32.9 #nm
beta = 2610 #nm
gamma =2*4.1 #nm
eta_1 = 1.66
eta_2 = 1.27


#radius = 15
#radius = [35,40,45,50,55,60]
radius = [20]

def rot(alpha):
    return np.matrix( [[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]] )

def get_circle(r,n=12,inner_circle=False,centre_dot=False):
    x = np.zeros(0)
    y = np.zeros(0)

    v = np.array( [r,0] )

    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    if inner_circle:
        v = np.array( [r/2,0] )
        n = int(n/2)
        for i in range(n):
            x2, y2 = (v*rot(2*np.pi/n*i+2*np.pi/(2*n))).A1
            x = np.hstack( (x,x2) )
            y = np.hstack( (y,y2) )

    if centre_dot:
        x = np.hstack( (x,0) )
        y = np.hstack( (y,0) )

    #xv, yv = np.meshgrid(x, y)
    #xv = np.ravel(xv)
    #yv = np.ravel(yv)
    return x,y

def get_trimer(dist,r,n):
    x = np.zeros(0)
    y = np.zeros(0)
    v = np.array( [0,0.5*(dist+2*r)/np.sin(np.pi/3)] )
    m = 3
    for i in range(m):
        x2, y2 = (v*rot(2*np.pi/m*i)).A1
        x1,y1 = get_circle(r,n)

        x = np.hstack( (x,x1+x2) )
        y = np.hstack( (y,y1+y2) )

    x += 500
    y += 500

    return x,y

def get_dimer(dist,r,n):
    x1,y1 = get_circle(r,n)
    x2,y2 = get_circle(r,n)
    x1 -= (r+dist/2)
    x2 += (r+dist/2)

    x = np.concatenate((x1,x2))+500
    y = np.concatenate((y1,y2))+500

    # x1,y1 = get_circle(r,n=n)
    # x2,y2 = get_circle(r,n=n)
    # x1 -= (r+dist/2)
    # x2 += (r+dist/2)
    #
    # cx = np.concatenate((x1,x2))+500
    # cy = np.concatenate((y1,y2))+500

    return x,y#, cx, cy

def get_hexamer(dist,r,n):
    x = np.zeros(0)
    y = np.zeros(0)
    v = np.array( [0.5*(dist+2*r)/np.sin(np.pi/6),0] )
    m = 6
    for i in range(m):
        x2, y2 = (v*rot(2*np.pi/m*i)).A1
        x1,y1 = get_circle(r,n)

        x = np.hstack( (x,x1+x2) )
        y = np.hstack( (y,y1+y2) )

    x += 500
    y += 500

    return x,y


def get_asymdimer(dist,r,n):
    x1,y1 = get_circle(r,n)
    r2 = 1.5*r
    x2,y2 = get_circle(r2,n)
    x1 -= r+dist/2
    x2 += r2+dist/2

    x = np.concatenate((x1,x2))+500
    y = np.concatenate((y1,y2))+500

    return x,y

def get_single(dist,r,n):
    x1, y1 = get_circle(r, n=n, inner_circle=False, centre_dot=True)

    #if r >= 50:
    #    x1,y1 = get_circle(r,n=48,inner_circle=True,centre_dot=True)
    #else:
    #    x1, y1 = get_circle(r, n=32, inner_circle=False, centre_dot=True)

    x = x1+3750
    y = y1+3750

    return x,y


def get_triple(dist,r,n):
    x1,y1 = get_circle(r,n)
    x2,y2 = get_circle(r,n)
    x3,y3 = get_circle(r,n)
    x1 -= 2*r+dist
    x2 += 2*r+dist

    x = np.concatenate((x1,x2,x3))+500
    y = np.concatenate((y1,y2,y3))+500

    return x,y

def get_line(dist,r):
    n = int(r/dist)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] += i*dist

    return x,y


current = 100 * 1e-12 # A
dwell_time = 200 * 1e-9 # s
dose_check_radius = 3 # nm
target_dose = 70 # uC/cm^2
n = 12

#outfilename = 'emre2.txt'
#outfilename = 'asymdimer.txt'
#outfilename = 'single.txt'
#outfilename = 'pillars_r'+str(radius[0])+'nm_dose'+str(target_dose)+'_test2.txt'
outfilename = 'test.txt'
#outfilename = 'anni_kreise.txt'


prefixes = ["pillar_dimer","pillar_trimer","pillar_hexamer","pillar_asymdimer","pillar_triple"]
structures = [get_dimer,get_trimer,get_hexamer,get_asymdimer,get_triple]
dists = [40]
#for i in range(50):
#    dists.append(dists[i]+1)
#for i in range(10):
#    dists.append(dists[i] + 5)

# prefixes = ["pillar_dimer"]#,"pillar_trimer","pillar_hexamer"]#,"pillar_asymdimer","pillar_triple"]
# structures = [get_dimer]#,get_trimer,get_hexamer]#,get_asymdimer,get_triple]
# dists = [40]
# radius = 100
#prefixes = ["kreis"]
#structures = [get_single]#,get_trimer,get_hexamer,get_asymdimer,get_triple]
#dists = [1]


normalization = 1
#http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
@jit(float64(float64),nopython=True)
def calc_prox(r):
    return (1/normalization) * (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*math.exp(-r**2/alpha**2) + (eta_1/beta**2)*math.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*math.exp(-math.sqrt(r/gamma)) )
# [return] = C/nm !!!
normalization = integrate.quad(lambda x: 2*np.pi*x*calc_prox(x), 0, np.inf)
print('norm:'+str(normalization))
#normalization = 2.41701729505915



@jit(float64(float64,float64,float64,float64),nopython=True)
def dist(x0,y0,x,y):
    return math.sqrt( (x0-x)*(x0-x)+(y0-y)*(y0-y) )


@jit(float64[:](float64[:],float64[:],float64[:],float64[:],float64[:]),nopython=True)
def calc_map_2(x0,y0,doses,x,y):
    exposure = np.zeros(len(x),dtype=np.float64)
    pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
    for i in range(len(x)):
        for j in range(len(x0)):
            exposure[i] += calc_prox(dist(x0[j],y0[j],x[i],y[i]))*doses[j]* pixel_area
    return exposure

@jit(float64[:](float64[:,:],float64[:]),nopython=True)
def calc_map(proximity,doses):
    exposure = np.zeros(proximity.shape[1],dtype=np.float64)
    for i in range(proximity.shape[1]):
        for j in range(proximity.shape[0]):
            exposure[i] += proximity[j,i]*doses[j]
    return exposure

@jit(float64[:, :](float64[:], float64[:]),nopython=True)
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

@jit(float64[:](float64[:],float64, float64),nopython=True)
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


@jit(float64[:](float64[:,:],float64[:,:]),nopython=True)
def calc_fitness(population,proximity):
    fitness = np.zeros(population.shape[1],dtype=np.float64)
    exposure = np.zeros(population.shape[1],dtype=np.float64)
    pixel_area =  1 #nm^2 #pixel_area * 1e-14  # cm^2
    #repetitions = np.zeros(population.shape[0],dtype=np.float64)

    for p in range(population.shape[1]):
        #for i in range(population.shape[0]):
        #    repetitions[i] = round(population[i,p])
        exposure = calc_map(proximity[:,:],population[:, p] * current * dwell_time)
            #exposure = calc_map(proximity[:, j, :], repetitions * current * dwell_time)
        exposure = (exposure* 1e6)/(pixel_area*1e-14 ) # uC/cm^2
        fitness[p] = np.sum(np.abs(target_dose-exposure))/exposure.shape[0]

    return fitness

@jit(float64[:,:](float64[:,:]),nopython=True)
def recombine_population(population):
    #n_recombination = 6
    #n_recombination = int(population.shape[1]/3)
    n_recombination = int(population.shape[1]/2)

    for i in range(int(n_recombination/2)):
        k = 2*i
        l = 2*i+1
        r_rec = recombine_arrays(population[:, k],population[:, l])
        population[:, -k] = r_rec[:, 0]
        population[:, -l] = r_rec[:, 1]

    return population

@jit(float64[:,:](float64[:,:],float64, float64),nopython=True)
def mutate_population(population,sigma,mutation_rate):

    for i in range(population.shape[1]):
        #if i < int(population.shape[1]/3):
        if i < 2:
            population[:, i] = mutate(population[:, i], sigma/10, mutation_rate)#
        elif i < 6:
            population[:, i] = mutate(population[:, i], sigma/2, mutation_rate)#
        else:
            population[:, i] = mutate(population[:, i], sigma, mutation_rate)  #
    return population

@jit(float64[:,:](float64[:,:]),nopython=True)
def check_limits(population):
    for i in range(population.shape[1]):
        for j in range(population.shape[0]):
            if population[j, i] < 0.5:
                population[j, i] = 0
    return population



@jit()
def partition(values, idxs, left, right):
    """
    Partition method
    """

    piv = values[idxs[left]]
    i = left + 1
    j = right

    while True:
        while i <= j and values[idxs[i]] <= piv:
            i += 1
        while j >= i and values[idxs[j]] >= piv:
            j -= 1
        if j <= i:
            break

        idxs[i], idxs[j] = idxs[j], idxs[i]

    idxs[left], idxs[j] = idxs[j], idxs[left]

    return j


@jit()
def argsort1D(values):

    idxs = np.arange(values.shape[0])

    left = 0
    right = values.shape[0] - 1

    max_depth = np.int(right / 2)

    ndx = 0

    tmp = np.zeros((max_depth, 2), dtype=np.int64)

    tmp[ndx, 0] = left
    tmp[ndx, 1] = right

    ndx = 1
    while ndx > 0:

        ndx -= 1
        right = tmp[ndx, 1]
        left = tmp[ndx, 0]

        piv = partition(values, idxs, left, right)

        if piv - 1 > left:
            tmp[ndx, 0] = left
            tmp[ndx, 1] = piv - 1
            ndx += 1

        if piv + 1 < right:
            tmp[ndx, 0] = piv + 1
            tmp[ndx, 1] = right
            ndx += 1

    return idxs

population_size = 60
max_iter = 200000
#max_iter = 50000

@jit()#(float64(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
def iterate(x0,y0,repetitions,target):
    mutation_rate = 0.3
    logpoints = np.arange(500,max_iter,500)
    checkpoints = np.arange(50,max_iter,50)

    population = np.zeros((len(x0),population_size),dtype=np.float64)
    fitness = np.zeros(population_size,dtype=np.float64)
    pixel_area =  1 #nm^2 #pixel_area * 1e-14  # cm^2


    proximity = np.zeros((population.shape[0],target.shape[0]),dtype=np.float64)
    convergence = np.zeros(max_iter)
    t = np.zeros(max_iter)

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
    starttime = time.time()
    for i in range(max_iter):
        fitness = calc_fitness(population, proximity)
        sorted_ind = np.argsort(fitness)
        #sorted_ind = argsort1D(fitness)

        if fitness[sorted_ind][0] < 0.001:
            break

        if i < 1000:
            sigma = 1
        elif i < 2000:
            sigma = 0.5
        else:
            if i in checkpoints:
                indices = np.arange(i-500,i,step=1)
                slope, intercept, r_value, p_value, std_err = linregress(t[indices],convergence[indices])
                #print(std_err)
                #if slope > 0.01:
                #    sigma -= 0.01
                #if (std_err > 0.0003) or (slope > 0):
                #if (std_err > 0.005) or (slope > 0):
                if (std_err > 0.1) or (slope > 0):
                    sigma *= 0.98
                if (std_err < 0.0001) and (slope > 0):
                    sigma *= 1.02

                if sigma < 0.001:
                    sigma = 0.001

        population = population[:,sorted_ind]
        population = recombine_population(population)
        population = mutate_population(population,sigma,mutation_rate)
        population = check_limits(population)
        if i in logpoints:
            print("{0:7d}: fitness: {1:1.5f}, sigma: {2:1.5f}".format(i, fitness[sorted_ind][0], sigma))
        convergence[i] = fitness[sorted_ind][0]
        t[i] = time.time() - starttime


    return population[:,0], t[:i], convergence[:i]




Outputfile = open(outfilename,'w')

for r in radius:
    for l in range(len(structures)):
        for k in range(len(dists)):
            print("structure #: {0:d}/{1:d}, dist #: {2:d}/{3:d}".format(l+1, len(structures), k+1, len(dists)))

            Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11500, 11500, 5, 5" + '\n')
            Outputfile.write('I 1' + '\n')
            Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
            Outputfile.write("FSIZE 20 micrometer" + '\n')
            Outputfile.write("UNIT 1 micrometer" + '\n')

            #Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11000, 11000, 5, 5" + '\n')
            #Outputfile.write('D ' + prefixes[l] + '-' + str(r) + ", 11000, 11000, 5, 5" + '\n')
            #Outputfile.write('I 1' + '\n')
            #Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
            #Outputfile.write("FSIZE 15 micrometer" + '\n')
            #Outputfile.write("UNIT 1 micrometer" + '\n')

            x0,y0 = structures[l](dists[k]+dose_check_radius*2,r-dose_check_radius,n)

            repetitions = np.ones(len(x0),dtype=np.float64)*100

            cx, cy = structures[l](dists[k],r,n)
            target = np.zeros((len(cx),2),dtype=np.float64)
            target[:,0] = cx
            target[:,1] = cy

            # x_c, y_c = get_circle(r*1.1, n=16,inner_circle=False,centre_dot=False)
            # target = np.zeros((len(x0),len(x_c),2),dtype=np.float64)
            # for i in range(len(x0)):
            #    target[i,:,0] = x_c
            #    target[i, :, 1] = y_c


            # # for lines
            # #x_c,y_c = get_line(0.5,1.5)
            # x_c = np.array([0])
            # y_c = np.array([0])
            # target = np.zeros((len(x0),len(x_c)*2,2),dtype=np.float64)
            # for i in range(len(x0)):
            #     target[i,:,0] = np.append(x_c,x_c)+x0[i]
            #     target[i, :, 1] = np.append(y_c-10,y_c+10) + y0[i]

            # fig = plt.figure()
            # plt.scatter(x0-500, y0-500, c="blue")
            # plt.scatter(target[:,0].ravel()-500, target[:,1].ravel()-500, c="red")
            # plt.tight_layout(.5)
            # plt.show()
            # plt.close()


            repetitions, t, convergence = iterate(x0, y0, repetitions,target)

            repetitions = np.array(np.round(repetitions),dtype=np.int)

            x0 = x0 -500
            y0 = y0 -500
            target = target - 500

            x = np.arange(np.min(x0)-50,np.max(x0)+50,step=0.2)
            y = np.arange(np.min(y0)-50,np.max(y0)+50,step=0.2)
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

            name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(r)+".png"
            name_pgf = "pics/" + prefixes[l] + "_" + str(dists[k]) + "_" + str(r) + ".pgf"
            fig = newfig(0.9)
            cmap = sns.cubehelix_palette(light=1, as_cmap=True,reverse=False)
            plot = plt.imshow(np.flipud(exposure),cmap=cmap,extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
            plt.colorbar()
            plt.contour(x.reshape(orig_shape), y.reshape(orig_shape), exposure, [target_dose])#[290,300, 310])
            #plt.scatter(x_t,y_t,c="red")
            #plt.show()
            plt.xlabel('x / nm')
            plt.ylabel('y / nm')
            plt.tight_layout(.5)
            plt.savefig(name,dpi=300)
            plt.savefig(name_pgf)
            plt.close()

            x = np.arange(np.min(x0)-10,np.max(x0)+10,step=0.2)
            y = np.arange(np.min(y0)-10,np.max(y0)+10,step=0.2)
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

            name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(r)+"_expected.png"
            name_pgf = name[:-4]+".pgf"
            fig = newfig(0.9)
            plot = plt.imshow(np.flipud(exposure >= target_dose),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
            #plt.contour(x_t.reshape(target_shape), y_t.reshape(target_shape), target.reshape(target_shape), [299],color="black")
            #plt.scatter(x_t,y_t,c="red")
            plt.scatter(x0,y0,c="blue")
            plt.scatter(target[:,0].ravel(), target[:,1].ravel(), c="red")
            plt.xlabel('x / nm')
            plt.ylabel('y / nm')
            plt.tight_layout()
            plt.savefig(name,dpi=300)
            plt.savefig(name_pgf)
            plt.close()

            name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(r)+"_convergence.png"
            name_pgf = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(r)+"_convergence.pgf"
            fig = newfig(0.9)
            print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
            plt.semilogy(t,convergence)
            plt.xlabel('time / s')
            plt.ylabel('Mean Error')
            plt.tight_layout()
            plt.savefig(name,dpi=300)
            plt.savefig(name_pgf)
            plt.close()

            #plt.scatter(x_t,y_t,c="red")
            #plt.scatter(x0,y0,c="blue")
            #plt.show()

            area = np.pi * (15*repetitions/np.max(repetitions))**2
            fig = newfig(0.9)
            plt.scatter(x0, y0, s=area, alpha=0.5,edgecolors="black",linewidths=1)
            plt.axes().set_aspect('equal', 'datalim')
            name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(r)+"_scatter.png"
            name_pgf = "pics/"+prefixes[l]+"_"+str(dists[k])+"_"+str(r)+"_scatter.pgf"
            plt.xlabel('x / nm')
            plt.ylabel('y / nm')
            plt.tight_layout()
            plt.savefig(name,dpi=300)
            plt.savefig(name_pgf)
            plt.close()
            x0 = x0/1000+0.5
            y0 = y0/1000+0.5
            print(repetitions)
            for j in range(len(x0)):
                if repetitions[j] > 1:
                    Outputfile.write('RDOT '+str(x0[j]) + ', ' + str(y0[j]) + ', ' + str((repetitions[j])) + '\n')
            Outputfile.write('END' + '\n')
            Outputfile.write('\n')
            Outputfile.write('\n')

Outputfile.close()
