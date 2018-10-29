import numpy as np
from numba import jit, float64, int64, njit, prange
import math
from scipy import integrate
import time
from scipy.stats import linregress
import parameters

"""
In this file the PSF (power spectrum function, the energy distribution of the beam) and the genetic (optimization)
algorithm are being defined. Uses Numba for performance optimization - Numba decorators are used in this file.
jit = just in time compilation. More: http://numba.pydata.org/numba-doc/0.35.0/user/jit.html
"""

# ---------- PSF -------------

normalization = 1


@njit(float64(float64, float64))
def _gauss(r, sigma):
    return (1 / (sigma ** 2)) * math.exp(-r ** 2 / sigma ** 2)


@njit(float64(float64))
def tripple_gaussian_improved(r):
    """
    According to eq. 5 from http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
    :param r: Radius from centre
    :return: Deposited power
    """
    return (1 / normalization) * (1 / (math.pi * (1 + parameters.eta_1 + parameters.eta_2))) * (
            _gauss(r, parameters.alpha) + parameters.eta_1 * _gauss(r, parameters.beta) + parameters.eta_2 / (
            24 * parameters.gamma ** 2) * math.exp(-math.sqrt(r / parameters.gamma)))


@njit(float64(float64))
def tripple_gaussian_simple(r):
    """
    Like eq. 2 from http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
    :param r: Radius from centre
    :return: Deposited power
    """
    return (1 / normalization) * (1 / (math.pi * (1 + parameters.eta_1 + parameters.eta_2))) * (
            _gauss(r, parameters.alpha) + parameters.eta_1 * _gauss(r, parameters.beta) + parameters.eta_2 * _gauss(
        r, parameters.gamma))


@njit(float64(float64))
def double_gaussian_simple(r):
    """
    Like eq. 2 from http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
    :param r: Radius from centre
    :return: Deposited power
    """
    return (1 / normalization) * (1 / (math.pi * (1 + parameters.eta_1))) * (
            _gauss(r, parameters.alpha) + parameters.eta_1 * _gauss(r, parameters.beta))


calc_prox = tripple_gaussian_improved  # Used PSF
# [return] = C/nm !!!
normalization = integrate.quad(lambda x: 2 * np.pi * x * calc_prox(x), 0, np.inf)


# print('norm:'+str(normalization)) # <-- uncomment to check normalization

# ----------- Genetic Algorithm --------------

@njit(float64(float64, float64, float64, float64))
def dist(x0, y0, x, y):
    """
    Calculates the distance between two points
    :param x0: x coordinate of point 1
    :param y0: y coordinate of point 1
    :param x: x coordinate of point 2
    :param y: y coordinate of point 2
    :return: distance
    """
    return math.sqrt((x0 - x) ** 2 + (y0 - y) ** 2)


@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:]), parallel=True)
def calc_map(x0, y0, doses, x, y):
    """
    Calculates the dose of several exposure points at serveral arbitrary points.
    Can be used to make a map of the dose distribution.
    :param x0: x coordinate of exposure points
    :param y0: y coordinate of exposure points
    :param x: x coordinate of points where dose is calculated
    :param y: y coordinate of points where dose is calculated
    :return: calculated doses points at x and y
    """
    exposure = np.zeros(len(x), dtype=np.float64)
    pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
    for i in prange(len(x)):
        for j in range(len(x0)):
            r = dist(x0[j], y0[j], x[i], y[i])
            exposure[i] += calc_prox(r) * doses[j] * pixel_area
    return exposure


@njit(float64[:](float64[:, :], float64[:]), parallel=True)
def calc_exposure(proximity, doses):
    """
    Calculates the effective dose in each point.
    :param proximity: 2D array containing the mutual distances between all exposure and check points
    :param doses: A list of the current doses
    :return: A list of the calculated doses in all the check points
    """
    exposure = np.zeros(proximity.shape[1], dtype=np.float64)
    for i in prange(proximity.shape[1]):
        for j in range(proximity.shape[0]):
            exposure[i] += proximity[j, i] * doses[j]
    return exposure


@njit(float64[:, :](float64[:], float64[:]), parallel=True)
def recombine_arrays(arr1, arr2):
    """
    Recombines two arrays (mother and father) to two new arrays (children), models bisexual reproduction
    :param arr1: mother array
    :param arr2: father array
    :return: two children
    """
    res = np.zeros((len(arr1), 2), dtype=np.float64)
    res[:, 0] = arr1
    res[:, 1] = arr2
    # n_crossover = int(len(arr1)/3)
    n_crossover = int(len(arr1) * parameters.crossover_size)

    i = 0
    for i in prange(n_crossover):
        k = np.random.randint(0, len(arr1) - 1)
        alpha = np.random.random()
        res[k, 0] = alpha * arr1[k] + (1 - alpha) * arr2[k]
        res[k, 1] = alpha * arr2[k] + (1 - alpha) * arr1[k]
    return res

# Alternative recombination methods

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


@njit(float64[:](float64[:], float64), parallel=True)
def mutate(arr, sigma):
    """
    Introduces random varations (mutations) to the elements of an array.
    :param arr: array to mutate
    :param sigma: strength/scale of mutations
    :return: mutated array
    """
    for i in prange(arr.shape[0]):
        if np.random.random() < parameters.mutation_rate:
            mutation = np.random.normal() * sigma
            if mutation > sigma * 1.0:
                mutation = sigma
            if mutation < -sigma * 1.0:
                mutation = -sigma
            arr[i] = arr[i] + mutation
    return arr


@njit(float64[:](float64[:, :], float64[:, :], float64[:]), parallel=True)
def calc_fitness(population, proximity, fixed_dose_exposure):
    """
    Calculates the fitness for every individual of the population
    :param proximity: Matrix with pre-computed proximity values
    :param fixed_dose_exposure: Array with the doses from the fixed dose points
    :return: Array with fitness values
    """
    fitness = np.zeros(population.shape[1], dtype=np.float64)
    pixel_area = 1  # nm^2 #pixel_area * 1e-14  # cm^2

    for p in range(population.shape[1]):
        exposure = calc_exposure(proximity[:, :], population[:, p])
        if len(fixed_dose_exposure) == len(exposure):
            exposure += fixed_dose_exposure
        exposure = (exposure * 1e6) / (pixel_area * 1e-14)  # uC/cm^2
        fitness[p] = np.mean(np.abs(np.subtract(parameters.target_dose, exposure))) ** 2

    if parameters.force_low_gradient:
        for p in range(population.shape[1]):
            fitness[p] = fitness[p] + np.mean(
                np.abs(population[:-1, p] - population[1:, p])) * 1e14 * parameters.gradient_weight

    return fitness


@njit(float64[:, :](float64[:, :]), parallel=True)
def recombine_population(population):
    """
    Recombines the individuals of a population
    :param population: Population matrix
    :return: new recombined population
    """
    # n_recombination = int(population.shape[1]/3)
    n_recombination = int(population.shape[1] / 2)
    # n_recombination = int(population.shape[1])

    for i in prange(int(n_recombination / 2)):
        k = 2 * i
        l = 2 * i + 1
        r_rec = recombine_arrays(population[:, k], population[:, l])
        population[:, -k] = r_rec[:, 0]
        population[:, -l] = r_rec[:, 1]

    return population

# Alternative function for recombining the population

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


@njit(float64[:, :](float64[:, :], float64), parallel=True)
def mutate_population(population, sigma):
    """
    Mutate all individuals of a population
    :param population: Population matrix
    :param sigma: strength/scale of mutations
    :return: new recombined population
    """
    # for i in prange(population.shape[1]):
    #    population[:, i] = mutate(population[:, i], sigma)

    for i in range(population.shape[1]):
        # if i < int(population.shape[1]/3):
        if i < 4:
            population[:, i] = mutate(population[:, i], sigma / 10)  #
        elif i < 10:
            population[:, i] = mutate(population[:, i], sigma / 2)  #
        else:
            population[:, i] = mutate(population[:, i], sigma)  #
    return population


@njit(float64[:, :](float64[:, :]), parallel=True)
def check_limits(population):
    """
    Check if all individuals of the are bigger than 0, meaning that no negativ exposure is allowed
    :param population: Population matrix
    :return: corrected population
    """
    for i in prange(population.shape[1]):
        for j in range(population.shape[0]):
            if population[j, i] < 0:
                population[j, i] = 0
    return population


@jit()  # (float64(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
def iterate(x0, y0, cx, cy, fixed_dose_points=None, verbose: bool = True, report_every: int = 500):
    """
    This is the main function. Calling this will start the algorithm and iterate until the convergence criterions are met.
    :param x0: x coordinates of the exposure points
    :param y0: y coordinates of the exposure points
    :param cx: x coordinates of the dose-check points
    :param cy: y coordinates of the dose-check points
    :param fixed_dose_points: Definition of points with a fixed dose (are not optimized). The format is a list
                              containing three same-length lists:
                                - x coordinates of the fixed exposure points
                                - y coordinates of the fixed exposure points
                                - doses of the fixed exposure points
    :param verbose: Should we be verbose in the output prints?
    :param report_every: How often shall we give an intermediate report?
    :return: best found doses, required time for each iteration, fitness of each iteration
    """

    if verbose:
        print('Preparing optimization ...')

    # At which points to report on the progress?
    logpoints = np.arange(report_every, parameters.max_iter, report_every)
    checkpoints = np.arange(50, parameters.max_iter, 50)

    # List containing the convergence data for each iteration
    convergence = np.zeros(parameters.max_iter)

    # List containing the required time for each iteration
    t = np.zeros(parameters.max_iter)

    # Create a 2D Matrix of the distances between the exposure and dose-check points
    proximity = np.zeros((len(x0), cx.shape[0]), dtype=np.float64)
    for i in range(proximity.shape[0]):
        for j in range(proximity.shape[1]):
            proximity[i, j] = calc_prox(dist(x0[i], y0[i], cx[j], cy[j]))

    # Do a similar calculation for potential points with a fixed (non-optimized) dose
    if fixed_dose_points:
        fd_proximity = np.zeros((fixed_dose_points.shape[1], cx.shape[0]))
        for i in range(fixed_dose_points.shape[1]):
            for j in range(cx.shape[0]):
                fd_proximity[i, j] = calc_prox(
                    dist(fixed_dose_points[0, i], fixed_dose_points[1, i], cx[j], cy[j]))
        fixed_dose_exposure = calc_exposure(proximity=fd_proximity, doses=fixed_dose_points[2])
    else:
        fixed_dose_exposure = np.zeros((1,)) # required to fulfil Numba argument type specification

    # population is a 2D array containing the doses for each member of the population
    population = np.zeros((len(x0), parameters.population_size), dtype=np.float64)
    # Assign starting doses to each point for each member of the population, with some randomness
    start = np.linspace(0, parameters.starting_dose * 10, num=parameters.population_size)
    for i in range(parameters.population_size):
        population[:, i] = np.repeat(start[i], len(x0))
        for j in range(len(x0)):
            population[j, i] = population[j, i] + (np.random.rand() - 0.5) * parameters.starting_dose * 0.1
            if population[j, i] < 0:
                population[j, i] = 0

    starting_sigma = parameters.starting_dose / 2

    sigma = 0
    slope = 0.0
    variance = 0.0
    start_time = time.time()

    print('Starting optimization ...')

    for i in range(parameters.max_iter):
        fitness = calc_fitness(population, proximity, fixed_dose_exposure)
        sorted_ind = np.argsort(fitness)
        fitness = fitness[sorted_ind]
        population = population[:, sorted_ind]

        # If target fitness is already reached, exit the loop
        if 100 * np.sqrt(fitness[0]) / parameters.target_dose < parameters.target_fitness:
            break

        if i < 2000:
            sigma = starting_sigma  # 1
        elif i < 4000:
            sigma = starting_sigma / 2  # 0.5
        else:
            if i in checkpoints:
                indices = np.arange(i - 500, i, step=1)
                slope, intercept, r_value, p_value, std_err = linregress(t[indices], convergence[indices])
                variance = np.var(fitness) / fitness[0]
                if slope > 0 and variance > 0.0001:
                    if sigma > starting_sigma * 0.000001:
                        sigma *= 0.98

                if slope > 0 and variance < 0.0001:
                    if sigma < starting_sigma:
                        sigma *= 1.02

        population = recombine_population(population)
        population = mutate_population(population, sigma)

        population = check_limits(population)
        if i in logpoints and verbose:
            # Print a short summary of the optimization progress
            sfitn = 100 * np.sqrt(fitness[0]) / parameters.target_dose
            ssig = sigma / parameters.starting_dose
            ps = "{0:7d}: fitness: {1:1.5f}%, sigma_rel: {2:1.5f}, var: {3:1.5f}, slope: {4:1.5f}"
            print(ps.format(i, sfitn, ssig, variance, slope))

        convergence[i] = fitness[0]
        t[i] = time.time() - start_time

    print('Done', end='')
    if verbose:
        print(' in %.1fs' % (time.time() - start_time))
    print(" -> Mean Error: {0:1.5f}%, sigma: {1:1.5f}".format(convergence[:i][-1], sigma))

    return population[:, 0], t[:i], convergence[:i]
