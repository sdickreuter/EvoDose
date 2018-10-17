
# Parameters for Exposure
current = 100 * 1e-12 # A
dwell_time = 500 * 1e-9 # s
target_dose = 600 # uC/cm^2
starting_dose = current * dwell_time * 300 # C
#starting_dose = target_dose*1e-15

# Parameters for PSF
alpha = 32.9*1.5 #nm
beta = 2610 #nm
gamma = 4.1 * 1.5 #nm
eta_1 = 1.66
eta_2 = 1.27

# Parameters for Genetic Algorithm
population_size = 50
max_iter = 100000
target_fitness = 0.1

# Advanced Parameters for Genetic Algorithm
crossover_size = 1/3 # 0.0 - 1.0
mutation_rate = 0.2