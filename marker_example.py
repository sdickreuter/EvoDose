import numpy as np
from plotsettings import *
import matplotlib.pyplot as plt

import structures
import algorithm
from output import output_xenos, output_raith
from plot_functions import *
import parameters

# ----- Parameters for Exposure -----
parameters.current = 100 * 1e-12  # A
parameters.dwell_time = 800 * 1e-9  # s
parameters.target_dose = 600  # uC/cm^2

# this is reasonable starting point for setting the starting_dose,
# but if convergence is bad consider tuning this value
parameters.starting_dose = parameters.target_dose * 1e-16

# ----- Parameters for Genetic Algorithm -----
parameters.population_size = 50
parameters.max_iter = 1000000
# this is a reasonable value for structures with good convergence that is a balance between
# computation time and accuracy
parameters.target_fitness = 0.1

algorithm.calc_prox = algorithm.tripple_gaussian_simple

outfilename = 'dimer.txt'

# ------------- Make Structures ---------------------

x0, y0, cx, cy = structures.get_square_marker_1()

# Plot the setup
name = "pics/" + outfilename[:-4] + "_setup.png"
plot_setup(name, (x0, y0, cx, cy))

# ----------- Run Iterations ------------------------

doses, t, convergence = algorithm.iterate(x0, y0, cx, cy)
print("time for iteration: " + str(np.round(np.max(t), 2)) + " seconds")

# --------------- Write Pattern -------------
output_xenos(outfilename, 'line_test', x0, y0, doses)
output_raith(outfilename[:-3] + 'asc', 1, x0, y0, doses)

# ----------- Prepare Data for Plots ----------------------------------
x, y, exposure = calc_dose_map(x0, y0, doses)

# ----------- Plot results ----------------------------------
name = "pics/" + outfilename[:-4] + ".png"
plot_dose_map(name, x, y, exposure)

name = "pics/" + outfilename[:-4] + "_expected.png"
plot_expected_shape(name, x, y, exposure, x0, y0, cx, cy)

name = "pics/" + outfilename[:-4] + "_convergence.png"
plot_convergence(name, t, convergence)

name = "pics/" + outfilename[:-4] + "_scatter.png"
plot_exposurepoints(name, x0, y0, doses)
