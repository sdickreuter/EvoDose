import numpy as np
from plotsettings import *
import matplotlib.pyplot as plt

import structures
import algorithm
from output import output_xenos, output_raith
from plot_functions import *
import parameters
import os

# ----- Parameters for Exposure -----
parameters.current = 34.8 * 1e-12  # A
parameters.dwell_time = 59 * 10 * 1e-9  # s
parameters.target_dose = 79.8 * 10  # uC/cm^2

parameters.alpha = 175
parameters.beta = 1141
parameters.gamma = 3738

parameters.eta_1 = 0.433
parameters.eta_2 = 0.984

# this is reasonable starting point for setting the starting_dose,
# but if convergence is bad consider tuning this value
parameters.starting_dose = 0.25 * parameters.target_dose * 1e-16

# ----- Parameters for Genetic Algorithm -----
parameters.population_size = 50
parameters.max_iter = int(10e5)
# this is a reasonable value for structures with good convergence that is a balance between
# computation time and accuracy
parameters.target_fitness = 1e-5

algorithm.calc_prox = algorithm.tripple_gaussian_simple

outfilename = 'marker_1.txt'

# ------------- Make Structures ---------------------

struct = structures.get_square_marker_1(size=200-3, n=5, corner_comp=8, centre_dot=True, dose_check_radius=39, offset=3)
x0, y0, cx, cy = struct

# Make sure the picture output folder exists
folder = 'pics'
if not os.path.isdir(folder):
    os.mkdir(folder)
# Plot the setup
name = folder + "/" + outfilename[:-4] + "_setup.png"
plot_setup(name, struct)

# ----------- Run Iterations ------------------------

doses, t, convergence = algorithm.iterate(x0, y0, cx, cy)
print("time for iteration: " + str(np.round(np.max(t), 2)) + " seconds")

# --------------- Write Pattern -------------
output_xenos(outfilename, 'line_test', x0, y0, doses)
output_raith(outfilename[:-3] + 'asc', 1, x0, y0, doses)

# ----------- Prepare Data for Plots ----------------------------------
x, y, exposure = calc_dose_map(x0, y0, doses)

# ----------- Plot results ----------------------------------
name = folder + "/" + outfilename[:-4] + ".png"
plot_dose_map(name, x, y, exposure)

name = folder + "/" + outfilename[:-4] + "_expected.png"
plot_expected_shape(name, x, y, exposure, x0, y0, cx, cy)

name = folder + "/" + outfilename[:-4] + "_convergence.png"
plot_convergence(name, t, convergence)

name = folder + "/" + outfilename[:-4] + "_scatter.png"
plot_exposurepoints(name, x0, y0, doses)
