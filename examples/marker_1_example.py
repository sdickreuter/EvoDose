from EvoDose import algorithm, structures
from EvoDose.output import output_xenos, output_raith
from EvoDose.plot_functions import *
import os

# ----- Parameters for Exposure -----
f = 15
parameters.current = 34.8 * 1e-12  # A
parameters.dwell_time = 59 * (f / 14) * 1e-9  # s
parameters.target_dose = 79.8 * f  # uC/cm^2

# Beam settings
algorithm.calc_prox = algorithm.tripple_gaussian_simple
parameters.alpha = 175
parameters.beta = 1141
parameters.gamma = 3738

parameters.eta_1 = 0.433
parameters.eta_2 = 0.984

# this is reasonable starting point for setting the starting_dose,
# but if convergence is bad consider tuning this value
parameters.starting_dose = 0.5 * parameters.target_dose * 1e-16

# ----- Parameters for Genetic Algorithm -----
parameters.population_size = 50
parameters.max_iter = int(1e5)
# this is a reasonable value for structures with good convergence that is a balance between
# computation time and accuracy
parameters.target_fitness = 0.8e-5


outfilename = 'marker_1.txt'

# ------------- Make Structures ---------------------
print('Creating the structures...')

offs = 0
small_size = 200
corner_comp = 9.8
struct = structures.get_square_marker_1(size=small_size - offs, n=5, corner_comp=corner_comp, centre_dot=False,
                                        dose_check_radius=39.9, offset=offs)
x0, y0, cx, cy = struct

cdl = (small_size - offs)/2 # Centre dot location
d1 = 2.65e-13
fixed_doses = (
    [-cdl, +cdl, ], # x
    [+cdl, -cdl, ], # y
    [d1, d1, ] # doses
)

# Make sure the picture output folder exists
folder = 'pics'
if not os.path.isdir(folder):
    os.mkdir(folder)
# Plot the setup
print('Plotting the structure setup...')
name = folder + "/" + outfilename[:-4] + "_setup.png"
plot_setup(name, struct)

# ----------- Run Iterations ------------------------

doses, t, convergence = algorithm.iterate(*struct)

# Merge fixed dose points with the rest for plotting
x0, y0 = structures.merge_basic(x0, y0, *fixed_doses[:2])
doses = np.hstack([doses, fixed_doses[2]])

# --------------- Write Pattern -------------
print("Writing eBeam files...")
output_xenos(outfilename, 'line_test', x0, y0, doses)
output_raith(outfilename[:-3] + 'asc', 1, x0, y0, doses)

# ----------- Prepare Data for Plots ----------------------------------
print("Calculating Dose Map...")
x, y, exposure = calc_dose_map(x0, y0, doses)

# ----------- Plot results ----------------------------------
print("Plotting Dose Map...")
name = folder + "/" + outfilename[:-4] + ".png"
plot_dose_map(name, x, y, exposure)

print("Plotting Expected Shape...")
name = folder + "/" + outfilename[:-4] + "_expected.png"
plot_expected_shape(name, x, y, exposure, x0, y0, cx, cy)

print("Plotting Convergence Data...")
name = folder + "/" + outfilename[:-4] + "_convergence.png"
plot_convergence(name, t, convergence)

print("Plotting Dose Amplitudes...")
name = folder + "/" + outfilename[:-4] + "_scatter.png"
plot_exposurepoints(name, x0, y0, doses)
