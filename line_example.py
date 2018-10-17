import numpy as np
from plotsettings import *
import matplotlib.pyplot as plt

from structures import get_line
from algorithm import iterate
from output import output_xenos, output_raith
from plot_functions import *
import parameters

#----- Parameters for Exposure -----
parameters.current = 100 * 1e-12 # A
parameters.dwell_time = 800 * 1e-9 # s
parameters.target_dose = 600 # uC/cm^2
parameters.starting_dose = parameters.target_dose*1e-16

#----- Parameters for Genetic Algorithm -----
parameters.population_size = 50
parameters.max_iter = 1000000

# if using force_low_gradient, the target_fitness has to be adjusted, because
# the fitness now can't reach zero anymore as there is always a gradient present.
# Also gradient_weight can be adjusted to determine how much the gradient influences
# the fitness
parameters.force_low_gradient = True
parameters.target_fitness = 1.08

outfilename = 'line.txt'

#------------- Make Structures ---------------------

x0, y0, cx, cy = get_line(500,60,50)

# plt.scatter(x0,y0)
# plt.scatter(cx,cy)
# plt.show()
# plt.close()

#----------- Run Iterations ------------------------

doses, t, convergence = iterate(x0, y0, cx, cy)

#------- adjust position
x0 = x0 +500
y0 = y0 +500
cx = cx + 500
cy = cy + 500

#--------------- Write Pattern -------------
output_xenos(outfilename,'line_test',x0,y0,doses)
output_raith(outfilename[:-3]+'asc',1,x0,y0,doses)

#----------- Prepare Data for Plots ----------------------------------
x,y,exposure = calc_dose_map(x0,y0,doses)

#----------- Make Plots ----------------------------------
name = "pics/"+outfilename[:-4]+".png"
plot_dose_map(name,x,y,exposure)

name = "pics/"+outfilename[:-4]+"_expected.png"
plot_expected_shape(name,x,y,exposure,x0,y0,cx,cy)

name = "pics/"+outfilename[:-4]+"_convergence.png"
plot_convergence(name,t,convergence)

name = "pics/"+outfilename[:-4]+"_scatter.png"
plot_exposurepoints(name,x0,y0,doses)