import numpy as np
from plotsettings import *
import matplotlib.pyplot as plt

from structures import get_dimer
from algorithm import iterate, calc_map
from output import output_xenos, output_raith
import parameters

# Parameters for Exposure
parameters.current = 100 * 1e-12 # A
parameters.dwell_time = 800 * 1e-9 # s
parameters.target_dose = 600 # uC/cm^2
parameters.starting_dose = parameters.target_dose*1e-16

# Parameters for Genetic Algorithm
parameters.population_size = 50
parameters.max_iter = 1000000
parameters.target_fitness = 0.1

outfilename = 'dimer.txt'

#------------- Make Structures ---------------------

x0, y0, cx, cy = get_dimer(20,40,24)

# plt.scatter(x0,y0)
# plt.scatter(cx,cy)
# plt.show()
# plt.close()

#----------- Run Iterations ------------------------

doses, t, convergence = iterate(x0, y0, cx, cy)

#--------------- Write Pattern -------------
output_xenos(outfilename,'line_test',x0,y0,doses)
output_raith(outfilename[:-3]+'asc',1,x0,y0,doses)

#----------- Prepare Data for Plots ----------------------------------

x = np.arange(np.min(x0)-50,np.max(x0)+50,step=1)
y = np.arange(np.min(y0)-50,np.max(y0)+50,step=1)
x, y = np.meshgrid(x, y)
orig_shape = x.shape
x = x.ravel()
y = y.ravel()
exposure = calc_map(x0, y0, doses, x, y) # C
exposure = exposure.reshape(orig_shape)
exposure = exposure * 1e6 # uC
pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
pixel_area = pixel_area * 1e-14  # cm^2
exposure = exposure/pixel_area # uC/cm^2

#----------- Make Plots ----------------------------------

name = "pics/"+outfilename[:-4]+".png"
plot = plt.imshow(np.flipud(exposure),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
cb = plt.colorbar()
#cb.set_label('Dosis / uC/cm^2 ')
cb.set_label(r'$Dose\, / \, \frac{\mu C}{cm^2} ')
plt.contour(x.reshape(orig_shape), y.reshape(orig_shape), exposure, [parameters.target_dose])
plt.xlabel(r'$x\,  /\,  nm')
plt.ylabel(r'$y\,  /\,  nm')
plt.tight_layout(.5)
plt.savefig(name,dpi=600)
plt.close()

#x = np.arange(np.min(x0)-50,np.max(x0)+50,step=0.2)
#y = np.arange(np.min(y0)-50,np.max(y0)+50,step=0.2)
x = np.arange(np.min(x0)-50,np.max(x0)+50,step=1)
y = np.arange(np.min(y0)-50,np.max(y0)+50,step=1)
x, y = np.meshgrid(x, y)
orig_shape = x.shape
x = x.ravel()
y = y.ravel()
exposure = calc_map(x0, y0, doses, x, y) # C
exposure = exposure.reshape(orig_shape)
exposure = exposure * 1e6 # uC
pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
pixel_area = pixel_area * 1e-14  # cm^2
exposure = exposure/pixel_area # uC/cm^2

name = "pics/"+outfilename[:-4]+"_expected.png"
plot = plt.imshow(np.flipud(exposure >= parameters.target_dose),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
plt.scatter(x0,y0,c="blue")
plt.scatter(cx.ravel(), cy.ravel(), c="red")
plt.axes().set_aspect('equal')
plt.xlabel(r'$x\,  /\,  nm')
plt.ylabel(r'$y\,  /\,  nm')
plt.tight_layout()
plt.savefig(name,dpi=600)
plt.close()

name = "pics/"+outfilename[:-4]+"_convergence.png"
print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
plt.semilogy(t,convergence)
plt.xlabel('time / s')
plt.ylabel('Mean Error')
plt.tight_layout()
plt.savefig(name,dpi=600)
plt.close()

name = "pics/"+outfilename[:-4]+"_scatter.png"
area = np.pi * (15*doses/np.max(doses))**2
#area = np.pi * ( 0.005*(np.max(y)-np.min(y)) * repetitions / np.max(repetitions)) ** 2
plt.scatter(x0, y0, s=area, alpha=0.5,edgecolors="black",linewidths=1)
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel(r'$x\,  /\,  nm')
plt.ylabel(r'$y\,  /\,  nm')
plt.tight_layout()
plt.savefig(name,dpi=600)
plt.close()
x0 = x0/1000+0.5
y0 = y0/1000+0.5


