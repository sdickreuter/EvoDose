import numpy as np
import matplotlib.pyplot as plt


from structures import get_line
from algorithm import iterate, calc_map
import parameters

# Parameters for Exposure
parameters.current = 100 * 1e-12 # A
parameters.dwell_time = 800 * 1e-9 # s
parameters.target_dose = 900 # uC/cm^2
# Parameters for Genetic Algorithm
parameters.population_size = 50
parameters.max_iter = 1000000
parameters.target_fitness = 0.5


# Name for all saved stuff
outfilename = 'line.txt'

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
exposure = calc_map(x0, y0, repetitions * parameters.dwell_time * parameters.current, x, y) # C
exposure = exposure.reshape(orig_shape)
exposure = exposure * 1e6 # uC
pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
pixel_area = pixel_area * 1e-14  # cm^2
exposure = exposure/pixel_area # uC/cm^2

#----------- Make Plots ----------------------------------

name = "pics/"+outfilename+".pdf"
plot = plt.imshow(np.flipud(exposure),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
cb = plt.colorbar()
#cb.set_label('Dosis / uC/cm^2 ')
cb.set_label(r'$Dose\, / \, \frac{\mu C}{cm^2} ')
plt.contour(x.reshape(orig_shape), y.reshape(orig_shape), exposure, [parameters.target_dose])#[290,300, 310])
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
exposure = calc_map(x0, y0, repetitions * parameters.dwell_time * parameters.current, x, y) # C
exposure = exposure.reshape(orig_shape)
exposure = exposure * 1e6 # uC
pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
pixel_area = pixel_area * 1e-14  # cm^2
exposure = exposure/pixel_area # uC/cm^2

name = "pics/"+outfilename+"_expected.pdf"
plot = plt.imshow(np.flipud(exposure >= parameters.target_dose),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
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
Outputfile.write('C '+str(int(parameters.dwell_time*1e9)) + '\n')
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

