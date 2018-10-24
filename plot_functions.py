import numpy as np
from plotsettings import *
import matplotlib.pyplot as plt
from algorithm import calc_map
import parameters
from structures import Structure


def calc_dose_map(x0, y0, doses, stepsize=1):
    x = np.arange(np.min(x0) - 50, np.max(x0) + 50, step=stepsize)
    y = np.arange(np.min(y0) - 50, np.max(y0) + 50, step=stepsize)
    x, y = np.meshgrid(x, y)
    orig_shape = x.shape
    x = x.ravel()
    y = y.ravel()
    exposure = calc_map(x0, y0, doses, x, y)  # C
    exposure = exposure.reshape(orig_shape)
    exposure = exposure * 1e6  # uC
    pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
    pixel_area = pixel_area * 1e-14  # cm^2
    exposure = exposure / pixel_area  # uC/cm^2
    x = x.reshape(orig_shape)
    y = y.reshape(orig_shape)
    return x, y, exposure


# ----------- Make Plots ----------------------------------

def plot_dose_map(filename, x, y, exposure):
    plot = plt.imshow(np.flipud(exposure), extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
    cb = plt.colorbar()
    cb.set_label(r'$Dose\, / \, \frac{\mu C}{cm^2} ')
    plt.contour(x, y, exposure, [parameters.target_dose])
    plt.xlabel(r'$x\,  /\,  nm')
    plt.ylabel(r'$y\,  /\,  nm')
    plt.tight_layout(.5)
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_expected_shape(filename, x, y, exposure, x0, y0, cx, cy):
    plot = plt.imshow(np.flipud(exposure >= parameters.target_dose),
                      extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
    plt.scatter(x0, y0, c="blue", s=0.4)
    plt.scatter(cx.ravel(), cy.ravel(), c="red", s=0.4)
    plt.axes().set_aspect('equal')
    plt.xlabel(r'$x\,  /\,  nm')
    plt.ylabel(r'$y\,  /\,  nm')
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_convergence(filename, t, convergence):
    plt.semilogy(t, convergence)
    plt.xlabel('time / s')
    plt.ylabel('Mean Error')
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_exposurepoints(filename, x0, y0, doses, scalefactor=15):
    area = np.pi * (scalefactor * doses / np.max(doses)) ** 2
    # area = np.pi * ( 0.005*(np.max(y)-np.min(y)) * repetitions / np.max(repetitions)) ** 2
    plt.scatter(x0, y0, s=area, alpha=0.5, edgecolors="black", linewidths=1)
    plt.axes().set_aspect('equal', 'datalim')
    plt.xlabel(r'$x\,  /\,  nm')
    plt.ylabel(r'$y\,  /\,  nm')
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


def plot_setup(filename: str, struct: Structure, scalefactor: float = 1):
    assert scalefactor > 0
    area = 0.4 * scalefactor
    plt.scatter(struct[0], struct[1], s=area, alpha=0.5, c='red', edgecolors="black", linewidths=1)
    plt.scatter(struct[2], struct[3], s=area, alpha=0.5, c='blue', edgecolors="black", linewidths=1)
    plt.axes().set_aspect('equal', 'datalim')
    plt.xlabel(r'$x\,  /\,  nm')
    plt.ylabel(r'$y\,  /\,  nm')
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()
