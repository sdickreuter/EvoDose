import numpy as np
import matplotlib as mpl

# from https://scipy-cookbook.readthedocs.io/items/Matplotlib_LaTeX_Examples.html
# and https://stackoverflow.com/questions/39614011/antialiased-text-rendering-in-matplotlib-pgf-backend

mpl.use("pgf")

def figsize(scale):
    fig_width_pt = 336.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "lualatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[ngerman]{babel}",
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        r"\usepackage{wasysym}",
        ],
    "lines.antialiased" : True, # render lines in antialised (no jaggies)
    "patch.antialiased" : True, # render patches in antialised (no jaggies)
    "text.antialiased" : True, # If True (default), the text will be antialiased.
    }

def newfig():
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax

mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks", {'axes.linewidth': 0.5,
                        'xtick.direction': 'in',
                        'ytick.direction': 'in',
                        'xtick.major.size': 3,
                        'xtick.minor.size': 1.5,
                        'ytick.major.size': 3,
                        'ytick.minor.size': 1.5
                        })

sns.set_context("paper", rc= {  "xtick.major.width":0.5,
                            "ytick.major.width":0.5,
                            "xtick.minor.width":0.5,
                            "ytick.minor.width":0.5})
