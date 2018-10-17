# EvoDose
Python code for electron beam lithography (EBL) proximity correction based on the minimization of the error with an evolutionary algorithm. Can export files for XENOS and Raith EBL systems.

See dimer_example.py on how to use it.

## How it works
Patterns are defined by exposure points (blue) and check points (red):
![pillar_dimer_36_30_expected](https://user-images.githubusercontent.com/6985888/47080356-1513f180-d208-11e8-9ddf-df3a0c6d2a63.png)

The check points define the outline of the pattern.
EvoDose now tries to optimize the doses of every exposure point so that the the dose that reaches the check points is as close to the wanted dose (here  600 uC/cm^2) as possible.
After optimization the exposure points for example look like this (circle radii show the dose):
![pillar_dimer_36_30_scatter](https://user-images.githubusercontent.com/6985888/47080504-7045e400-d208-11e8-832c-213e92c3c716.png)

With this one can also plot a dose distribution:
![pillar_dimer_36_30](https://user-images.githubusercontent.com/6985888/47080558-9b303800-d208-11e8-89ec-14eaaadf46af.png)

The black line is where the dose is exactly at 600 uC/cm^2, and gives a clue on how the exposed pattern will look like.
The plots are made with dimer_example.py.

## Scope

EvoDose is suited best for small structures, where the dose inside the structure will be reached automatically by the exposure points on the outside. It is possible to setup structures with checkpoints on the inside, but it is harder to get good convergence, because the placement and distance of these inner check points from the exposure points has great influence on the degrees of freedom of the minimization problem.

Also for thin elongated structures, the algorithm struggles to converge nicely (see line_example.py). One solution is to change the fitness function to include the divergence of the dose of the exposure points which helps the algorithm to get a more symmetric result, see see line_example.py for reference.

## PSF

The psf from Aya et. al. (http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf) is used, but in principle any psf can be used. In the code the psf can be found in algorithm.py as the calc_prox function.
An example on how to get the parameters for the psd (see parameters.py) from measurements is shown in fit_psf.py.

## Tips & Tricks

If convergence does not work, consider the following points:
- Change behaviour of the iterate function in algorithm.py, it has a simple mechanism built-in for lowering the sigma value with increasing iterations and thus annealing the populations of the algorithm. This greatly enhances convergence, but the way it is done might not be good for all problems, as the algorithm might get stuck in a local minima.
- In algorithm.py there are different, commented out, versions of recombine_population and recombine_arrays that will change the convergence behaviour.
- Check if parameters.target_fitness is actually reachable for your structure (very important when parameters.force_low_gradient = True), and tune it accordingly
- Tuning parameters.population_size can help with convergence, but also increases computation time
- Tune parameters.mutation_rate and/or parameters.crossover_size
- **Most important**: Convergence is only possible if the minimization of your structure is possible, try using as few check and exposure points as possible and adjust their position to enhance convergence. This is especially important for check points that are inside of structures.
