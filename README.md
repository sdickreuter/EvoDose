# EvoDose
Python code for electron beam lithography (EBL) proximity correction based on the minimization of the error with an evolutionary algorithm. Can export files for XENOS and Raith EBL systems.

.. TODO ..

Here's how it works:
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

EvoDose is suited best for small structures, where the dose inside the structure will be reached automatically by the exposure points on the outside. It is possible to setup structures with checkpoints on the inside, but it is harder to get good results.
Also for thin elongated structures, the algorithm struggles to converge nicely (see line_example). One solution we used is to change the fitness function (TODO) to include kind of the divergence of the dose of the exposure points which helps the algorithm to get a more symmetric result.
