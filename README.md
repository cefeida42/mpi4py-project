# Parallelization in Python using mpi4py

This was my project for the university course "Applications of Super-computers in Astronomy" which is a part of the PhD program in astrophysics at Faculty of Mathematics, University of Belgrade.

Here, I utilize mpi4py package to parallize the task of calculating metrics for active galaxies light curve quality estimation in the context of Legacy Survey of Space and Time observing strategies. Basically, I use domain decomposition method of parallelization to split the same calculations on same input data, but for different observing strategies.

The code is not fully reproducible because our light curve simulation functions are not publicly available yet. It only serves to demonstrate my skills in mpi4py.
