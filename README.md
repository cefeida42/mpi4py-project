# Parallelization in Python using mpi4py

This was my project for university course "Applications of Super-computers in Astronomy".

Here, I utilize mpi4py package to parallize the task of calculating metrics for active galaxies light curve quality estimation in the context of Legacy Survey of Space and Time observing strategies. Basically, I use domain decomposition method of parallelization to split the same calculations on same input data, but for different observing strategies.

The code is not fully reproducible because our light curve simulation functions are not publicly available yet. It only serves to demonstrate my skills in mpi4py.
