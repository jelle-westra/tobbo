This is repo is a restructured version of https://github.com/BayesOptApp/Topology_Optimization;
inital changes made are found in the 'jelle' branch of the forked version: https://github.com/BayesOptApp/Topology_Optimization
I will write the things I changed later...

The system is setup as follows (I will write a proper readme later):

a ProblemTO contains 3 main elements:
- Topology: a data container to store the geometry related aspects
- Parameterization: a way of going from a design vector to geometry
- Model: the physics model

For an evaluation of the design vector of x:
1. The Parameterization updates the geometry of Topology
2. The Topology gets passed to the physics Model which updates its state
3. An objective is calculated from the updated physics Model

When writing a custom Parameterization you essentially only have to implement:
```
Parameterization.compute_geometry(self, x: np.ndarray) -> MultiPolygon : ...
```
which calculates the geomtery in the form of Polygons from the design vector x.
Rasterization towards a binary material distribution is then done automatically.
Also this Polygon representation allows the constraints to be general; i.e. they 
should work on any arbitrary parameterization as long as it returns Polygons.

For now I have three different parameterizations to test the system:
- Original MMCs
- Regular rectangles
- Capsules (rectangles with a circular cap)

# TO-DO
- Comment the code
- Write a proper readme
- Make use of the defined domain everywhere instead of hardcoding values
- Make a logging system that logs all settings/state of the procedure, including the software version
- Add live plotting again
- Make into python package
- Add the C version of fill_matrix to go faster
- Implement the geo from mask to see how discrete will do
- Adding the notebooks with example calculations