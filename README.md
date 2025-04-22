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

- Constraints benchmarking
- The order of constraints matter, what are going to do with this?
- ELA
- Can we use the ELA as warm-start procedure of the optimization process itself -> starting with a good initial population instead of random sampling one. 
- What is the end goal here, just correlate ELA features with performance? In a real-life problem you would likely first spend some budget on ELA to select an algorithm and then to do an/some optimization run? Then we might as well use the evaluations some how in the optimizer -> easy in BO.