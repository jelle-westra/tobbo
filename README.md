# tobbo
A package for Topology Optimization (TO) using Black-Box Optimization (BBO) methods: *tobbo*. TO problems are concerned with finding optimal material distributions within some spacial domain given an objective. Conventially these problems are solved using special purpose gradient-based optimization methods.

BBO methods provide a general framework for solving TO problems for arbitrary objectives, geometry parameterization, and physics. That is, there is no need for devising gradient-based procedures. For example in the case of nonlinear dynamics, such as the simulation of crash impacts, gradient-based optimization methods are infeasible; BBO uses only a trial $\mathbf{x}$ and observation $f_\text{obj}(\mathbf{x})$ scheme.

![demo](./package/assets/demo.svg)

-------

This repo is two-fold in structure as it houses experiment scripts (Case-Study), as used for a master thesis project, alongside a standalone package (tobbo) which evolved throughout the project (Package).

**Contents**
- Case-Study
- Package
    - Installation
    - Getting Started
    - tobbo Package Structure
- Team


## Case-Study
In our study we consider minimization of compliance for static loading of a horizontal cantilever (2D, plane-stress), under constraint of using only 50% of the design domain as material, the rest is void. This is known as a classic *structural* TO problem. The schematic in the introduction above shows evolution of such a problem using the CMA-ES BBO method and four (symmterized) *Curved Moving Morphable Component* (Curved MMC) beams.

Such a Curved MMC has 10 Degrees of Freedom (DOF), two endpoints, a thickness, and five additional deformation parameters. Note the design domain has 100x50 elements meaning that the problem originally has 5000 DOF. By using parameterization like this we effectively reduce the DOF by constraining the desings, in this case to be only (curved) beam structures. This results in problems like this to be tractable for BBO methods given the reduced effective dimension.

An other parameterization could be to only use non-deformed straight beams: 5 DOF per beam. These different parameterizations lead to different characteristics of the fitness landscapes. Different BBO algorithms are designed to exploit different landscape features; unfortunately there is no jack of all trades. 

*Our goal is to reseach the interplay between parameterizations and  BBO procedures.*

![studied parameterizations](./package/assets/parameterizations.png)

### Constrained Problem Definition
Volume is penalized based on the exceeded volume, and connectivity is penalized based on the least distance required to connect the disconnected design. That is, the design needs to be connected to the wall and loading point, and to the design itself, such that from anywhere on the structure we can move to any other point on the structure.

The infeasible region creates an artificial valley towards feasible which is relative to the simulation cheaply computed since it only consists of simple geometrical checks.

Given we only have a limited (expensive) simulation budget, *the optimizer is free to make as many infeasible calls as it want.* The schematic in the introduction shows infeasible designs in orange and feasible designs that went through the actual simulation in blue.

### Main Findings
Our results show that parameterization has a dominant influence on optimization performance in constrained TO. When the chosen parameterization effectively captured the structural characteristics of the problem, such as the Curved MMCs, all algorithms reliably converged to high-quality designs. 

Conversely, weak or overly restrictive parameterizations such as Honeycomb parameterization create a strong reliance on Algorithm Selection for competitive performance.

We demonstrate that well-chosen design representations are key to achieving robust and efficient optimization outcomes in a constrained real-world engineering problem, independent of using a simple or a state-of-the-art optimizer.

![20D-boxplots](./package/assets/20D-boxplots.svg)

### Reproducibility
Our experiments can be reproduced using `main.py`, for the three dimensions: **10D**, **20D**, and **50D** we create a problem definition using the three different parameterization above: **Honeycomb**, **MMC**, and **Curved MMC** ($3\times3$). For each of these 9 settings we study the optimization procedure for three different BBO algorithms:
- **DE** a classical Differential Evolution algorithm driven by population recombination and mutation.
- **CMA-ES** a robust, evolution-based optimizer for continuous search spaces that adapts its sampling distribution via covariance matrix updates.
- **HEBO** a state-of-the-art Bayesian Optimization method designed for sample efficiency on low-budget continuous problems.

## Package
### Installation

After review the package will be added to pypi, for now

```
# assuming python>=3.11 
pip install ./package/
```

### Getting Started

Tutorial can be found tutorial.ipynb

Minimal Example:
```[language=python]
from tobbo.core import Topology, OptimizationMethod, run_experiment
from tobbo.parameterizations.mmc import Capsules, MMCEndpointsConfig
from tobbo.problems.cantilever import create_horizontal_cantilever_problem
```

Setting up a TO experiment with tobbo is four-fold:

1) **Domain** definition; setting up the Topology container.
2) **Parameterization** definition; a mapping for generating geometry from design vectors.
3) **Problem** definition; physics model, constraints, objective.
3) **Optimization**; using a BBO method for minimizing the objective.


#### Problem Domain 
Topology is used as data container for domain configuration, and holding the geometry in polygonal- and discretized binary mask form. This binary mask is used to activate material in the mesh of the physics model. The polygonal representation is used for calculation of the constraint functions. 
```
topology: Topology = Topology(
    continuous=False,
    domain_size=(100, 50),
    density=1.
)
```
We use the rasterized polygonal geometry for calculating constraint (`continuous=False`), such that it is one-to-one with the physics mesh. Sometimes the underlying continuous geometry can be barely connected, whereas after rasterization this is not the case, this results to spurious calls to the physics simulation.

#### Parameterization
A Parameterization generates the geometry based on the parameterized design vector.
```
parameterization = Capsules(
    topology,
    symmetry_x=False,
    symmetry_y=True,
    representation=MMCEndpointsConfig, 
    n_components=1,
    deformer=None,
    n_samples=1000
)
```
We generate a single capsule-shaped beam (`n_components=1`), mirror it among the horizontal centerline of the domain (`symmetry_y=True`), represented by two endpoints and a thickness (`MMCEndPointsConfig`, 5dof). 

![parameterization-rasterization](package/assets/rasterization.svg)

> **NOTE** Contrary, the (`MMCAngularConfig`, 5dof) representation uses a centerpoint, width, thickness, and angle. These differences in representation reflect in completely different fitness landscapes, although both can produce exactly idential geometries. 
>
> This interplay between problem formulation (parameterization) and effect on optimization procedures is exactly where our focus lies.

#### Problem
```
# the horizontal cantilever loading problem
problem = create_horizontal_cantilever_problem(topology, parameterization)
```
#### Optimization

```
run_experiment(problem, 
    budget=100, 
    seed=1, 
    name='minimal-example', 
    method=OptimizationMethod.CMAES
)
```

### `tobbo` Package Structure

```
./package/ the standalone tobbo package
    tobbo/
        core/
        models/
        parameterizations/
        problems/
```


## Authors
After review the authors will be added.

...