import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.affinity import rotate, scale, translate
from shapely import unary_union

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Tuple

from TO.core import Parameterization, Topology

unit_circle = lambda n_samples : Polygon(np.c_[np.cos(t := np.linspace(0,2*np.pi,n_samples)), np.sin(t)])
unit_hexagon = lambda : unit_circle(7)
unit_square = lambda : box(-1,-1,1,1)
unit_triangle = lambda : Polygon([(-1,-1), (1,-1), (0,1)])
infill = lambda poly, fac : poly.difference(scale(poly, 1-fac, 1-fac))

class GridSampler(ABC):
    @abstractmethod
    def compute(self, 
        topology: Topology,
        symmetry_x: bool, symmetry_y: bool,
        cell_size_x: float, cell_size_y: float
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def compute_cell(self, cell: Polygon, x: float, y: float, i: int, j:int) -> Polygon :
        # transforming the unit_cell to grid-based cell, for regular grids just translation is enough
        return translate(cell, x, y)

class RectangularGrid(GridSampler):
    @staticmethod
    def compute( 
        topology: Topology,
        symmetry_x: bool, symmetry_y: bool,
        cell_size_x: float, cell_size_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        domain_size_x = topology.domain_size_x/2 if symmetry_x else topology.domain_size_x
        domain_size_y = topology.domain_size_y/2 if symmetry_y else topology.domain_size_y
        
        (X, Y) = np.mgrid[
            cell_size_x/2:domain_size_x + cell_size_x/2:cell_size_x,
            cell_size_y/2:domain_size_y + cell_size_y/2:cell_size_y,
        ]
        X += domain_size_x/2 - X.mean()
        Y += domain_size_y/2 - Y.mean()
        return (X, Y)
    
@dataclass
class AlternatingGrid(GridSampler, ABC):
    horizontal: bool

    @abstractmethod
    def delta_x_horizontal(self, cell_size_x: float, cell_size_y: float) -> float : ...

    @abstractmethod
    def delta_y_horizontal(self, cell_size_x: float, cell_size_y: float) -> float : ...

    def compute(self,
        topology: Topology,
        symmetry_x: bool, symmetry_y: bool,
        cell_size_x: float, cell_size_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        delta_x = self.delta_x_horizontal(cell_size_x, cell_size_y)
        delta_y = self.delta_y_horizontal(cell_size_x, cell_size_y)
        if not(self.horizontal) : (delta_x, delta_y) = (delta_y, delta_x)
        
        (X, Y) = RectangularGrid.compute(topology, symmetry_x, symmetry_y, delta_x, delta_y)
        if (self.horizontal) :
            X[:, 0::2] += delta_x/4
            X[:, 1::2] -= delta_x/4
        else:
            Y[0::2] += delta_y/4
            Y[1::2] -= delta_y/4
        return (X, Y)

class HexGrid(AlternatingGrid):
    def delta_x_horizontal(self, cell_size_x: float, cell_size_y: float) -> float :
        return np.sqrt(3)*cell_size_y/2
    
    def delta_y_horizontal(self, cell_size_x: float, cell_size_y: float) -> float :
        return 3*cell_size_x/4
    
    def compute_cell(self, cell: Polygon, x: float, y: float, i: int, j: int):
        if (self.horizontal) :
            return translate(rotate(cell, 90), x, y)
        return super().compute_cell(cell, x, y, i, j)

    
class DiamondGrid(AlternatingGrid):
    def delta_x_horizontal(self, cell_size_x: float, cell_size_y: float) -> float :
        return np.hypot(cell_size_x, cell_size_y)
    
    def delta_y_horizontal(self, cell_size_x: float, cell_size_y: float) -> float :
        delta_x = self.delta_x_horizontal(cell_size_x, cell_size_y)
        return np.sqrt(cell_size_x**2 - (delta_x/2)**2)
    
    def compute_cell(self, cell: Polygon, x: float, y: float, i: int, j: int):
        return translate(rotate(cell, 45), x, y)
    
@dataclass
class TriGrid(GridSampler):
    horizontal: bool

    def compute(self,
        topology: Topology,
        symmetry_x: bool, symmetry_y: bool,
        cell_size_x: float, cell_size_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        topology_extended = replace(topology, domain=box(0, 0, 
            topology.domain_size_x + (cell_size_x/2 if self.horizontal else 0), 
            topology.domain_size_y + (cell_size_y/2 if not(self.horizontal) else 0)
        ))
        (X, Y) = RectangularGrid.compute(
            topology_extended, symmetry_x, symmetry_y, 
            cell_size_x/2 if (self.horizontal) else cell_size_x, 
            cell_size_y if (self.horizontal) else cell_size_y/2
        )
        (size_x, size_y) = topology.domain_size
        if (symmetry_x) : size_x /= 2
        if (symmetry_y) : size_y /= 2
        if not(self.horizontal) : Y -= Y.max() - size_y
        return (X, Y)
    
    def compute_cell(self, cell: Polygon, x: float, y: float, i: int, j:int) -> Polygon :
        if (self.horizontal):
            cell_oriented = scale(cell, 1, -1 if (i%2 ^ j%2) else 1, origin=(0,0))
        else:
            cell_oriented = rotate(cell, 90, origin=(0,0))
            cell_oriented = scale(cell_oriented, -1 if (i%2 ^ j%2) else 1, 1, origin=(0,0))
        return translate(cell_oriented, x, y)

@dataclass
class Cells(Parameterization, ABC):
    sampler: GridSampler
    unit_cell: Polygon
    cell_size_x: float
    cell_size_y: float

    def __post_init__(self):
        self.cell: Polygon = scale(self.unit_cell, self.cell_size_x/2, self.cell_size_y/2)
        (self.X, self.Y) = self.sampler.compute(self.topology, self.symmetry_x, self.symmetry_y, self.cell_size_x, self.cell_size_y)
        self.dimension = self.X.size
        # calculate the geo of the tiling
        self.cells: np.ndarray[Polygon] = np.array([
            self.sampler.compute_cell(self.cell, self.X[i,j], self.Y[i,j], i, j) for i in range(len(self.X)) for j in range(len(self.X[0]))
        ])

    @abstractmethod
    def compute_geometry(self, x: np.ndarray) -> MultiPolygon: ...

class BinaryCells(Cells):
    def compute_geometry(self, x: np.ndarray) -> MultiPolygon:
        return unary_union(self.cells[x > 0.5]).buffer(1e-2)

class InfillCells(Cells):
    def compute_geometry(self, x: np.ndarray) -> MultiPolygon:
        # `xi` denotes the percentage of infill as with the MMC parameterizations
        r = min(self.cell_size_x, self.cell_size_y)/2
        return unary_union([
            cell.difference(cell.buffer(-r*xi)).buffer(1e-2)
            for (xi, cell) in zip(x, self.cells)
        ])