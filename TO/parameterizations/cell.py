import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.affinity import scale, translate
from shapely import unary_union

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from TO.parameterization import Parameterization
from TO.topology import Topology

unit_circle = lambda n_samples : Polygon(np.c_[np.cos(t := np.linspace(0,2*np.pi,n_samples)), np.sin(t)])
unit_hexagon = lambda : unit_circle(7)
unit_square = lambda : box(-1,-1,1,1)

class GridSampler(ABC):
    @abstractmethod
    def compute(self, 
        topology: Topology,
        symmetry_x: bool, symmetry_y: bool,
        cell_size_x: float, cell_size_y: float
    ) -> Tuple[np.ndarray, np.ndarray]: ...

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
    
class HexGrid(GridSampler):
    @staticmethod
    def compute( 
        topology: Topology,
        symmetry_x: bool, symmetry_y: bool,
        cell_size_x: float, cell_size_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        (X, Y) = RectangularGrid.compute(topology, symmetry_x, symmetry_y, 3*cell_size_x/4, np.sqrt(3)*cell_size_y/2)
        Y[0::2] += np.sqrt(3)/8*cell_size_y
        Y[1::2] -= np.sqrt(3)/8*cell_size_y
        return (X, Y)

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
        # pre-shift all geometry to cell centers
        self.cells: np.ndarray[Polygon] = np.array([
            translate(self.cell, x, y) for (x, y) in zip(self.X.flatten(), self.Y.flatten())
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