from abc import ABC, abstractmethod
from .topology import Topology

class Model(ABC):
    @abstractmethod
    def update(self, topology: Topology) : ...