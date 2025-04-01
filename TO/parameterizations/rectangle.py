import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate, scale

from .mmc import MMC, MMCAngularConfig

class Rectangles(MMC):
    def compute_base_polygon(self) -> Polygon :
        assert (self.n_samples >= 4), ''
        return box(-1, -1, 1, 1) # independent on the number of samples
    
    def compute_polygon(self, config: MMCAngularConfig) -> Polygon :
        return translate(rotate(scale(
                self.base_polygon, config.rx, config.ry # scale
            ), config.theta, use_radians=True # rotate
        ), config.x, config.y) # translate