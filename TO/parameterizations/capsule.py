import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate, scale

from .mmc import MMC, MMCAngularConfig, MMCEndpointsConfig

class Capsules(MMC) :
    def compute_base_polygon(self) -> Polygon :
        # NOTE: the base shape of the capsule is a semi-circle
        # on `compute_polygon` we then first move this semicircle to the endpoint mirror and connect it to the other side
        return Polygon(np.c_[np.cos(t := np.linspace(-np.pi/2, np.pi/2, self.n_samples//2)), np.sin(t)])

    def compute_polygon(self, config: MMCAngularConfig) -> Polygon :
        r = min(config.rx, config.ry)
        # move the semi-circle to the capsules endpoint
        poly: Polygon = translate(scale(self.base_polygon, r, r, origin=(0,0)), config.rx-r, 0).union(
            box(0,-r, config.rx-r, r) # adding the straight middle portion of the capsule
        ).buffer(1e-3)
        # mirroring the half capsule on the y-axis to create the full geo
        poly = poly.union(scale(poly, -1, 1, origin=(0,0))).buffer(1e-3)

        return translate(rotate(
            poly, config.theta, use_radians=True # rotate
        ), config.x, config.y) # translate