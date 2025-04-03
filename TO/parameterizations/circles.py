from .ellipses import Ellipses
from .mmc import MMCAxiSymmetricConfig

class Circles(Ellipses):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, representation=MMCAxiSymmetricConfig)