from .mmc import MMCAxiSymmetricConfig
from .lame_curve import LameCurves

class Ellipses(LameCurves):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, m=2)