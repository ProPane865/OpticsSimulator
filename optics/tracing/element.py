"""Closed refractive element: geometry and cached sidewall construction."""

from typing import Any

import numpy as np

from .. import constants as _c
from ..primitive import LensSidewall, Material, Surface, Transform

_SIDEWALL_UNSET = object()


class RefractiveElement:
    surface1: Surface
    surface2: Surface
    material: Material

    def __init__(self, surface1: Surface, surface2: Surface, material: Material) -> None:
        self.surface1 = surface1
        self.surface2 = surface2
        self.material = material
        self._sidewall: Any = _SIDEWALL_UNSET

    @property
    def transform(self) -> Transform:
        return self.surface1.transform

    @property
    def sidewall(self) -> LensSidewall | None:
        if self._sidewall is _SIDEWALL_UNSET:
            if (
                self.surface1.aperture is None
                or self.surface2.aperture is None
            ):
                self._sidewall = None
            else:
                front = rear = None

                try:
                    front = self.surface1.aperture.boundary(self.surface1)
                    rear = self.surface2.aperture.boundary(self.surface2)
                except ValueError:
                    front = rear = None

                if (
                    front is not None
                    and rear is not None
                    and front.shape == rear.shape
                    and not np.allclose(front, rear, atol=_c.SIDEWALL_MATCH_ATOL)
                ):
                    self._sidewall = LensSidewall(front, rear)
                else:
                    self._sidewall = None
        return self._sidewall
