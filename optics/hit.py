from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Hit:
    point: np.ndarray
    t: float

    u: float | None = None
    v: float | None = None