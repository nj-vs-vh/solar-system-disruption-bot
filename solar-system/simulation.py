from dataclasses import dataclass
import numpy as np

from numpy.typing import NDArray


Vector = NDArray[np.float64]


@dataclass
class Body:
    name: str
    color: str
    m: float  # in M ☉
    r_0: Vector  # in a.u.
    v_0: Vector  # 


SOLAR_SYSTEM = [
    Body(
        "Sun",
        "#ffed69",
        m=1,
        r_0=np.array([0.0, 0.0]),
        v_0=np.array([0.0, 0.0]),
    )
]
