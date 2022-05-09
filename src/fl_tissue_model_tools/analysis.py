import numpy as np
import numpy.typing as npt


def compute_area_prop(img: npt.NDArray, ref_area: int, min_val: float=0) -> float:
    return np.sum(img > 0) / ref_area