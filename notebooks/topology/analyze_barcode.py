import numpy as np
import numpy.typing as npt

def num_branches(barcode: npt.ArrayLike) -> int:
    return len(barcode)

def total_branch_length(barcode: npt.ArrayLike) -> int:
    total = 0
    for bar in barcode:
        total += abs(bar[0]-bar[1])
    return total

def average_branch_length(barcode: npt.ArrayLike) -> float:
    return (total_branch_length(barcode)/num_branches(barcode))