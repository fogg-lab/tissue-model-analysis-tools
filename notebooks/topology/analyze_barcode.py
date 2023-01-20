import numpy as np
import numpy.typing as npt

def num_branches(barcode: npt.ArrayLike) -> int:
    return len(barcode)

def total_branch_length(barcode: npt.ArrayLike, minimum_branch_length) -> float:
    total = 0
    for i in range(0, len(barcode)-1):
    
        
        bar_1 = barcode[i][0]
        bar_2 = barcode[i][1]

        # Don't count branches of infinite length
        if bar_1 == float("inf") or bar_2 == float("inf"):
            continue
                
        branch_length = abs(bar_1-bar_2)

        if branch_length > minimum_branch_length:
            total += branch_length
        
    return total

def average_branch_length(barcode: npt.ArrayLike, minimum_branch_length=0) -> float:
    return (total_branch_length(barcode, minimum_branch_length)/num_branches(barcode))

def pixels_to_microns(pixels: float, im_width: int, well_width: float) -> float:
    """
    Convert pixels to microns in specified resolution.
    """
    return (well_width / im_width) * pixels