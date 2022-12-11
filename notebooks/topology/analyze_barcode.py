import numpy as np
import numpy.typing as npt

def num_branches(barcode: npt.ArrayLike) -> int:
    return len(barcode)

def total_branch_length(barcode: npt.ArrayLike) -> float:
    total = 0
    for i in range(0, len(barcode)-1):
    
        
        bar_1 = barcode[i][0]
        bar_2 = barcode[i][1]
                
        total += abs(bar_1-bar_2)
        
    return total

def average_branch_length(barcode: npt.ArrayLike) -> float:
    return (total_branch_length(barcode)/num_branches(barcode))