import math
import numpy as np
import pandas as pd

def pmw_to_zipf(pmw):
    """
    Convert frequency per million (PMW) to Zipf scale.
    Formula: Zipf = log10(PMW) + 3
    """
    if pmw <= 0:
        return np.nan
    return math.log10(pmw) + 3

def zipf_to_band(zipf):
    """
    Assign 1–5 Zipf band based on score:
    Band 1: 7.0–7.9
    Band 2: 6.0–6.9
    Band 3: 5.0–5.9
    Band 4: 4.0–4.9
    Band 5: 1.0–3.9
    """
    if pd.isna(zipf):
        return np.nan
    elif zipf >= 7.0:
        return 1
    elif zipf >= 6.0:
        return 2
    elif zipf >= 5.0:
        return 3
    elif zipf >= 4.0:
        return 4
    else: 
        return 5
