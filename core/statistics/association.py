import numpy as np

def safe_ll_term(O, E):
    res = O * np.log(O / E)
    return np.where((O > 0) & (E > 0), res, 0.0)

def vec_sig(x):
    if x >= 15.13: return '*** (p<0.001)'
    elif x >= 10.83: return '** (p<0.01)'
    elif x >= 3.84: return ' * (p<0.05)'
    else: return 'ns'
