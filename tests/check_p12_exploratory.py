import numpy as np
from scripts.p12_exploratory import fixed_blend
# Deliberate tie: ranks must average and row order must not affect predictions.
a=np.tile(np.array([.1,.1,.9])[:,None],(1,12))
b=np.tile(np.array([.9,.3,.1])[:,None],(1,12))
expected=np.tile(np.array([(.8*1.5+.2*3)/3,(.8*1.5+.2*2)/3,(.8*3+.2*1)/3])[:,None],(1,12))
assert np.allclose(fixed_blend(a,b),expected)
assert np.allclose(fixed_blend(a[::-1],b[::-1]),expected[::-1])
try:fixed_blend(a,b[:2])
except ValueError:pass
else:raise AssertionError('Coverage mismatch accepted')
print('Fixed blend tie, alignment and shape checks passed')

x=np.tile(np.arange(58)[:,None],(1,12))
assert fixed_blend(x,x).max() <= 1.
