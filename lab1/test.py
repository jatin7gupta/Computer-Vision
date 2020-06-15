import numpy as np
x =  np.arange(2, 11).reshape(3,3)
y =  np.arange(2, 11).reshape(3,3)
z = x*y
print(np.sum(z))
