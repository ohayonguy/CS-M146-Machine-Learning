import numpy as np
mat = [[1, 0], [1, 3]]
w, v = np.linalg.eig(mat)
print(w)
print(v)