import numpy as np
import matplotlib.pyplot as plt
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
x1, x2 = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x1, x2, '.')
plt.axis('equal')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('scatter_of_x1_and_x2_d.png')