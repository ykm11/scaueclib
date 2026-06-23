from scaueclib import OnlineBivariate
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)

N = 100_000
dim = 200
target_1 = 12
target_2 = 21

bivariate = OnlineBivariate(dim)

data = np.random.normal(0.0, 1.0, size=(N, dim)).astype(np.float32)

z = np.random.normal(0.0, 1.0, size=N).astype(np.float32)

data[:, target_1] += 3.0 * z
data[:, target_2] += 3.0 * z

for i in range(N):
    bivariate.update(data[i])

cov_ = bivariate.get_cov()

plt.imshow(cov_)
plt.colorbar()
plt.show()
