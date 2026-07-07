from scaueclib import OnlineBivariateTVLA

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(0)

N = 100_000
dim = 200

target_1 = 12
target_2 = 21

tvla = OnlineBivariateTVLA(dim)

z = np.random.normal(0.0, 1.0, size=N).astype(np.float32)
data = np.random.normal(0.0, 1.0, size=(N, dim)).astype(np.float32)

coins = np.random.randint(0, 2, size=N)
mask = coins == 0
data[mask, target_1] += 3.0 * z[mask]
data[mask, target_2] -= 3.0 * z[mask]

for i in range(N):
    tvla.update(data[i], coins[i])

t = tvla.get_tvalue()


plt.imshow(t)
plt.colorbar()
plt.show()

