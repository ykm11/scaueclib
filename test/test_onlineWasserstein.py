import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scaueclib import OnlineWasserstein

np.random.seed(0)

N = 100_00
dim = 1000
target0 = 123
target1 = 876
adc_bit = 8

wd_vec = OnlineWasserstein(dim, adc_bit)

x0 = np.random.normal(1 << (adc_bit - 1), 5, size=(N // 2, dim))
x1 = np.random.normal(1 << (adc_bit - 1), 5, size=(N // 2, dim))

x0 = np.round(x0).astype(np.uint32)
x1 = np.round(x1).astype(np.uint32)

x1[:, target0] += 2
x1[:, target1] -= 3

for i in range(N // 2):
    wd_vec.update(x0[i], 0)
    wd_vec.update(x1[i], 1)

wd = wd_vec.get_distance()

sci_wd = np.zeros(dim)
for i in range(dim):
    sci_wd[i] = wasserstein_distance(x0[:,i], x1[:,i])

print(wd[:10])
print(sci_wd[:10])
#plt.plot(wd)
#plt.show()

