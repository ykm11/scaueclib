import numpy as np
from scaueclib import OnlineTVLA

np.random.seed(0)

N = 100_000
dim = 1000
target = 123

tvla = OnlineTVLA(dim)

x0 = np.random.normal(0.0, 1.0, size=(N // 2, dim)).astype(np.float32)
x1 = np.random.normal(0.0, 1.0, size=(N // 2, dim)).astype(np.float32)

x1[:, target] += 0.1

for i in range(N // 2):
    tvla.update(x0[i], 0)
    tvla.update(x1[i], 1)

t = tvla.get_tvalue()

print("argmax abs t:", np.argmax(np.abs(t)))
print("target:", target)
print("t[target]:", t[target])
print("max abs t:", np.max(np.abs(t)))
