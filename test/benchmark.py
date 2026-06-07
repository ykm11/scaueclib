import numpy as np
from scaueclib import OnlineCPA
import time
from scalib.attacks import OnlineCorrVector

np.random.seed(0)

N = 100_000
dim = 1_000
target = 123

y = np.random.randint(0, 9, size=N).astype(np.float64)
x = np.random.normal(0, 1, size=(N, dim)).astype(np.float32)

# 相関を埋め込む
x[:, target] += 0.5 * y.astype(np.float32)

np.random.seed(0)

def bench_cpp():
    cpa = OnlineCPA(dim)

    t0 = time.perf_counter()
    for i in range(N):
        cpa.update(x[i], float(y[i]))
    t1 = time.perf_counter()

    corr = cpa.get_pearson_corr()

    return t1 - t0, corr


def bench_numpy_batch():
    t0 = time.perf_counter()

    cpa = OnlineCorrVector(dim)
    for i in range(N):
        cpa.update(x[i], float(y[i]))

    corr = cpa.pearson()

    t1 = time.perf_counter()
    return t1 - t0, corr

# warmup
bench_cpp()

cpp_update_t, corr_cpp = bench_cpp()
np_t, corr_np = bench_numpy_batch()


print(f"C++ update      : {cpp_update_t:.6f} sec")
print(f"NumPy update     : {np_t:.6f} sec")
print(f"speedup         : {np_t / cpp_update_t:.6f}")

print("cpp argmax:", np.argmax(np.abs(corr_cpp)))
print("numpy argmax:", np.argmax(np.abs(corr_np)))
print("target:", target)

print("cpp corr[target]:", corr_cpp[target])
print("numpy corr[target]:", corr_np[target])

print("max abs diff:", np.max(np.abs(corr_cpp - corr_np)))
print("allclose:", np.allclose(corr_cpp, corr_np, rtol=1e-5, atol=1e-5))
