import numpy as np

np.random.seed(0)

N = 100000
dim = 1000

# leakage model
y = np.random.randint(0, 9, N).astype(np.float64)

# trace
x = np.random.normal(0, 1, (N, dim)).astype(np.float32)

# サンプル点123にだけ相関を埋め込む
x[:, 123] += 0.5 * y

# 真値
print(np.corrcoef(x[:, 123], y)[0, 1])


from scaueclib import OnlineCPA

cpa = OnlineCPA(dim)

for i in range(N):
    cpa.update(x[i], float(y[i]))

corr = cpa.get_pearson_corr()
argmax_idx = np.argmax(abs(corr))
print(corr[argmax_idx-5:argmax_idx+5])
