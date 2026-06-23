from scaueclib import OnlineBivariate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(0)

N = 100_000

def low_dim():
    dim = 200
    target_1 = 12
    target_2 = 21

    bivariate = OnlineBivariate(dim)

    z = np.random.normal(0.0, 1.0, size=N).astype(np.float32)
    data = np.random.normal(0.0, 1.0, size=(N, dim)).astype(np.float32)
    data[:, target_1] += 3.0 * z
    data[:, target_2] += 3.0 * z

    for i in range(N):
        bivariate.update(data[i])

    cov_ = bivariate.get_cov()

    plt.imshow(cov_)
    plt.colorbar()
    plt.show()

def high_dim():
    dim = 1500
    target_1 = 12
    target_2 = 21

    bivariate = OnlineBivariate(dim)

    for i in tqdm(range(N)):
        data = np.random.normal(0.0, 1.0, dim)

        z = np.random.normal(0.0, 1.0)
        data[target_1] += 7.0 * z
        data[target_2] += 7.0 * z

        bivariate.update(data)

    cov_ = bivariate.get_cov() / 50

    plt.imshow(cov_)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    #low_dim()
    high_dim()
