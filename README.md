## Side-channel analysis and assessments tools


### Build 

```
$ mkdir build; cd build
$ cmake ..
$ make
$ cp *.so /path/to/your_working_dir
```

### Classes

- OnlineCPA (computes Pearson's correlation coefficients at each point)
- OnlineTVLA (first-order t-test)
- OnlineBivariateTVLA
- OnlineWasserstain (quantitative assessment with the Wasserstein distance)


### Usage

For correlation power analysis.
```Python
from scaueclib import OnlineCPA

onlineVec = OnlineCPA(dim)

for i in range(N):
  onlineVec.update(trace[i], label[i]) # trace = np.array, label = integer
onlineVec.get_pearson_corr()
```


For the first-order t-test.
```Python
from scaueclib import OnlineTVLA

onlineVec = OnlineTVLA(dim)

for i in range(N):
  onlineVec.update(trace[i], coin[i]) # trace = np.array, coin = 0 or 1
onlineVec.get_tvalue()
```

For a leakage assessment with the Wasserstein distance.
```Python
from scaueclib import OnlineWasserstain
onlineVec = OnlineWasserstein(dim, adc_bit)

for i in range(N):
  onlineVec.update(trace[i], coin[i]) # trace = np.array, coin = 0 or 1
onlineVec.get_distance()
```


#### Contact  
ykm11 (haruka.hirata11 [at] gmail.com)
