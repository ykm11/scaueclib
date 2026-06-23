### Tools for side-channel analysis


### Build 

```
$ mkdir build; cd build
$ cmake ..
$ make
```

### Usage

```Python
from scaueclib import OnlineCPA

onlineVec = OnlineCPA(dim)
onlineVec.update(trace, label)
onlineVec.get_pearson_corr()
```

#### Contact  
ykm11
