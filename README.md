# Code for Paper "Preserving Full Spectrum Information in Imaging Mass Spectrometry Data Reduction"

BioRxiv: [https://www.biorxiv.org/content/10.1101/2024.09.30.614425v1](https://www.biorxiv.org/content/10.1101/2024.09.30.614425v1)

In-house Dependencies: imzy and UniPy


## Download and Unpack Data
DOI: [https://doi.org/10.4121/a6efd47a-b4ec-493e-a742-70e8a369f788](https://doi.org/10.4121/a6efd47a-b4ec-493e-a742-70e8a369f788)

Data: [https://surfdrive.surf.nl/files/index.php/s/TLPccCCAP7Uat5r](https://surfdrive.surf.nl/files/index.php/s/TLPccCCAP7Uat5r)

There are 2 data sets: (1) FT-ICR (fticr_data.tar.gz), (2) qTOF (all other files, i.e. xaa-xaz). Below are the steps explained for Linux/Mac unpacking of the data.

### FT-ICR
Unpack data
```
tar -xfz fticr_data.tar.gz -C fticr_data
```

### qTOF
Given the 100+ GB data set size of the qTOF in compressed format, we need to gather all differnt split files (4GB each) to unpack the whole folder.
```
cat xa* | tar xfz -
```

## From .d to NumPy Array / SciPy Sparse Array
Use our favorite toolbox for this, e.g. [https://github.com/vandeplaslab/imzy](imzy)
The code below provides a simple sample of code when B fits into memory (in a dense format).

```python
import numpy as np
import scipy.sparse as scis
from imzy import get_reader

path = "path/to/file"
reader = get_reader(path)
f = reader[0]
out = np.zeros((reader.n_mz_bins, len(reader.framelist)), dtype=f.dtype)

for k, i in enumerate(reader.framelist):
    f = reader[i]._init_csc()
    out[f.indices, k] = f.data    
    
B = scis.csc_matrix(out, dtype='float32', copy=False)
```

When B (in a dense format) is too large to fit in memory, one can use the following
```
import numpy as np
import scipy.sparse as scis
from imzy import get_reader

path = "path/to/file"
reader = get_reader(path)
vec_size = 0
for i in reader.framelist:
    vec_size += reader.bo[i][1][5]

b_data = np.zeros((vec_size,), dtype='float32')
b_indices = np.zeros((vec_size,), dtype='int32')
b_indptr = np.zeros((len(iter_var)+1), dtype='int64')

def read_in(q, e, i):
    a = reader[i]
    lsize = a.data.shape[0]
    b_data[q:q+lsize] = a.data
    b_indices[q:q+lsize] = a.indices
    b_indptr[e+1] = b_indptr[e]+a.indptr[1]

    return q+lsize

q = 0
for e, i in enumerate(iter_var):
    q = read_in(q, e, i)
    
    if e%1_000 == 0: # Print out
        print(np.round(e/len(iter_var)*100, 2), '% in', np.round(t.time()-tic), 's', end='\r')

b_data = b_data[:q]
b_indices = b_indices[:q]

B = scis.csc_matrix((b_data, b_indices, b_indptr), copy=False)
```


## Download and Installing Package
% Under construction %
```
get | cd intall
```

## Applying Methods
### Preprocessing
We apply a 5-95% Total Ion Current Count (TIC) Normalization


### SVT
```Python
from full_profile import svt
inst = svt.svt(100, r, tau_factor=1e-2, delta=1, method='scipy_gesdd', dense=True, verbose=True)
inst.run(B)
```

### FPC
```Python
from full_profile import fpc
inst = fpc.fpc(maxiter=100, tau_factor=1e-3, delta=1.4, method='arpack', verbose=True)
inst.run(B)
```

### DFC
```Python
from full_profile import dfc
inst = dfc.dfc(maxiter=100, tau_factor=1e-3, delta=1.4, method='arpack', verbose=True)
inst.run(B)
```
