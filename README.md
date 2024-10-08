# Code for Paper "Preserving Full Spectrum Information in Imaging Mass Spectrometry Data Reduction"

BioRxiv: [https://www.biorxiv.org/content/10.1101/2024.09.30.614425v1](https://www.biorxiv.org/content/10.1101/2024.09.30.614425v1)

In-house dependencies: [https://github.com/vandeplaslab/full_profile/blob/main/imzy](imzy) and [https://github.com/vandeplaslab/full_profile/blob/main/unipy](UniPy)

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
Use your favorite toolbox for this, e.g. [https://github.com/vandeplaslab/imzy](imzy)
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
```python
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
This package depends heavily on [https://github.com/vandeplaslab/unipy](UniPy) make sure to install Unipy before this package. Then, get the latest package version of full_profile, e.g. through GitHub CLI (https://cli.github.com/)
```
gh repo clone vandeplaslab/full_profile
```
Install package (e.g. through pip) and get required dependencies (first ```cd``` into the full_profile folder)
```
pip install .
```

## Applying Methods
Once the package is installed we can apply the paper methods onto data.

### Preprocessing
For our purposes, we first applied a 5-95% Total Ion Current Count (TIC) normalization (i.e. ```C```) on the raw data (see Supplementary Materials of paper). The method can be found here: [https://github.com/vandeplaslab/imzy/blob/add-norms/src/imzy/_normalizations/_extract.py](https://github.com/vandeplaslab/imzy/blob/add-norms/src/imzy/_normalizations/_extract.py) or we can apply it directly onto our sparse data by
```python
def calculate_normalizations_optimized(spectrum: np.ndarray) -> np.ndarray:
    """Calculate various normalizations, optimized version.

    This function expects a float32 spectrum.
    """
    # Filter positive values once and reuse
    positive_spectrum = spectrum[spectrum > 0]

    # Calculating quantiles once for all needed
    if positive_spectrum.size > 1:
        q95, q90, q10, q5 = np.nanquantile(positive_spectrum, [0.95, 0.9, 0.1, 0.05])
    else:
        q95, q90, q10, q5 = 0, 0, 0, 0

    # Using logical indexing with boolean arrays might be faster due to numba optimization
    condition_q95 = spectrum < q95
    condition_q5 = spectrum > q5

    return np.sum(spectrum[condition_q5 & condition_q95])  # 5-95% TIC

C = np.zeros((B.shape[1],1))
for i in range(B.shape[1]):
    C[i] = calculate_normalizations_optimized(B[:,i].toarray())
    print(i, 'of', B.shape[1], end='\r')

C_sp = (1/(C/np.median(C)))  # Apply rescaler (np.median(C)) for ease of visualization
 
B *= C_sp # Apply normalization
```


### SVT
Example of how to apply singular value thresholding algorithm.
```Python
from full_profile import svt
inst = svt.svt(maxiter=100, tau_factor=1e-2, delta=1, method='scipy_gesdd', dense=True, verbose=True)
inst.run(B)
```

This line creates an instance of the SVT (likely "Singular Value Thresholding") class with the following parameters:

- ```maxiter=100```: First positional argument is the maximal number of iterations 
- ```tau_factor=1e-2```: Sets the tau factor to 0.01 (see Supplementary Materials)
- ```delta=1```: Sets delta, i.e. step size, to 1 (see Supplementary Materials)
- ```method='scipy_gesdd'```: Specifies the underlying svd method as 'scipy_gesdd' (multiple available see UniPy package)
- ```dense=True```: Sets the dense flag to True if enough memory is available to reconstruct B into dense memory (a True statement speeds up the calculations)
- ```verbose=True```: Enables verbose output

### FPC
Example of how to apply the fixed point continuation algorithm.
```Python
from full_profile import fpc
inst = fpc.fpc(maxiter=100, tau_factor=1e-3, delta=1.4, method='arpack', verbose=True)
inst.run(B)
```
This line creates an instance of the FPC (likely "Fixed Point Continuation") class with the following parameters:
- ```maxiter=100```: First positional argument is the maximal number of iterations 
- ```tau_factor=1e-2```: Sets the tau factor to 0.01 (see Supplementary Materials)
- ```delta=1```: Sets delta, i.e. step size, to 1 (see Supplementary Materials)
- ```method='scipy_gesdd'```: Specifies the underlying svd method as 'scipy_gesdd' (multiple available see UniPy package)
- ```dense=True```: Sets the dense flag to True if enough memory is available to reconstruct B into dense memory (a True statement speeds up the calculations)
- ```verbose=True```: Enables verbose output


### DFC % Under construction %
```Python
from full_profile import dfc
inst = dfc.dfc(maxiter=100, tau_factor=1e-3, delta=1.4, method='arpack', verbose=True)
inst.run(B)
```

## Latest News
2024/10/04: we are working hard to make UniPy available along with the DFC implementation
