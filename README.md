# Code for Paper "Preserving Full Spectrum Information in Imaging Mass Spectrometry Data Reduction"

BioRxiv: [https://www.biorxiv.org/content/10.1101/2024.09.30.614425v1](https://www.biorxiv.org/content/10.1101/2024.09.30.614425v1)

Dependencies: this toolbox depends on a few other in-house toolboxes. i.e. unipy, pyspa and ...


## Download and Unpack Data
DOI: [https://10.4121/a6efd47a-b4ec-493e-a742-70e8a369f788](https://doi.org/10.4121/a6efd47a-b4ec-493e-a742-70e8a369f788)

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

## From .d to NumPy Array
- Put code here on how to get to our format

## Applying Methods
- Include preprocessing (5%-95% TIC normalization)
- Examples on how to apply
