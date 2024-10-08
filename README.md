Code for Paper "Preserving Full Spectrum Information in Imaging Mass Spectrometry Data Reduction"

# Download data
DOI: [https://10.4121/a6efd47a-b4ec-493e-a742-70e8a369f788](https://doi.org/10.4121/a6efd47a-b4ec-493e-a742-70e8a369f788)
Data: [https://surfdrive.surf.nl/files/index.php/s/TLPccCCAP7Uat5r](https://surfdrive.surf.nl/files/index.php/s/TLPccCCAP7Uat5r)

There are 2 data sets: (1) FT-ICR (fticr_data.tar.gz), (2) qTOF (all other files, i.e. xaa-xaz). Below are the steps explained for Linux/Mac unpacking of the data.

## FT-ICR
Unpack data
```
tar -xf fticr_data.tar.gz -C fticr_data
```

## qTOF
```
cat *.tar.gz.* | tar xvfz -
```
