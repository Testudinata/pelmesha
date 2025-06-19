# Package pelmesha

`pelmesha` (Peak Extraction Library for Mass spectrometry Enhanced by Statistical High-throughput Analysis) is a Python package that allows users to process Mass Spectrometry Imaging (MSI) data from imzml files. It provides several features, including:
1) Conversion of raw imzml data to the hdf5 format (function `imzml2hdf5` from module `pspectra`)
2) Processing of mass spectra (function `Raw2proc` from module `pspectra`)
3) Creation of peaklists from the data (functions `proc2peaklist` and `Raw2peaklist` from module `pspectra`)
4) Generation of a feature matrix from a single image's peaklist by grouping peaks (function `Pgrouping_KD` from module `pfeats`)
5) Generation of a feature matrix for multiple images from the peaklists of each image (function `Roi_Pgrouping_KD` from module `pfeats`)

The processing of mass spectra includes several steps:
- **Data resampling** — This process allows you to bring data to a uniform scale between points on the `mz` and to a single scale on the `mz`.
- **Alignment of spectra relative to reference peaks** using the [`msalign`](https://github.com/lukasz-migas/msalign) tool. It should be noted that `msalign` does a worse job with non-continuous and non-uniform data, so it is strongly recommended to perform a resampling process before using it. Also `msalign` is modified in this package for correct work with other steps.
- **Baseline correction** using the [pybaselines](https://pybaselines.readthedocs.io) package.
- **Smoothing** — based on code snippets from the [mMass](https://github.com/xxao/mMass) library, which provide smoothing using the moving average, Gaussian, and Savitsky-Goley algorithms.
- **Peak-picking** — peaks in the spectrum are searched and filtered.

In the mass spectra in the image, there is a slight difference in the peak values, even after alignment during processing. To further analyze these spectra, it will be necessary to group the peaks based on their relative positions and create a feature matrix as a result.

The `Pgrouping_KD` and `Roi_Pgrouping_KD` functions combine the wandering peak values from signals in mass spectra into a single `mz` value based on their kernel density estimation. This is achieved by determining the centers around which the peak values are located.

To ensure accurate results, high-quality bandwidth selection is required. This can be done manually or automatically. The probability density is estimated using the FFTKDE function from the [kdepy package](https://github.com/tommyod/KDEpy), which is extremely fast and uses Fourier transforms for calculation.
