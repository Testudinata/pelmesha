# pelmesha

**Peak Extraction Library for Mass spectrometry Enhanced by Statistical High-throughput Analysis**

`pelmesha` is a Python package for processing **Mass Spectrometry Imaging (MSI)** data stored in `.imzml` (and, experimentally, `.cdf`) files. It loads raw spectra, processes them, detects peaks, corrects their m/z values with a kernel density estimate (KDE), and aggregates multiple samples/ROIs into a unified feature matrix.

## Features

- **Loading & metadata extraction** — reads raw MSI data and builds a structured metadata HDF5 file (`*_ingredients.hdf5`) containing sample metadata, per-ROI index ranges, m/z ranges, and spatial coordinates.
- **Configuration-driven processing pipeline** — a configuration system (`Configs`, `PipelineConfigurator`, `PreparedDataSource`) that validates parameters, distributes them to the pipeline steps, and supports YAML serialisation. The lightweight `KDEConfigs` class is built on Pydantic.
- **Spectrum processing** — smoothing, baseline correction, resampling to a uniform m/z scale, and alignment against reference peaks using a slightly modified version of the [`msalign`](https://github.com/lukasz-migas/msalign) implementation.
- **Peak picking** — detection of peaks together with their area, FWHM points, peak-base boundaries, and signal-to-noise ratio.
- **KDE-based m/z correction** — peaks that wander slightly across spectra are grouped into single m/z values based on their kernel density estimate (using [KDEpy](https://github.com/tommyod/KDEpy)).
- **Multi-sample aggregation** — builds a feature matrix from the peak lists of several samples and ROIs, with optional occurrence filtering, duplicate merging, pivoting, and coordinate merging.
- **Reference peaks** — generates a reference peak list from a reference source and uses it to align the other samples registered in the same `DataSet`.

## Installation

```bash
pip install pelmesha
```

Requires Python `>= 3.10`. The main dependencies are `numpy`, `pandas`, `scipy`, `h5py`, `pyimzml`, `pybaselines`, `KDEpy`, `scikit-learn`, `xarray`, `pydantic`, `pyyaml`, and `pyarrow`.

## Quick start

### 1. Load data sources

Create a `DataSet` from a list of raw files (or directories):

```python
from pelmesha import DataSet

ds = DataSet(sources=["/data/sample1.imzml", "/data/sample2.imzml"])
```

Each source is wrapped in a `PreparedDataSource` and registered by its sample name:

```python
print(ds)                 # text table of the registered sources
ds["sample1"]             # access a prepared source by sample name
```

### 2. Configure the processing pipeline

Per-source processing and KDE configurations can be adjusted either for all ROIs at once or for a single ROI:

```python
# Update a parameter for all ROIs of one sample
ds["sample1"].update({"smooth_window": 7})
ds["sample1"].update_kde(bwc=1.5)

# Or configure a specific ROI directly
roi_config = ds["sample1"].roi_configs["R00"]
roi_config["SNR_threshold"] = 4
roi_config["smooth_algo"] = "GA"

kde_config = ds["sample1"].roi_kde_configs["R00"]
kde_config["bwc"] = 1.5

# Exclude specific methods from the pipeline by deleting them from the config
# This disables baseline correction for this RO
roi_config.delete('Baseline') # Baseline correction will be not implemented

# Change the algorithm or method using set_method
# This replaces the method with 'modpoly' and updates default parameters
roi_config.set_method('Baseline','modpoly') 
```

Configurations are stored next to each source file and can be saved/loaded as YAML (`*_processing_recipe.yaml` and `*_kde_recipe.yaml`).

### 3. Process spectra and pick peaks

```python
ds.process()    # smoothing → baseline → resampling → alignment
ds.peakpick()   # smoothing → baseline → resampling → alignment and detect peaks for every spectrum
```

These write `*_processed_spectra.hdf5` and `*_peaklists.hdf5` next to each source.

### 4. Estimate peak density

```python
ds.estimate_peak_density_kde()
```

This writes the per-ROI peak probability density into `*_peaks_density.hdf5`.

### 5. Build a feature matrix

```python
fm = ds.feature_matrix(countf=10, pivot_values="Intensity")
```

Optionally save it as Parquet together with the coordinates:

```python
fm = ds.feature_matrix(save_path="results/feature_matrix.parquet",
                       merge_with_coords=True)
```

### 6. Use reference peaks for alignment (optional)

Reference peaks are **optional**. If you want cross-sample alignment, generate the reference peak list **before** running `process` / `peakpick` on the other samples:

```python
ds.set_reference_source("/data/reference.imzml")
ds.get_reference_peaks()
ds.set_align_peaks_from_ref()
```

`set_align_peaks_from_ref` then assigns the reference peaks to the selected samples/ROIs of the `DataSet` as their alignment targets.

## Pipeline steps

The per-spectrum processing pipeline consists of the following steps:

1. **Smoothing** — moving-average (`MA`), Gaussian (`GA`), or Savitzky–Golay (`SG`) filters.
2. **Baseline correction** — using the [pybaselines](https://pybaselines.readthedocs.io) library.
3. **Resampling** — brings the data onto a uniform m/z scale (`resample_mz_scale`).
4. **Alignment** — calibration and alignment relative to reference peaks using the bundled, slightly modified [`msalign`](https://github.com/lukasz-migas/msalign) implementation in [`Aligner`](src/pelmesha/align.py).

After peak picking, the probability density function (PDF) of the peaks is built and saved for every individual ROI, using the parameters specified for that ROI. Once both peak picking and the PDF estimation are complete, the resulting files are ready to be combined into a single dataset and to form a common feature matrix across all the samples and ROIs.

## Project structure

| Module | Purpose |
| ------ | ------- |
| [`filling`](src/pelmesha/filling.py) | `DataSource`, `DataManager`, and format-specific loaders (imzML, CDF). |
| [`dough`](src/pelmesha/dough.py) | Utility classes: `LinkedList`, `AdaptiveParameter`, `Indexator`, `SliceIndexator`. |
| [`cookbook`](src/pelmesha/cookbook.py) | Configuration system: `Configs`, `PipelineConfigurator`, `PreparedDataSource`, `KDEConfigs`. |
| [`kneading`](src/pelmesha/kneading.py) | Base pipeline functions: processing, smoothing, peak picking, KDE estimation, `msalign`. |
| [`serving`](src/pelmesha/serving.py) | Orchestration and visualisation: `DataSet`, `Pipeline`, `Drawer`. |
| [`align`](src/pelmesha/align.py) | The `Aligner` class implementing signal calibration/alignment. |
| [`utensils`](src/pelmesha/utensils.py) | Assorted helper functions and constants. |

## License

Distributed under the Apache-2.0 license. See [`LICENSE.txt`](LICENSE.txt).
