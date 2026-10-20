# Graph Report - pelmesha  (2026-10-15)

## Corpus Check
- 12 files · ~53,769 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: .lock 2, (none) 1, .ipynb 1)

## Summary
- 748 nodes · 1352 edges · 57 communities (16 shown, 41 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 60 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cf060cbe`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DataSource
- utensils.py
- KDEConfigs
- Aligner
- ndarray
- PreparedDataSource
- serving.py
- .shift
- Pipeline
- BaseLoader
- AdaptiveParameter
- apply_kde_mzcorrection
- Configs
- Indexator
- PipelineConfigurator
- DataSet
- kneading.py
- LinkedList
- loader_imzml_rust
- DataManager
- loader_imzml
- .set_reference_source
- Pelmesha Peak Extraction Library
- .get_reference_peaks
- _compute_KDE
- ._escape_html
- _parse_docstring
- loader_cdf
- _callable_fallback
- _inspect_defaults
- .__init__
- modify_raw_spectrum
- PyPI CI distribution build
- _consensus_peaks_summary
- Per-spectrum processing pipeline
- del_hdf5
- pelmesha
- chunking_datasets

## God Nodes (most connected - your core abstractions)
1. `DataSet` - 51 edges
2. `PreparedDataSource` - 49 edges
3. `Configs` - 47 edges
4. `DataSource` - 42 edges
5. `Indexator` - 40 edges
6. `PipelineConfigurator` - 35 edges
7. `Pipeline` - 25 edges
8. `BaseLoader` - 23 edges
9. `KDEConfigs` - 22 edges
10. `SliceIndexator` - 22 edges

## Surprising Connections (you probably didn't know these)
- `Reference peak alignment` --references--> `Aligner`  [INFERRED]
  README.md → src/pelmesha/align.py
- `Per-spectrum processing pipeline` --references--> `Aligner`  [INFERRED]
  README.md → src/pelmesha/align.py
- `MSI data loading & metadata extraction` --references--> `DataSource`  [INFERRED]
  README.md → src/pelmesha/filling.py
- `MSI data loading & metadata extraction` --references--> `loader_imzml`  [INFERRED]
  README.md → src/pelmesha/filling.py
- `MSI data loading & metadata extraction` --references--> `loader_cdf`  [INFERRED]
  README.md → src/pelmesha/filling.py

## Import Cycles
- 3-file cycle: `src/pelmesha/__init__.py -> src/pelmesha/cookbook.py -> src/pelmesha/kneading.py -> src/pelmesha/__init__.py`
- 3-file cycle: `src/pelmesha/__init__.py -> src/pelmesha/serving.py -> src/pelmesha/kneading.py -> src/pelmesha/__init__.py`
- 4-file cycle: `src/pelmesha/__init__.py -> src/pelmesha/serving.py -> src/pelmesha/cookbook.py -> src/pelmesha/kneading.py -> src/pelmesha/__init__.py`

## Hyperedges (group relationships)
- **Distribution Workflows** — _github_workflows_publish_to_pypi_distribution, _github_workflows_publish_to_pypi_publish, _github_workflows_publish_to_test_pypi_distribution, _github_workflows_publish_to_test_pypi_publish [INFERRED 0.85]
- **KDE m/z Correction Approach** — readme_kde_mz_correction, readme_fftkde, readme_treekde, readme_adaptive_bandwidth, readme_kde_additivity [INFERRED 0.85]
- **Pipeline Architecture Stages** — plans_architecture_diagramm_loading_metadata, plans_architecture_diagramm_preparation, plans_architecture_diagramm_orchestration, plans_architecture_diagramm_reference_peaks_stage, plans_architecture_diagramm_feature_matrix_path [INFERRED 0.85]

## Communities (57 total, 41 thin omitted)

### Community 1 - "utensils.py"
Cohesion: 0.12
Nodes (4): mspeaks_KD_legacy(), shift(), show_df(), split_pdtable_by_medfwhm()

### Community 4 - "ndarray"
Cohesion: 0.20
Nodes (6): Peak picking with area, FWHM and SNR, msalign(), peakpicker(), peakpicker_legacy(), peakpicking_base(), process_spectra_base()

### Community 7 - ".shift"
Cohesion: 0.12
Nodes (3): format_time(), generate_function(), time_loop()

### Community 8 - "Pipeline"
Cohesion: 0.05
Nodes (8): Main path to feature matrix, Pipeline main path to feature matrix, SliceIndexator, AbsoluteFormatter, Drawer, Pipeline, _consesusing_peaks(), _frequency_filtration()

### Community 9 - "BaseLoader"
Cohesion: 0.10
Nodes (3): BaseLoader, loader_hdf5, loader_mzxml

### Community 11 - "apply_kde_mzcorrection"
Cohesion: 0.11
Nodes (17): Feature matrix aggregation, segment_probability_distribution(), _align_kde_mz_grids(), _append(), apply_kde_mzcorrection(), _compute_max(), _compute_max_with_ext_correction(), _fwhm_interp() (+9 more)

### Community 17 - "kneading.py"
Cohesion: 0.24
Nodes (4): _gaussian_filter(), _movaver(), _savgol(), smoothing()

### Community 21 - "DataManager"
Cohesion: 0.17
Nodes (9): Pipeline architecture diagram, Loading & metadata stage, Orchestration stage (serving), Preparation stage (cookbook configs), Loading & metadata stage (user flow), Preparation stage (user flow), User flow diagram, MSI data loading & metadata extraction (+1 more)

### Community 24 - "Pelmesha Peak Extraction Library"
Cohesion: 0.29
Nodes (5): Apache License 2.0, Configuration-driven processing pipeline, FFTKDE fast algorithm, Pelmesha Peak Extraction Library, TreeKDE default algorithm

### Community 25 - ".get_reference_peaks"
Cohesion: 0.18
Nodes (3): Reference peaks stage, Reference peaks flow, Reference peak alignment

### Community 28 - "_parse_docstring"
Cohesion: 0.20
Nodes (5): _detect_doc_style(), _parse_docstring(), _split_google(), _split_numpy(), _split_rst()

### Community 32 - "_callable_fallback"
Cohesion: 0.28
Nodes (3): _callable_fallback(), _callable_to_yaml(), _yaml_to_callable()

### Community 38 - ".__init__"
Cohesion: 0.25
Nodes (3): check_xy(), convert_peak_values_to_index(), find_nearest_index()

### Community 39 - "modify_raw_spectrum"
Cohesion: 0.29
Nodes (4): add_zero_points_to_peaks(), crop_spectrum(), modify_raw_spectrum(), reduce_signal_to_zero()

### Community 44 - "PyPI CI distribution build"
Cohesion: 0.67
Nodes (4): PyPI CI distribution build, PyPI trusted publishing, TestPyPI CI distribution build, TestPyPI trusted publishing

### Community 52 - "Per-spectrum processing pipeline"
Cohesion: 0.40
Nodes (3): Per-spectrum processing pipeline, preprocess_configuration_base(), resample_mz_scale()

## Knowledge Gaps
- **2 isolated node(s):** `pelmesha`, `Apache License 2.0`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 360 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **41 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataSet` connect `DataSet` to `DataSource`, `KDEConfigs`, `PreparedDataSource`, `serving.py`, `Pipeline`, `.reference_config`, `Configs`, `Indexator`, `PipelineConfigurator`, `DataManager`, `.set_reference_source`, `Pelmesha Peak Extraction Library`, `.get_reference_peaks`, `._repr_html_`?**
  _High betweenness centrality (0.201) - this node is a cross-community bridge._
- **Why does `Configs` connect `Configs` to `_inspect_defaults`, `.replace_config`, `._update_nested`, `serving.py`, `.__getitem__`, `.delete`, `Any`, `PipelineConfigurator`, `DataSet`, `kneading.py`, `cookbook.py`, `DataManager`, `Pelmesha Peak Extraction Library`, `._escape_html`?**
  _High betweenness centrality (0.179) - this node is a cross-community bridge._
- **Why does `Indexator` connect `Indexator` to `DataSource`, `PreparedDataSource`, `serving.py`, `Pipeline`, `BaseLoader`, `DataSet`, `cookbook.py`, `loader_imzml_rust`, `loader_imzml`, `loader_cdf`, `._repr_html_`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `DataSet` (e.g. with `Configs` and `KDEConfigs`) actually correct?**
  _`DataSet` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PreparedDataSource` (e.g. with `Indexator` and `SliceIndexator`) actually correct?**
  _`PreparedDataSource` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `DataSource` (e.g. with `MSI data loading & metadata extraction` and `PreparedDataSource`) actually correct?**
  _`DataSource` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Indexator` (e.g. with `PreparedDataSource` and `BaseLoader`) actually correct?**
  _`Indexator` has 9 INFERRED edges - model-reasoned connections that need verification._