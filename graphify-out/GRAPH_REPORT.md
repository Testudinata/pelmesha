# Graph Report - pelmesha  (2026-11-10)

## Corpus Check
- 12 files · ~56,136 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .lock 2, (none) 1, .toml 1)

## Summary
- 792 nodes · 1410 edges · 57 communities (13 shown, 44 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 57 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b011834a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DataSource
- utensils.py
- KDEConfigs
- Aligner
- kneading.py
- PreparedDataSource
- Indexator
- BaseLoader
- AdaptiveParameter
- ndarray
- .get_mz_discretion_coeffs
- .shift
- LinkedList
- loader_imzml_rust
- Pelmesha Peak Extraction Library
- loader_imzml
- DataSet
- _parse_docstring
- .set_reference_source
- _compute_KDE
- ._repr_html_
- .feature_matrix
- loader_cdf
- _callable_fallback
- _inspect_defaults
- PipelineConfigurator
- AbsoluteFormatter
- Pipeline
- Drawer
- .__init__
- .create_metafile
- PyPI CI distribution build
- Configs
- peakpicker
- apply_kde_mzcorrection
- pelmesha

## God Nodes (most connected - your core abstractions)
1. `Configs` - 53 edges
2. `DataSet` - 51 edges
3. `PreparedDataSource` - 49 edges
4. `DataSource` - 42 edges
5. `Indexator` - 40 edges
6. `PipelineConfigurator` - 35 edges
7. `Pipeline` - 25 edges
8. `BaseLoader` - 24 edges
9. `KDEConfigs` - 22 edges
10. `SliceIndexator` - 22 edges

## Surprising Connections (you probably didn't know these)
- `KDE additivity optimization` --references--> `segment_probability_distribution()`  [INFERRED]
  README.md → src/pelmesha/kneading.py
- `Reference peak alignment` --references--> `Aligner`  [INFERRED]
  README.md → src/pelmesha/align.py
- `Per-spectrum processing pipeline` --references--> `Aligner`  [INFERRED]
  README.md → src/pelmesha/align.py
- `MSI data loading & metadata extraction` --references--> `DataSource`  [INFERRED]
  README.md → src/pelmesha/filling.py
- `MSI data loading & metadata extraction` --references--> `loader_imzml`  [INFERRED]
  README.md → src/pelmesha/filling.py

## Import Cycles
- 3-file cycle: `src/pelmesha/__init__.py -> src/pelmesha/cookbook.py -> src/pelmesha/kneading.py -> src/pelmesha/__init__.py`
- 3-file cycle: `src/pelmesha/__init__.py -> src/pelmesha/serving.py -> src/pelmesha/kneading.py -> src/pelmesha/__init__.py`
- 4-file cycle: `src/pelmesha/__init__.py -> src/pelmesha/serving.py -> src/pelmesha/cookbook.py -> src/pelmesha/kneading.py -> src/pelmesha/__init__.py`

## Hyperedges (group relationships)
- **Distribution Workflows** — _github_workflows_publish_to_pypi_distribution, _github_workflows_publish_to_pypi_publish, _github_workflows_publish_to_test_pypi_distribution, _github_workflows_publish_to_test_pypi_publish [INFERRED 0.85]
- **KDE m/z Correction Approach** — readme_kde_mz_correction, readme_fftkde, readme_treekde, readme_adaptive_bandwidth, readme_kde_additivity [INFERRED 0.85]
- **Pipeline Architecture Stages** — plans_architecture_diagramm_loading_metadata, plans_architecture_diagramm_preparation, plans_architecture_diagramm_orchestration, plans_architecture_diagramm_reference_peaks_stage, plans_architecture_diagramm_feature_matrix_path [INFERRED 0.85]

## Communities (57 total, 44 thin omitted)

### Community 1 - "utensils.py"
Cohesion: 0.13
Nodes (6): check_xy(), chunking_datasets(), generate_function(), mspeaks_KD_legacy(), shift(), split_array_by_num_distance()

### Community 4 - "kneading.py"
Cohesion: 0.12
Nodes (15): Per-spectrum processing pipeline, add_zero_points_to_peaks(), _build_headers(), crop_spectrum(), _gaussian_filter(), modify_raw_spectrum(), _movaver(), msalign() (+7 more)

### Community 9 - "BaseLoader"
Cohesion: 0.08
Nodes (3): BaseLoader, loader_hdf5, loader_mzxml

### Community 11 - "ndarray"
Cohesion: 0.21
Nodes (11): _append(), _compute_max(), _compute_max_with_ext_correction(), _fwhm_interp(), _fwhm_interp_old(), mspeaks_KD(), _peakpicker_core(), _peaks_snr_window() (+3 more)

### Community 21 - "Pelmesha Peak Extraction Library"
Cohesion: 0.11
Nodes (8): Apache License 2.0, Loading & metadata stage, Configuration-driven processing pipeline, FFTKDE fast algorithm, MSI data loading & metadata extraction, Pelmesha Peak Extraction Library, TreeKDE default algorithm, DataManager

### Community 24 - "_parse_docstring"
Cohesion: 0.20
Nodes (5): _detect_doc_style(), _parse_docstring(), _split_google(), _split_numpy(), _split_rst()

### Community 25 - ".set_reference_source"
Cohesion: 0.14
Nodes (10): Pipeline architecture diagram, Orchestration stage (serving), Preparation stage (cookbook configs), Reference peaks stage, DataSet entry point & add_sources, Loading & metadata stage (user flow), Preparation stage (user flow), Reference peaks flow (+2 more)

### Community 26 - "_compute_KDE"
Cohesion: 0.22
Nodes (4): _compute_KDE(), segment_probability_distribution(), _set_KDE_X_plot(), split_pdtable_by_peaks_gap()

### Community 29 - ".feature_matrix"
Cohesion: 0.11
Nodes (6): Feature matrix aggregation, _consensing_peaks(), _consensus_peaks_summary(), _frequency_filtration(), _nunique_summary(), show_df()

### Community 32 - "_callable_fallback"
Cohesion: 0.28
Nodes (3): _callable_fallback(), _callable_to_yaml(), _yaml_to_callable()

### Community 37 - "Pipeline"
Cohesion: 0.17
Nodes (5): Main path to feature matrix, Pipeline main path to feature matrix, Pipeline, _align_kde_mz_grids(), _summerize_kde_mz()

### Community 44 - "PyPI CI distribution build"
Cohesion: 0.67
Nodes (4): PyPI CI distribution build, PyPI trusted publishing, TestPyPI CI distribution build, TestPyPI trusted publishing

### Community 48 - "peakpicker"
Cohesion: 0.40
Nodes (3): Peak picking with area, FWHM and SNR, peakpicker(), peakpicking_base()

## Knowledge Gaps
- **2 isolated node(s):** `pelmesha`, `Apache License 2.0`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 388 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **44 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Configs` connect `Configs` to `_inspect_defaults`, `PipelineConfigurator`, `kneading.py`, `serving.py`, `.__getitem__`, `._load_doc_ref`, `Any`, `._update_nested`, `cookbook.py`, `Pelmesha Peak Extraction Library`, `DataSet`, `.set_reference_source`, `._repr_html_`?**
  _High betweenness centrality (0.195) - this node is a cross-community bridge._
- **Why does `DataSet` connect `DataSet` to `DataSource`, `KDEConfigs`, `PipelineConfigurator`, `dtype`, `PreparedDataSource`, `serving.py`, `._add_single_source`, `Indexator`, `Pipeline`, `.reference_config`, `Configs`, `._resolve_sample_rois_map`, `Pelmesha Peak Extraction Library`, `.items`, `.keys`, `.set_reference_source`, `.feature_matrix`, `._repr_html_`?**
  _High betweenness centrality (0.194) - this node is a cross-community bridge._
- **Why does `Indexator` connect `Indexator` to `DataSource`, `dtype`, `PreparedDataSource`, `serving.py`, `._normalize_indices`, `Drawer`, `BaseLoader`, `Pipeline`, `.get_mz_discretion_coeffs`, `.get_coords`, `cookbook.py`, `loader_imzml_rust`, `loader_imzml`, `DataSet`, `.feature_matrix`, `loader_cdf`, `._repr_html_`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `DataSet` (e.g. with `Configs` and `KDEConfigs`) actually correct?**
  _`DataSet` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PreparedDataSource` (e.g. with `Indexator` and `SliceIndexator`) actually correct?**
  _`PreparedDataSource` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `DataSource` (e.g. with `MSI data loading & metadata extraction` and `PreparedDataSource`) actually correct?**
  _`DataSource` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Indexator` (e.g. with `PreparedDataSource` and `BaseLoader`) actually correct?**
  _`Indexator` has 9 INFERRED edges - model-reasoned connections that need verification._