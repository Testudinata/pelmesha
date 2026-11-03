# Graph Report - pelmesha  (2026-10-23)

## Corpus Check
- 17 files · ~58,024 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 35 file(s) not represented in the graph (top: .hdf5 11, .ibd 6, .imzML 6)

## Summary
- 864 nodes · 1559 edges · 67 communities (16 shown, 51 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 75 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7336ce82`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DataSource
- utensils.py
- KDEConfigs
- Aligner
- kneading.py
- PreparedDataSource
- serving.py
- Indexator
- BaseLoader
- AdaptiveParameter
- ndarray
- PipelineConfigurator
- DataSet
- filling.py
- LinkedList
- cookbook.py
- loader_imzml_rust
- DataManager
- loader_imzml
- .set_reference_source
- _parse_docstring
- .set_align_peaks_from_ref
- _compute_KDE
- ._repr_html_
- _make_datasource
- .feature_matrix
- loader_cdf
- _callable_fallback
- _inspect_defaults
- TestSplitIdxsInputNormalization
- Drawer
- _assert_count
- TestSplitIdxsDiscontinuous
- PyPI CI distribution build
- Configs
- TestSplitIdxsBudgetParams
- _assert_covers
- _flat_batches
- Pipeline
- Pelmesha Peak Extraction Library
- ds
- pelmesha

## God Nodes (most connected - your core abstractions)
1. `DataSet` - 54 edges
2. `PreparedDataSource` - 50 edges
3. `Indexator` - 49 edges
4. `Configs` - 47 edges
5. `DataSource` - 43 edges
6. `PipelineConfigurator` - 36 edges
7. `_make_datasource()` - 27 edges
8. `Pipeline` - 25 edges
9. `KDEConfigs` - 23 edges
10. `BaseLoader` - 23 edges

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

## Communities (67 total, 51 thin omitted)

### Community 1 - "utensils.py"
Cohesion: 0.09
Nodes (9): check_xy(), chunking_datasets(), convert_peak_values_to_index(), find_nearest_index(), generate_function(), _index_to_segment(), mspeaks_KD_legacy(), shift() (+1 more)

### Community 3 - "Aligner"
Cohesion: 0.07
Nodes (3): Aligner, format_time(), time_loop()

### Community 4 - "kneading.py"
Cohesion: 0.10
Nodes (17): Peak picking with area, FWHM and SNR, Per-spectrum processing pipeline, add_zero_points_to_peaks(), crop_spectrum(), _gaussian_filter(), modify_raw_spectrum(), _movaver(), msalign() (+9 more)

### Community 8 - "Indexator"
Cohesion: 0.12
Nodes (5): Indexator, SliceIndexator, counting_in_out(), wrapper(), kde_segments_not_overlap()

### Community 9 - "BaseLoader"
Cohesion: 0.09
Nodes (3): BaseLoader, loader_hdf5, loader_mzxml

### Community 11 - "ndarray"
Cohesion: 0.20
Nodes (11): _align_kde_mz_grids(), _append(), _compute_max(), _compute_max_with_ext_correction(), _fwhm_interp(), mspeaks_KD(), _peakpicker_core(), _peaks_snr_window() (+3 more)

### Community 15 - "PipelineConfigurator"
Cohesion: 0.07
Nodes (3): Configuration-driven processing pipeline, PipelineConfigurator, test_config_changes_direct()

### Community 16 - "DataSet"
Cohesion: 0.13
Nodes (3): DataSet, test_building_metadata(), test_config_changes_dataset()

### Community 21 - "DataManager"
Cohesion: 0.15
Nodes (10): Pipeline architecture diagram, Loading & metadata stage, Orchestration stage (serving), Preparation stage (cookbook configs), Reference peaks stage, Loading & metadata stage (user flow), Preparation stage (user flow), User flow diagram (+2 more)

### Community 24 - "_parse_docstring"
Cohesion: 0.20
Nodes (5): _detect_doc_style(), _parse_docstring(), _split_google(), _split_numpy(), _split_rst()

### Community 26 - "_compute_KDE"
Cohesion: 0.22
Nodes (4): _compute_KDE(), segment_probability_distribution(), _set_KDE_X_plot(), split_pdtable_by_peaks_gap()

### Community 29 - ".feature_matrix"
Cohesion: 0.11
Nodes (7): Feature matrix aggregation, apply_kde_mzcorrection(), _consensus_peaks_summary(), _consesusing_peaks(), _frequency_filtration(), _nunique_summary(), Peak_assignment()

### Community 32 - "_callable_fallback"
Cohesion: 0.28
Nodes (3): _callable_fallback(), _callable_to_yaml(), _yaml_to_callable()

### Community 44 - "PyPI CI distribution build"
Cohesion: 0.67
Nodes (4): PyPI CI distribution build, PyPI trusted publishing, TestPyPI CI distribution build, TestPyPI trusted publishing

### Community 50 - "Pipeline"
Cohesion: 0.17
Nodes (4): Main path to feature matrix, Pipeline main path to feature matrix, Pipeline, _summerize_kde_mz()

### Community 52 - "Pelmesha Peak Extraction Library"
Cohesion: 0.33
Nodes (4): Apache License 2.0, FFTKDE fast algorithm, Pelmesha Peak Extraction Library, TreeKDE default algorithm

## Knowledge Gaps
- **2 isolated node(s):** `pelmesha`, `Apache License 2.0`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 416 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **51 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Indexator` connect `Indexator` to `DataSource`, `PreparedDataSource`, `serving.py`, `._normalize_indices`, `BaseLoader`, `.get_mz_discretion_coeffs`, `DataSet`, `filling.py`, `cookbook.py`, `loader_imzml_rust`, `loader_imzml`, `_make_datasource`, `.feature_matrix`, `loader_cdf`, `._repr_html_`, `TestSplitIdxsInputNormalization`, `Drawer`, `dtype`, `_assert_count`, `TestSplitIdxsBudgetParams`, `Pipeline`, `.get_mz_stream`?**
  _High betweenness centrality (0.221) - this node is a cross-community bridge._
- **Why does `DataSet` connect `DataSet` to `DataSource`, `KDEConfigs`, `PreparedDataSource`, `serving.py`, `Indexator`, `PipelineConfigurator`, `cookbook.py`, `DataManager`, `.set_reference_source`, `.set_align_peaks_from_ref`, `.feature_matrix`, `._repr_html_`, `dtype`, `.reference_config`, `Configs`, `Pipeline`, `Pelmesha Peak Extraction Library`, `.__getattr__`, `ds`, `.check_config_changes`, `.close`, `.__getitem__`, `.items`, `.__iter__`, `.keys`, `.save_configs`, `.values`?**
  _High betweenness centrality (0.186) - this node is a cross-community bridge._
- **Why does `DataSource` connect `DataSource` to `KDEConfigs`, `kneading.py`, `PreparedDataSource`, `serving.py`, `._normalize_indices`, `Indexator`, `Drawer`, `DataSet`, `filling.py`, `cookbook.py`, `DataManager`, `.set_reference_source`, `_make_datasource`, `loader_cdf`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `DataSet` (e.g. with `Configs` and `KDEConfigs`) actually correct?**
  _`DataSet` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `PreparedDataSource` (e.g. with `Indexator` and `SliceIndexator`) actually correct?**
  _`PreparedDataSource` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Indexator` (e.g. with `PreparedDataSource` and `BaseLoader`) actually correct?**
  _`Indexator` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `DataSource` (e.g. with `MSI data loading & metadata extraction` and `PreparedDataSource`) actually correct?**
  _`DataSource` has 7 INFERRED edges - model-reasoned connections that need verification._