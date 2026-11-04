# Graph Report - pelmesha  (2026-11-04)

## Corpus Check
- 12 files · ~55,983 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .lock 2, (none) 1, .toml 1)

## Summary
- 784 nodes · 1402 edges · 51 communities (11 shown, 40 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 60 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `39d5da71`
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
- apply_kde_mzcorrection
- Configs
- LinkedList
- loader_imzml_rust
- Pelmesha Peak Extraction Library
- loader_imzml
- DataSet
- _parse_docstring
- .get_reference_peaks
- _compute_KDE
- ._repr_html_
- _consesusing_peaks
- loader_cdf
- _callable_fallback
- _inspect_defaults
- PipelineConfigurator
- AbsoluteFormatter
- PyPI CI distribution build
- KDE-based m/z correction
- pelmesha

## God Nodes (most connected - your core abstractions)
1. `Configs` - 51 edges
2. `DataSet` - 51 edges
3. `PreparedDataSource` - 49 edges
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

## Communities (51 total, 40 thin omitted)

### Community 1 - "utensils.py"
Cohesion: 0.07
Nodes (14): check_xy(), chunking_datasets(), convert_peak_values_to_index(), del_hdf5(), find_nearest_index(), format_time(), generate_function(), _index_to_segment() (+6 more)

### Community 4 - "kneading.py"
Cohesion: 0.10
Nodes (18): Peak picking with area, FWHM and SNR, Per-spectrum processing pipeline, add_zero_points_to_peaks(), _build_headers(), crop_spectrum(), _gaussian_filter(), modify_raw_spectrum(), _movaver() (+10 more)

### Community 8 - "Indexator"
Cohesion: 0.05
Nodes (7): Main path to feature matrix, Pipeline main path to feature matrix, Indexator, SliceIndexator, Drawer, Pipeline, _frequency_filtration()

### Community 9 - "BaseLoader"
Cohesion: 0.09
Nodes (3): BaseLoader, loader_hdf5, loader_mzxml

### Community 11 - "apply_kde_mzcorrection"
Cohesion: 0.11
Nodes (17): Feature matrix aggregation, segment_probability_distribution(), _align_kde_mz_grids(), _append(), apply_kde_mzcorrection(), _compute_max(), _compute_max_with_ext_correction(), _fwhm_interp() (+9 more)

### Community 21 - "Pelmesha Peak Extraction Library"
Cohesion: 0.13
Nodes (12): Apache License 2.0, Pipeline architecture diagram, Loading & metadata stage, Orchestration stage (serving), Preparation stage (cookbook configs), Loading & metadata stage (user flow), Preparation stage (user flow), User flow diagram (+4 more)

### Community 24 - "_parse_docstring"
Cohesion: 0.20
Nodes (5): _detect_doc_style(), _parse_docstring(), _split_google(), _split_numpy(), _split_rst()

### Community 25 - ".get_reference_peaks"
Cohesion: 0.18
Nodes (3): Reference peaks stage, Reference peaks flow, Reference peak alignment

### Community 29 - "_consesusing_peaks"
Cohesion: 0.25
Nodes (3): _consensus_peaks_summary(), _consesusing_peaks(), _nunique_summary()

### Community 32 - "_callable_fallback"
Cohesion: 0.28
Nodes (3): _callable_fallback(), _callable_to_yaml(), _yaml_to_callable()

### Community 44 - "PyPI CI distribution build"
Cohesion: 0.67
Nodes (4): PyPI CI distribution build, PyPI trusted publishing, TestPyPI CI distribution build, TestPyPI trusted publishing

## Knowledge Gaps
- **2 isolated node(s):** `pelmesha`, `Apache License 2.0`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 384 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **40 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataSet` connect `DataSet` to `DataSource`, `KDEConfigs`, `PipelineConfigurator`, `PreparedDataSource`, `serving.py`, `Indexator`, `.reference_config`, `Configs`, `Pelmesha Peak Extraction Library`, `.get_reference_peaks`, `._repr_html_`?**
  _High betweenness centrality (0.194) - this node is a cross-community bridge._
- **Why does `Configs` connect `Configs` to `_inspect_defaults`, `PipelineConfigurator`, `kneading.py`, `serving.py`, `.__getitem__`, `.__setitem__`, `._update_flat`, `Any`, `cookbook.py`, `._load`, `Pelmesha Peak Extraction Library`, `DataSet`, `._repr_html_`?**
  _High betweenness centrality (0.189) - this node is a cross-community bridge._
- **Why does `Indexator` connect `Indexator` to `DataSource`, `PreparedDataSource`, `serving.py`, `BaseLoader`, `.get_metadata`, `.get_mz_discretion_coeffs`, `cookbook.py`, `loader_imzml_rust`, `loader_imzml`, `DataSet`, `loader_cdf`, `._repr_html_`?**
  _High betweenness centrality (0.165) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `DataSet` (e.g. with `Configs` and `KDEConfigs`) actually correct?**
  _`DataSet` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PreparedDataSource` (e.g. with `Indexator` and `SliceIndexator`) actually correct?**
  _`PreparedDataSource` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `DataSource` (e.g. with `MSI data loading & metadata extraction` and `PreparedDataSource`) actually correct?**
  _`DataSource` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Indexator` (e.g. with `PreparedDataSource` and `BaseLoader`) actually correct?**
  _`Indexator` has 9 INFERRED edges - model-reasoned connections that need verification._