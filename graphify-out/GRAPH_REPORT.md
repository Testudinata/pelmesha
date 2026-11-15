# Graph Report - pelmesha  (2026-11-17)

## Corpus Check
- 18 files · ~57,015 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .lock 2, (none) 1, .toml 1)

## Summary
- 924 nodes · 1619 edges · 44 communities (13 shown, 31 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 69 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Architecture & Data Model
- BaseLoader Abstract Interface
- DataSet ROI Access API
- Config Models & Validation
- Spectral Alignment Algorithms
- Pipeline Configurator Serialization
- TestPyPI Publish Workflow
- Peak Picking Preprocessing
- DataSource ROI Access
- Config Serialization Utilities
- PyPI Publish Workflow
- Processing Pipeline Orchestration
- Plotting & Visualization
- Third-Party Dependencies
- CDF Loader
- KDE Peak Picking Kernels
- imzML Loader
- Signal Alignment Utilities
- Changelog Workflow
- Adaptive Parameter Type
- KDE m/z Correction
- HTML Config Rendering
- Index Segment Utilities
- LinkedList Array Sync
- Multiprocessing Manager
- Docstring Reference Resolution
- Parameter Inspection & Binding
- Configs Base Class
- DataLoader Factory
- KDE Distribution Estimation
- Matplotlib Tick Formatter
- OpenCode Permissions
- Pelmesha Package

## God Nodes (most connected - your core abstractions)
1. `Configs` - 50 edges
2. `DataSet` - 50 edges
3. `PreparedDataSource` - 47 edges
4. `DataSource` - 41 edges
5. `Indexator` - 40 edges
6. `PipelineConfigurator` - 32 edges
7. `Pipeline` - 24 edges
8. `SliceIndexator` - 22 edges
9. `BaseLoader` - 22 edges
10. `KDEConfigs` - 21 edges

## Surprising Connections (you probably didn't know these)
- `DataSet (entry point)` --semantically_similar_to--> `DataSet`  [INFERRED] [semantically similar]
  plans/User_flow_diagramm.md → README.md
- `DataSet` --semantically_similar_to--> `DataSet`  [INFERRED] [semantically similar]
  README.md → plans/Architecture_diagramm.md
- `PreparedDataSource` --semantically_similar_to--> `PreparedDataSource`  [INFERRED] [semantically similar]
  README.md → plans/Architecture_diagramm.md
- `Configs` --semantically_similar_to--> `Configs`  [INFERRED] [semantically similar]
  README.md → plans/Architecture_diagramm.md
- `PipelineConfigurator` --semantically_similar_to--> `PipelineConfigurator`  [INFERRED] [semantically similar]
  README.md → plans/Architecture_diagramm.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **KDE-based Feature Matrix Construction Flow** — plans_architecture_diagramm_peakpick, plans_architecture_diagramm_estimate_peak_density_kde, plans_architecture_diagramm_summerize_kde_mz, plans_architecture_diagramm_apply_kde_mzcorrection, plans_architecture_diagramm_feature_matrix, plans_architecture_diagramm_filter_consensus_peaks [EXTRACTED 1.00]
- **MSI Data Loading and Metadata Preparation Flow** — plans_architecture_diagramm_dataloader, plans_architecture_diagramm_loader_imzml, plans_architecture_diagramm_loader_cdf, plans_architecture_diagramm_baseloader, plans_architecture_diagramm_per_roi_metadata, plans_architecture_diagramm_datasource [EXTRACTED 1.00]
- **MultiProcessorManager Concurrency Topology** — plans_multiprocessing_manager_plan_free_ids_queue, plans_multiprocessing_manager_plan_thr_batcher, plans_multiprocessing_manager_plan_batches_queue, plans_multiprocessing_manager_plan_batch_generator, plans_multiprocessing_manager_plan_thr_pool, plans_multiprocessing_manager_plan_thr_writer, plans_multiprocessing_manager_plan_meta_dict [EXTRACTED 1.00]

## Communities (44 total, 31 thin omitted)

### Community 0 - "Architecture & Data Model"
Cohesion: 0.06
Nodes (56): Apache License 2.0, apply_kde_mzcorrection, Pelmesha Architecture Diagram, BaseLoader (ABC), Configs, Custom User Processing, DataManager.get_loader, DataSet (+48 more)

### Community 1 - "BaseLoader Abstract Interface"
Cohesion: 0.05
Nodes (4): BaseLoader, loader_hdf5, loader_imzml_rust, loader_mzxml

### Community 4 - "Spectral Alignment Algorithms"
Cohesion: 0.06
Nodes (4): Aligner, format_time(), generate_function(), time_loop()

### Community 6 - "TestPyPI Publish Workflow"
Cohesion: 0.07
Nodes (29): name, runs-on, steps, name, url, jobs, build, publish-to-testpypi (+21 more)

### Community 7 - "Peak Picking Preprocessing"
Cohesion: 0.11
Nodes (16): add_zero_points_to_peaks(), _build_headers(), crop_spectrum(), _gaussian_filter(), modify_raw_spectrum(), _movaver(), msalign(), peakpicker() (+8 more)

### Community 9 - "Config Serialization Utilities"
Cohesion: 0.08
Nodes (7): _callable_fallback(), _callable_to_yaml(), _detect_doc_style(), _split_google(), _split_numpy(), _split_rst(), _yaml_to_callable()

### Community 10 - "PyPI Publish Workflow"
Cohesion: 0.08
Nodes (28): name, runs-on, steps, name, url, jobs, build, publish-to-pypi (+20 more)

### Community 15 - "KDE Peak Picking Kernels"
Cohesion: 0.19
Nodes (13): _align_kde_mz_grids(), _append(), _compute_max(), _compute_max_with_ext_correction(), _fwhm_interp(), _fwhm_interp_old(), mspeaks_KD(), _peakpicker_core() (+5 more)

### Community 17 - "Signal Alignment Utilities"
Cohesion: 0.12
Nodes (6): check_xy(), convert_peak_values_to_index(), del_hdf5(), find_nearest_index(), mspeaks_KD_legacy(), shift()

### Community 18 - "Changelog Workflow"
Cohesion: 0.10
Nodes (20): runs-on, steps, GITHUB_REPO, OUTPUT, jobs, changelog, name, on (+12 more)

### Community 21 - "KDE m/z Correction"
Cohesion: 0.12
Nodes (8): apply_kde_mzcorrection(), chunking_datasets(), _consensing_peaks(), _consensus_peaks_summary(), _frequency_filtration(), _nunique_summary(), Peak_assignment(), split_array_by_num_distance()

### Community 37 - "KDE Distribution Estimation"
Cohesion: 0.22
Nodes (4): _compute_KDE(), segment_probability_distribution(), _set_KDE_X_plot(), split_pdtable_by_peaks_gap()

### Community 40 - "OpenCode Permissions"
Cohesion: 0.40
Nodes (4): permission, bash, edit, $schema

## Knowledge Gaps
- **7 isolated node(s):** `pelmesha`, `Custom User Processing`, `apply_kde_mzcorrection`, `Multiprocessing Manager Plan`, `batch_function` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 440 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **31 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Configs` connect `Configs Base Class` to `DataSet ROI Access API`, `Nested Config Updating`, `Pipeline Configurator Serialization`, `Config Parameter Deletion`, `Peak Picking Preprocessing`, `Config Serialization Utilities`, `Third-Party Dependencies`, `Parameter Group Mapping`, `HTML Config Rendering`, `Config Update & Disabling`, `Docstring Reference Resolution`, `Parameter Inspection & Binding`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `DataSet` (e.g. with `Configs` and `KDEConfigs`) actually correct?**
  _`DataSet` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `pelmesha`, `Custom User Processing`, `apply_kde_mzcorrection` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Architecture & Data Model` be split into smaller, more focused modules?**
  _Cohesion score 0.06497175141242938 - nodes in this community are weakly interconnected._
- **Why does `DataSet` connect `DataSet ROI Access API` to `DataSet Source Management`, `DataSet Table Rendering`, `Configs Base Class`, `Config Models & Validation`, `Pipeline Configurator Serialization`, `DataSource ROI Access`, `Processing Pipeline Orchestration`, `Plotting & Visualization`, `Third-Party Dependencies`, `Index Segment Utilities`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `PreparedDataSource` (e.g. with `Indexator` and `SliceIndexator`) actually correct?**
  _`PreparedDataSource` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Should `BaseLoader Abstract Interface` be split into smaller, more focused modules?**
  _Cohesion score 0.050505050505050504 - nodes in this community are weakly interconnected._