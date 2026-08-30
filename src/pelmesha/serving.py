from pelmesha.cookbook import Configs, PreparedDataSource, PipelineConfigurator, KDEConfigs
from pelmesha.filling import DataSource
from pelmesha.dough import Indexator, SliceIndexator
from pelmesha.kneading import _compute_KDE
from pelmesha.utensils import _summerize_kde_mz, _consesusing_peaks, apply_kde_mzcorrection, _frequency_filtration, _consensus_peaks_summary, show_df, _nunique_summary
from sklearn.preprocessing import normalize
from itertools import pairwise
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import yaml
import warnings
from functools import partial
from typing import Callable
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from h5py import File
from urllib.parse import quote
import sys

class DataSet:
    # TODO:
    # 1. Add a way to delete all already-processed data of the datasets added
    #    to a DataSet instance at once, with a yes/no confirmation prompt
    #    (ideally without requiring new dependencies).
    # 2. Consider showing, for an m/z-correction result, which datasets were
    #    used to build it, likely by loading the native Parquet data so the
    #    processing that happened can be fully reconstructed.
 
    """
    Central class for managing multiple mass spectrometry data sources.

    Each registered source is wrapped in a :class:`PreparedDataSource` and
    stored in ``self.sources`` keyed by its sample name. The class provides
    source discovery and addition, per-source processing and peak picking,
    KDE-based peak-density estimation, aggregation of several sources into a
    unified feature matrix, reference-peak handling, and coordinate
    collection.

    Key capabilities:
    1. Discovers and registers data sources from a list of paths or
       directories.
    2. Prevents sample-name conflicts and duplicate file additions.
    3. Manages processing and KDE configurations for every source, saving
       them next to each source file.
    4. Runs the processing/peak-picking pipeline for selected samples and
       ROIs.
    5. Aggregates the peak lists of all requested samples/ROIs into a single
       feature matrix, with optional m/z correction, filtering, pivoting,
       and coordinate merging.
    6. Supports generating a reference peak list from a reference source and
       using it to align the remaining samples.

    Parameters
    ----------
    sources : list, dict, or None, optional
        Optional initial sources. Can be:
        - A list of file paths (configs are loaded from disk if available).
        - A dict ``{path: config_or_None}`` where ``config`` is a config or
          a path to a YAML file.
        - ``None`` (empty DataSet; add sources later).
    RamGb_limit_usage : int or float, optional
        RAM limit in GB used for batch processing. Default ``2``.
    """
    _EXCLUDED_EXTENSIONS = {'ingredients.hdf5', 'specdata.hdf5', 'peaklists.hdf5',"peaks_density.hdf5"}

    def __init__(self,
                 sources = None,
                 configs = None,
                 kde_configs: dict[str,KDEConfigs] | KDEConfigs | str = None,
                 RamGb_limit_usage: float = 2,
                 rebuild_metadata: bool = False,
                 **kwargs):
        """
        Initialize a DataSet instance.

        Creates an empty ``sources`` dictionary, stores the RAM usage limit,
        and (optionally) adds the provided initial sources. If ``sources`` is
        not ``None``, they are passed to :meth:`add_sources`.

        Parameters
        ----------
        sources : list | dict | None, optional
            Initial sources to add. See :meth:`add_sources` for accepted formats.
            Default ``None``.
        configs : dict | str | PipelineConfigurator | None, optional
            Processing config to apply to the added sources. Default ``None``.
        kde_configs : KDEConfigs | dict | None, optional
            KDE configs to apply to the added sources. Default ``None``.
        RamGb_limit_usage : int | float, optional
            RAM limit in GB used for batch processing. Default ``2``.
        """
        self.sources = {}
        self.RamGb_limit_usage = RamGb_limit_usage
        self._reference_file_path = None
        self.reference_source: PreparedDataSource = None
        self._reference_peaks = None
        self._reference_peaks_weights = None

        if sources is not None:
            self.add_sources(sources, configs, kde_configs, **kwargs)
            for source in self.sources.values():
                source._datasource.create_metafile(rebuild_metadata=rebuild_metadata)
        elif sources is None and (configs is not None or kde_configs is not None or kwargs):
            warnings.warn("No sources provided. Configs or KDE configs are skipped. Use add_sources to add sources with configs.")

        print(f"DataSet is initialized. Current data samples:")
        print(self)
    
    #################################################################
    # Reference file methods                                        #
    #################################################################
    def set_reference_source(self,
                             source: DataSource | PreparedDataSource | str,
                             configs: PipelineConfigurator | dict[str, PipelineConfigurator] | None = None,
                             kde_configs: KDEConfigs | dict[str, KDEConfigs] | None = None,
                             rebuild_metadata: bool = False,
                             **kwargs):
        """
        Set the reference source used to generate the reference peak list.

        If *source* is a raw :class:`DataSource`, it is wrapped into a
        :class:`PreparedDataSource`. If it is already prepared, its configs
        and KDE configs are (re)loaded or updated from the provided values.

        Parameters
        ----------
        source : DataSource | PreparedDataSource
            The data source to use as a reference.
        configs : PipelineConfigurator | dict[str, PipelineConfigurator] | None, optional
            Processing configs to apply. Default ``None``.
        kde_configs : KDEConfigs | dict[str, KDEConfigs] | None, optional
            KDE configs to apply. Default ``None``.
        **kwargs
            Additional keyword arguments forwarded to config resolution.
        """
        if isinstance(source, PreparedDataSource):
            KDEkwargs, Configkwargs = source._resolve_kwargs(kwargs)
            if configs is not None:
                source._load(configs, **Configkwargs)
            else:
                source.update(**Configkwargs)
            if kde_configs is not None:
                source._load_kde_configs(kde_configs, **KDEkwargs)
            else:
                source.update_kde(**KDEkwargs)
            source._datasource.create_metafile(rebuild_metadata=rebuild_metadata)
        else:
            if isinstance(source, [DataSource,str]):
                source = PreparedDataSource(source, configs, kde_configs, rebuild_metadata = rebuild_metadata, **kwargs)

        self.reference_source = source

    def save_all_reference_configs(self):
        """
        Save the current configs for source to its ``processed_pelmesha/REFERENCE_<sample_name>_processing_recipe.yaml`` and ``processed_pelmesha/REFERENCE_<sample_name>_kde_recipe.yaml`` file.
        """
        pathsave_base = self.reference_source._default_save_path(prefix="REFERENCE")
        self.reference_source.dump(pathsave_base+"_processing_recipe.yaml")
        self.reference_source.dump_kde_configs(pathsave_base+"_kde_recipe.yaml")

    def get_reference_peaks(self,
                            roi: str | list[str] | None = None,
                            step: int = 500,
                            num_peaks_per_step: int = 5,
                            min_occurence: float = 0.1,
                            return_weight: bool = True,
                            free_cpus: int = 1,
                            Ram_GB_limit: float = 2,
                            dtypeconv: np.dtype | str | None = None):
        """
        Get reference peaks for a given ROI.

        Runs peak picking and KDE density estimation for the reference
        source, corrects m/z values, and selects the most common peaks
        within each m/z step. Optionally stores their occurrence weights.

        Parameters
        ----------
        roi : str | list[str] | None, optional
            ROI name or list of ROI names (default ``None`` - all ROIs).
        step : int, optional
            Width of the m/z window used to collect candidate peaks (default ``500``).
        num_peaks_per_step : int, optional
            Number of most common peaks kept per m/z step (default ``5``).
        min_occurence : float, optional
            Minimum normalized occurrence weight a peak must have to be kept
            (default ``0.1``).
        return_weight : bool, optional
            Whether to store the occurrence weights of selected peaks
            (default ``True``).
        free_cpus : int, optional
            Number of CPUs to reserve for other tasks (default ``1``).
        Ram_GB_limit : float, optional
            RAM limit in GB for batch processing (default ``2``).
        dtypeconv : np.dtype | str | None, optional
            Data type conversion for processing (default ``None``).

        Returns
        -------
        None
            Populates ``_reference_peaks`` (the selected m/z list) and, if
            ``return_weight``, ``_reference_peaks_weights`` (their occurrence
            weights) attributes in place.
        """
        ref_source = self.reference_source
        if ref_source is None:
            raise ValueError(
                "Reference source is not set. Call `set_reference_source` "
                "before requesting reference peaks."
            )

        if step <= 0:
            raise ValueError(f"`step` must be positive, got {step}.")
        if num_peaks_per_step <= 0:
            raise ValueError(
                f"`num_peaks_per_step` must be positive, got {num_peaks_per_step}."
            )
        if not 0 < min_occurence <= 1:
            raise ValueError(
                f"`min_occurence` must be in the (0, 1] range, got {min_occurence}."
            )
        if free_cpus < 0:
            raise ValueError(f"`free_cpus` cannot be negative, got {free_cpus}.")

        sample = ref_source.sample_name
        cpu_num = max(1, cpu_count() - free_cpus)
        roi_metadata = ref_source.roi_metadata
        dtypeconv = np.dtype(
            dtypeconv
            if dtypeconv is not None
            else ref_source._datasource.metadata.iloc[0]["dtype_raw"]
        )

        # Normalize the ROI selection to a list of ROI names.
        if roi is None:
            roi = list(roi_metadata.index)
        elif isinstance(roi, str):
            roi = [roi]
        missing = [r for r in roi if r not in roi_metadata.index]
        if missing:
            raise ValueError(
                f"Unknown ROI(s) for the reference source: {missing}. "
                f"Available ROIs: {list(roi_metadata.index)}."
            )

        pipeline = Pipeline(ref_source)
        print(f"Processing REFERENCE SAMPLE {sample}...")

        sample_peaks: dict[str, pd.DataFrame] = {}
        kde_mz_list: list[np.ndarray] = []
        kde_density_list: list[np.ndarray] = []
        multiindex_keys: list[tuple[str, str]] = []

        for r in roi:
            peakpicking_stream = pipeline._multistream_pipeline(
                pipeline._peakpick_wrapper,
                roi=r,
                cpu_num=cpu_num,
                Ram_GB_limit=Ram_GB_limit,
                dtypeconv=dtypeconv,
            )
            multiindex_keys.append((sample, r))
            _, headers_list = next(peakpicking_stream)

            dict_peak = {}
            for n, peaklists in enumerate(peakpicking_stream):
                dict_peak[n] = peaklists
            if not dict_peak:
                raise RuntimeError(
                    f"Peak picking produced no data for ROI '{r}' of sample "
                    f"'{sample}'."
                )

            roi_peaklists = pd.DataFrame(np.vstack(tuple(dict_peak.values())), columns = headers_list).loc[:,['spectra_ind','mz','FWHM']].astype({"spectra_ind": int})
            sample_peaks[r] = roi_peaklists

            discret_coeffs = roi_metadata.loc[r, "discret_coeffs"]
            X_plot, Y_plot = _compute_KDE(
                roi_peaklists,
                discret_coeffs,
                cpu_num,
                **ref_source.roi_kde_configs[r].to_dict,
            )
            if X_plot.size == 0:
                continue
            # Normalize the probability distribution.
            Y_plot = normalize(Y_plot.reshape(1, -1), norm="l1").squeeze()
            kde_mz_list.append(X_plot)
            kde_density_list.append(Y_plot)
        if len(kde_mz_list) == 0:
            warnings.warn(
                f"No peaks PDF produced for sample '{sample}'."
                f"Reference peaklist are not generated"
            )
            return
        kde_mz, kde_density = _summerize_kde_mz(kde_mz_list, kde_density_list)

        if len(sample_peaks) > 1:
            feature_matrix = pd.concat(
                sample_peaks.values(),
                keys=multiindex_keys,
                names=["sample", "roi"],
            )
        else:
            feature_matrix = sample_peaks[r]

        feature_matrix = apply_kde_mzcorrection(
            feature_matrix, kde_mz, kde_density, cpu_num
        )

        # Occurrence weights per corrected m/z across all spectra.
        weights = (
            feature_matrix["spectra_ind"]
            .groupby(feature_matrix["mz"], observed=True)
            .size()
            .astype(float)
        )
        if weights.empty:
            raise RuntimeError(
                "No peaks remained after m/z correction for the reference "
                f"sample '{sample}'."
            )
        weights = weights / weights.max()

        selected_mz = list(weights[weights > min_occurence].index)
        if not selected_mz:
            raise RuntimeError(
                "No reference peaks passed the `min_occurence` threshold "
                f"({min_occurence}) for sample '{sample}'."
            )

        # Collect the most common peaks within each m/z window.
        align_list: list[float] = []
        window_edges = list(
            pairwise(
                np.arange(min(selected_mz), max(selected_mz) + step + 1, step)
            )
        )
        for low, high in window_edges:
            in_window = weights[(weights.index >= low) & (weights.index < high)]
            align_list.extend(
                list(
                    in_window.sort_values(ascending=False)
                    .head(num_peaks_per_step)
                    .index
                )
            )

        if not align_list:
            raise RuntimeError(
                f"No peaks selected for any m/z window for sample '{sample}'."
            )
        print(f'Resulted number of reference peaks: {len(align_list)}')
        if return_weight:
            self._reference_peaks_weights = weights.loc[align_list].to_list()
        self._reference_peaks = align_list
    def set_align_peaks_from_ref(self,
                                samples: list[str] | None = None,
                                rois: str | list[str] = None,
                                sample_rois_map: dict[str, list[str]] | None = None,
                                set_weights: bool = True):
        """
        Assign the stored reference peaks (and optionally their weights) to
        the selected samples/ROIs as alignment targets.

        Builds the ``{sample: [roi, ...]}`` selection via
        :func:`pelmesha.utensils._resolve_sample_rois_map`, then updates each
        matching data source so its ROIs align against ``self._reference_peaks``.
        Finally, saves the updated reference configs to disk.

        Parameters
        ----------
        samples : list[str] | None, optional
            Sample names to align. Default ``None`` (all samples).
        rois : str | list[str] | None, optional
            ROI name(s) to align for the selected samples. Default ``None``
            (all ROIs of each sample).
        sample_rois_map : dict[str, list[str]], optional
            Explicit ``{sample_name: [roi_name, ...]}`` selection. Takes
            precedence over *samples*/*rois* when provided. Default ``{}``.
        set_weights : bool, optional
            Whether to also assign the reference peak weights. Default
            ``True``.

        Notes
        -----
        If no reference peaks have been computed yet
        (``self._reference_peaks is None``), a warning is emitted and the
        method returns without doing anything.
        """
        
        if self._reference_peaks is None:
            warnings.warn(f"No reference peaks are set. Please get reference peaks first by calling `get_reference_peaks`.")
            return
        sample_rois_map = self._resolve_sample_rois_map(samples, rois, sample_rois_map)
        if not sample_rois_map:
            warnings.warn(f"No samples selected. Skipping.")
            return
            
        for sample, rois in sample_rois_map.items():
            datasource = self.sources[sample]
            datasource.update(rois = rois, align_peaks = self._reference_peaks)
            if set_weights and self._reference_peaks_weights is not None:
                datasource.update(rois = rois, align_pweights = self._reference_peaks_weights)
        self.save_all_reference_configs()
    @property
    def reference_file_path(self): 
        """Path to the reference data file used for reference peak list generation."""
        return self._reference_file_path

    @reference_file_path.setter
    def reference_file_path(self, path): 
        """Set reference file path. Resets reference peaks if config changes."""
        if path != self._reference_file_path:
            self._reference_file_path = path
            self._reference_peaks = None
            self._reference_peaks_weights = None

    @property
    def reference_config(self):
        """Config dict for reference file processing (may differ from analysis config)."""
        return self._reference_config

    @reference_config.setter
    def reference_config(self, config):
        """Set reference config. Resets cached reference peaks."""
        self._reference_config = config
        self._reference_peaks = None
        self._reference_peaks_weights = None

    #################################################
    # Sources methods                               #
    #################################################
    def add_sources(self, source, config = None, kde_configs = None, extensions = None, **kwargs):
        """
        Add one or more data sources to the DataSet.

        Accepts a single file path, a list of paths, a directory, or a dict
        mapping paths to per-source configs. Directories are searched for
        supported data files using :meth:`add_sources_from_paths`.

        Parameters
        ----------
        source : str | list | tuple | dict
            A single file path, a list/tuple of paths, or a dict
            ``{path: config_or_None}``. Dict entries may also be directories.
        config : dict | str | None, optional
            Config dict or path to a YAML file. Ignored for dict ``source``.
            If ``None``, attempts to load a saved config from the source directory.
        kde_configs : KDEConfigs | dict | None, optional
            KDE configs to apply to the added sources. Default ``None``.
        extensions : list[str] | None, optional
            File extensions to search for when *source* is a directory.
            If ``None``, uses the module default supported extensions.

        Raises
        ------
        FileNotFoundError
            If a source path does not exist.
        ValueError
            If a source path has already been added.
        TypeError
            If ``source`` is not of a supported type.
        """
        if isinstance(source, dict):
            for path, cfg in source.items():
                if os.path.isdir(path):
                    self.add_sources_from_paths([path],extensions,cfg)
                else:
                    self._add_single_source(path, cfg)
        elif isinstance(source, (list, tuple)):
            self.add_sources_from_paths(source,extensions,config, kde_configs, **kwargs) 
        elif isinstance(source, str):
            if os.path.isdir(source):
                self.add_sources_from_paths([source],extensions,config, kde_configs, **kwargs)
            else:
                self._add_single_source(source, config, kde_configs, **kwargs)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def __call__(self, source, config = None, kde_configs = None, extensions = None):
        """
        Callable interface — same as :meth:`add_source`.

        Usage: ``dataset(path, config)`` or ``dataset({path: config})``.
        """
        self.add_sources(source, config, kde_configs, extensions)
        return self

    def _add_single_source(self, path, config = None, kde_configs = None, **kwargs):
        """Internal: add a single source with duplicate protection."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source path does not exist: {path}")
        sample_name = os.path.splitext(os.path.basename(path))[0]
        folder_name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if folder_name != sample_name:
            sample_name = folder_name + "_" + sample_name

        source = DataSource(path, RamGb_limit_usage = self.RamGb_limit_usage)
        if any(ds.file_path == source.file_path for ds in self.sources.values()):
            warnings.warn(f"File '{source.file_path}' has already been added to DataSet. Source re-adding with configs")
        elif sample_name in self.sources:
            raise ValueError(
                f"Sample '{sample_name}' (from {path}) already exists in DataSet. "
                f"Existing path: {self.sources[sample_name].file_path}. "
                f"Please, rename file name '{sample_name}' or folder '{folder_name}'"
            )
        resolved_config = self._resolve_config(config)
        source = PreparedDataSource(source, resolved_config, kde_configs, **kwargs)
        self.sources[sample_name] = source

    @staticmethod
    def _resolve_config(config):
        """
        Resolve the processing config for a given source path.

        1. If ``config`` is a dict — use it in ``Configs``.
        2. If ``config`` is a str path to a YAML — load it.
        3. If ``config`` is ``None`` — load the default from ``Base_configs.yaml``.
        4. If ``config`` is instance ``Congigs`` use it directly 
        :param path: Path to the data source file.
        :param config: User-provided config (dict, str, or None).

        :type path: str
        :type config: dict or str or None

        :return: Resolved config dictionary.
        :rtype: dict
        """
        if isinstance(config, dict):
            return PipelineConfigurator(config)
        if isinstance(config, str):
            if os.path.exists(config):
                return PipelineConfigurator(config)
            else:
                raise FileNotFoundError(f"Config file not found: {config}")
        if isinstance(config, (PipelineConfigurator, Configs)):
            return config
        return {}

    def add_sources_from_paths(self, path_list, extensions = None, config = None, kde_configs = None, **kwargs):
        """
        Search directories for data files with given extensions and add them as sources.

        Automatically excludes files with processed-data extensions (``.hdf5``, etc.).

        :param path_list: List of directories or file paths to search.
        :param extensions: List of file extensions to look for (e.g. ``['.imzML', '.mzXML']``).
            If ``None``, uses ``['.imzML', '.mzXML', '.cdf']``.
        :param config: Config to apply to all discovered sources. If ``None``, each source
            will try to load its own saved config.

        :type path_list: list
        :type extensions: list or None
        :type config: dict or str or None
        """
        if extensions is None:
            from pelmesha.filling import SUPPORTED_FILE_EXTENSIONS
            extensions = SUPPORTED_FILE_EXTENSIONS

        found_paths = []
        for entry in path_list:
            if os.path.isfile(entry):
                ext = os.path.splitext(entry)[1].lower()
                if ext in extensions:
                    found_paths.append(entry)
            elif os.path.isdir(entry):
                for root, _, files in os.walk(entry):
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in extensions:
                            full_path = os.path.join(root, f)
                            if not any(excl in full_path.lower() for excl in self._EXCLUDED_EXTENSIONS):
                                found_paths.append(full_path)

        for path in found_paths:
            try:
                self._add_single_source(path, config, kde_configs, **kwargs)
            except (ValueError, FileNotFoundError) as e:
                warnings.warn(f"Skipping {path}: {e}")

    
    def save_configs(self): 
        """
        Save the current config for each source to its ``processed_pelmesha/<sample_name>_processing_recipe.yaml`` file.
        """
        for ds in self.sources.values():
            ds.save()
    def save_kde_configs(self): 
        """
        Save the current config for each source to its ``processed_pelmesha/<sample_name>_kde_recipe.yaml`` file.
        """
        for ds in self.sources.values():
            ds.dump_kde_configs()
    def check_config_changes(self): 
        """
        Compare current configs with saved configs for each source.

        :return: Dict mapping sample names to change info:
            ``{"sample_name": {"status": "unchanged"|"changed"|"new", "old": dict|None, "new": dict}}``
        :rtype: dict
        """
        changes = {}
        for sample_name, ds in self.sources.items():
            saved_config_path = os.path.join(os.path.dirname(ds.file_path), 'raw_pelmesha', 'config.yaml')
            current_config = getattr(ds, 'config', {})

            if os.path.exists(saved_config_path):
                with open(saved_config_path, 'r') as f:
                    saved_config = yaml.safe_load(f) or {}
                if saved_config == current_config:
                    changes[sample_name] = {"status": "unchanged", "old": saved_config, "new": current_config}
                else:
                    changes[sample_name] = {"status": "changed", "old": saved_config, "new": current_config}
            else:
                changes[sample_name] = {"status": "new", "old": None, "new": current_config}
        return changes
    
    ###############################################################################
    # Processing methods                                                          #
    ###############################################################################       
    def process(self,
                sample_name: list | str | None = None,
                free_cpus: int = 1, 
                draw: bool = True, 
                draw_mz_range: tuple[float, float] | None = None,
                draw_spectrum_idx: int | None = None,
                Ram_GB_limit: float = 2,
                h5chunk_size_MB: int = 10,
                dtypeconv: np.dtype | str | None = None,
                **kwargs) -> None:
        """
        Process all sources in the dataset through the Pipeline.
        
        For each source in the dataset, creates a Pipeline instance and
        runs processing. See Pipeline.process for detailed parameter docs.
        
        Parameters
        ----------
        free_cpus : int, optional
            Number of CPUs to leave free (default 1).
        draw : bool, optional
            Whether to draw processing results (default False).
        draw_mz_range : tuple[float, float] | None, optional
            m/z range for drawing (default None).
        draw_spectrum_idx : int | None, optional
            Spectrum index for drawing (default None).
        Ram_GB_limit : float, optional
            RAM limit in GB (default 2).
        h5chunk_size_MB : int, optional
            HDF5 chunk size in MB (default 10).
        dtypeconv : np.dtype | str | None, optional
            Data type conversion (default None).
        **kwargs
            Additional arguments forwarded to Pipeline.process.
        
        See Also
        --------
        Pipeline.process : The underlying processing method.
        """
        if sample_name is None:
            sample_name = list(self.sources.keys())
        elif isinstance(sample_name, str):
            sample_name = [sample_name]

        for sample in sample_name:
            print(f"Processing SAMPLE {sample}...")
            Pipeline(self.sources[sample]).process(
                free_cpus=free_cpus,
                draw=draw,
                draw_mz_range=draw_mz_range,
                draw_spectrum_idx=draw_spectrum_idx,
                Ram_GB_limit=Ram_GB_limit,
                h5chunk_size_MB=h5chunk_size_MB,
                dtypeconv=dtypeconv,
                **kwargs,
            )
    def peakpick(self,
                sample_name: list | str | None = None,
                free_cpus: int = 1, 
                draw: bool = True, 
                draw_mz_range: tuple[float, float] | None = None,
                draw_spectrum_idx: int | None = None,
                Ram_GB_limit: float = 2,
                h5chunk_size_MB: int = 10,
                dtypeconv: np.dtype | str | None = None,
                **kwargs) -> None:
        """
        Process all sources in the dataset through the Pipeline.
        
        For each source in the dataset, creates a Pipeline instance and
        runs processing. See Pipeline.process for detailed parameter docs.
        
        Parameters
        ----------
        free_cpus : int, optional
            Number of CPUs to leave free (default 1).
        draw : bool, optional
            Whether to draw processing results (default False).
        draw_mz_range : tuple[float, float] | None, optional
            m/z range for drawing (default None).
        draw_spectrum_idx : int | None, optional
            Spectrum index for drawing (default None).
        Ram_GB_limit : float, optional
            RAM limit in GB (default 2).
        h5chunk_size_MB : int, optional
            HDF5 chunk size in MB (default 10).
        dtypeconv : np.dtype | str | None, optional
            Data type conversion (default None).
        **kwargs
            Additional arguments forwarded to Pipeline.process.
        
        See Also
        --------
        Pipeline.process : The underlying processing method.
        """
        if sample_name is None:
            sample_name = list(self.sources.keys())
        elif isinstance(sample_name, str):
            sample_name = [sample_name]

        for sample in sample_name:
            print(f"Processing SAMPLE {sample}...")
            Pipeline(self.sources[sample]).peakpick(
                free_cpus=free_cpus,
                draw=draw,
                draw_mz_range=draw_mz_range,
                draw_spectrum_idx=draw_spectrum_idx,
                Ram_GB_limit=Ram_GB_limit,
                h5chunk_size_MB=h5chunk_size_MB,
                dtypeconv=dtypeconv,
                **kwargs
            )
    def estimate_peak_density_kde(self, 
                                  sample_name: list | str | None = None,
                                  free_cpus: int = 1,
                                  draw: bool = True,
                                  draw_borders: float = 1.5) -> None:
        """
        Estimate peak density using KDE.
        
        Parameters
        ----------
        sample_name : list | str | None, optional
            Sample name (default None).
        free_cpus : int, optional
            Number of CPUs to leave free (default 1).
        draw : bool, optional
            Whether to draw processing results (default False).
        draw_mz_range : tuple[float, float] | None, optional
            m/z range for drawing (default None).
        draw_spectrum_idx : int | None, optional
            Spectrum index for drawing (default None).
        Ram_GB_limit : float, optional
            RAM limit in GB (default 2).
        h5chunk_size_MB : int, optional
            HDF5 chunk size in MB (default 10).
        dtypeconv : np.dtype | str | None, optional
            Data type conversion (default None).
        
        See Also
        --------
        Pipeline.process : The underlying processing method.
        """
        if sample_name is None:
            sample_name = list(self.sources.keys())
        elif isinstance(sample_name, str):
            sample_name = [sample_name]

        for sample in sample_name:
            print(f"Estimating peak density for SAMPLE {sample}...")
            source = self.sources[sample]
            Pipeline(source).estimate_peak_density_kde(free_cpus=free_cpus,
                                                       draw=draw,
                                                       draw_borders=draw_borders)
    
    def feature_matrix(self,
                        samples: list[str] | None = None,
                        rois: str | list[str] | None = None,
                        sample_rois_map: dict[str, list[str]] | None = None,
                        countf: int = 10,
                        duplicates_drop: bool = True,
                        pivot_values: str | list[str] | None = None,  # 'Intensity', ['Intensity', 'Area']
                        fill_values = 0.0,
                        free_cpus: int = 1,
                        draw_borders: float = 1.5,
                        draw: bool = True,
                        save_path: str = None,
                        local_roi_idx: bool = True,
                        merge_with_coords: bool = False) -> pd.DataFrame:
        """
        Generate feature matrix from peak lists.
        
        Parameters
        ----------
        rois : dict
            Dictionary of ROIs to process. Example: {sample_name: [roi_name, ...]}.
        countf : int, optional
            Number of features to keep (default 10).
        duplicates_drop : bool, optional
            Whether to drop duplicates (default True).
        pivot_values : str | list[str], optional
            Values to pivot on (default None).
        fill_values : float, optional
            Value to fill missing values with (default 0.0).
        free_cpus : int, optional
            Number of CPUs to leave free (default 1).
        save_path : str, optional
            Path to save feature matrix (default None).
        
        Returns
        -------
        pd.DataFrame
            Feature matrix.
        """
        
        return Pipeline.feature_matrix(self,
                                       samples = samples,
                                       rois = rois,
                                       sample_rois_map = sample_rois_map,
                                       countf = countf,
                                       duplicates_drop = duplicates_drop,
                                       pivot_values = pivot_values,
                                       fill_values = fill_values,
                                       free_cpus = free_cpus,
                                       draw_borders = draw_borders,
                                       draw = draw,
                                       save_path = save_path,
                                       local_roi_idx = local_roi_idx,
                                       merge_with_coords = merge_with_coords)

    ##################################################################
    # QoL methods                                                    #
    ##################################################################

    def _resolve_sample_rois_map(self,
                                 samples: list[str],
                                 rois: str | list[str],
                                 sample_rois_map: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
        '''
        Resolve sample_rois_map from samples and rois.
        
        Parameters
        ----------
        samples : list[str]
            List of samples.
        rois : str | list[str]
            ROI or list of ROIs.
        sample_rois_map : dict[str, list[str]]
            Sample-ROI map.
        
        Returns
        -------
        dict[str, list[str]]
            Sample-ROI map.
        '''
        if isinstance(rois, str):
            rois = [rois]
        if sample_rois_map is None:
            sample_rois_map = {}
        if not sample_rois_map:
            if samples is None:
                samples = list(self.sources.keys())
            for sample in samples:
                if rois is None:
                    sample_rois_map[sample] = self.sources[sample].rois
                else:
                    sample_rois_map[sample] = []
                    for roi in rois:
                        if roi in self.sources[sample].rois:
                            sample_rois_map[sample].append(roi)
                        else:
                            warnings.warn(f"ROI '{roi}' not found in sample '{sample}'. Skipping.")
                    if not sample_rois_map[sample]:
                        del sample_rois_map[sample]
                        warnings.warn(f"No ROIs selected for sample '{sample}'. Skipping")
        else:
            if samples is not None:
                warnings.warn(f"Both samples and sample_rois_map are specified. Using sample_rois_map.")
            if rois is not None:
                warnings.warn(f"Both rois and sample_rois_map are specified. Using sample_rois_map.")
            sample_list = list(sample_rois_map.keys())    
            for sample in sample_list:
                if sample not in self.sources:
                    warnings.warn(f"Sample '{sample}' not found in DataSet. Skipping.")
                    del sample_rois_map[sample]
                else:
                    datasource = self.sources[sample]
                    ds_rois = datasource.rois
                    filtered_rois = []
                    for roi in ds_rois:
                        if roi in sample_rois_map[sample]:
                            filtered_rois.append(roi)
                    for roi in sample_rois_map[sample]:
                        if roi not in filtered_rois:
                            warnings.warn(f"ROI '{roi}' not found in sample '{sample}'. Skipping.")
                    sample_rois_map[sample] = filtered_rois
                    if not sample_rois_map[sample]:
                        del sample_rois_map[sample]
                        warnings.warn(f"No ROIs selected for sample '{sample}'. Skipping")

        if not sample_rois_map:
            warnings.warn("No samples or ROIs are setted. Returning empty map.")
        return sample_rois_map

    def coordinates(self,
                    samples: list[str] = None,
                    rois: str | list[str] = None, 
                    sample_rois_map: dict[str, list[str]] | None = None,
                    local_roi_idx: bool = True):
        """
        Collect spatial coordinates for all requested samples/ROIs.

        Gathers the coordinates of every (sample, ROI) pair into a single
        concatenated :class:`pandas.DataFrame` with a ``('sample', 'roi')``
        MultiIndex.

        Parameters
        ----------
        rois : dict[str, list[str]], optional
            Mapping of ``{sample_name: [roi_name, ...]}``. Samples not present
            in the dict use all of their ROIs. Default ``{}``.
        local_roi_idx : bool, optional
            If ``True``, re-index each ROI's coordinates locally (0-based per
            ROI) instead of using the global spectrum indices. Default ``True``.

        Returns
        -------
        pandas.DataFrame
            Concatenated coordinates table indexed by ``('sample', 'roi')``.
        """
        multiindex_keys = []
        coords = []
        sample_rois_map = self._resolve_sample_rois_map(samples, rois, sample_rois_map)
    
        for sample, rois in sample_rois_map.items():
            datasource = self.sources[sample]
            for roi in rois:
                multiindex_keys.append((sample, roi))
                coords.append(datasource.get_coords(roi))
                if local_roi_idx:
                    roi_coords = coords[-1].reset_index(drop=True)
                    roi_coords.index.name = coords[-1].index.name        
                    coords[-1] = roi_coords
                    
        return pd.concat(coords, keys=multiindex_keys, names = ['sample', 'roi'])

    def __getitem__(self, sample):
        """
        Return the prepared data source for *sample*.

        Enables ``dataset['sample_name']`` access.

        Parameters
        ----------
        sample : str
            Sample name.

        Returns
        -------
        PreparedDataSource
            The prepared data source registered under *sample*.
        """
        return self.sources[sample]

    def __getattr__(self, sample):
        """
        Attribute access fallback that resolves sample names as attributes.

        Enables ``dataset.sample_name`` as shorthand for
        ``dataset['sample_name']``. Names starting with an underscore are
        never treated as samples.

        Parameters
        ----------
        sample : str
            Attribute name / sample name.

        Returns
        -------
        PreparedDataSource
            The prepared data source for *sample*.

        Raises
        ------
        AttributeError
            If *sample* starts with ``'_'`` or is not a registered source.
        """
        if sample.startswith('_'):
            raise AttributeError(sample)
        if sample in self.sources:
            return self.sources[sample]
        raise AttributeError(f"DataSet has no source '{sample}'")

    def __iter__(self):
        """
        Iterate over all prepared data sources.

        Yields
        ------
        PreparedDataSource
            Each registered data source.
        """
        return iter(self.sources.values())

    def __len__(self):
        """
        Return the number of registered data sources.

        Returns
        -------
        int
            Number of sources in the DataSet.
        """
        return len(self.sources)

    def __contains__(self, item):
        """
        Check whether *item* is a registered sample name.

        Parameters
        ----------
        item : str
            Sample name to test.

        Returns
        -------
        bool
            ``True`` if the sample is registered in the DataSet.
        """
        return item in self.sources

    def audit_processing(self,
                        sources: str | list[str] | None = None,
                        roi: str | list[str] | None = None,
                        draw_mz_range: tuple[float, float] | None = None,
                        draw_spectrum_idx: int | None = None,
                        dtypeconv: np.dtype | None = None):
        """Audit the processing of the data source.

        Parameters
        ----------
        sources : str or list, optional
            The data source to audit. If None, all sources are audited.
        roi : str or list, optional
            The ROI to audit. If None, all ROIs are audited.
        draw_mz_range : tuple of float, optional
            If not None, draw a spectrum within the given m/z range.
        draw_spectrum_idx : int, optional
            If not None, draw the spectrum at the given index.
        dtypeconv : numpy.dtype, optional
            If not None, convert the data to the given dtype.
        """
        if sources is None:
            sources = list(self.sources.keys())
        if isinstance(sources, str):
            sources = [sources]
        for source in sources:
            Drawer(self.sources[source]).audit_processing(roi, draw_mz_range, draw_spectrum_idx, dtypeconv)
    
    # ------------------------------------------------------------------ #
    #  Table representations (__repr__ / _repr_html_)                    #
    #  Single Source of Truth: _generate_table_data()                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _file_url(path):
        """
        Convert a local filesystem path to a ``file:///`` URL that opens
        in the OS file manager (Explorer / Finder / Dolphin).

        Uses ``urllib.parse.quote`` to safely encode spaces, Unicode, and
        other special characters.

        Parameters
        ----------
        path : str
            Absolute filesystem path (e.g. ``C:\\Data\\my sample\\``).

        Returns
        -------
        str
            Properly encoded ``file:///`` URL.
        """
        # Normalise backslashes → forward slashes, then URL-encode
        return "file:///" + quote(path.replace("\\", "/"), safe="/:")

    @staticmethod
    def _escape_html(text):
        """
        Escape HTML special characters (&, <, >, ', ") in *text*.

        Parameters
        ----------
        text : str
            Raw string to escape.

        Returns
        -------
        str
            HTML-safe string.
        """
        table = {
            "&": "&",
            '"': '"',
            "'": "'",
            ">": ">",
            "<": "<",
        }
        return "".join(table.get(c, c) for c in text)

    def _generate_table_data(self):
        """
        Generate structured table data — the **single source of truth**
        for both ``__repr__`` and ``_repr_html_``.

        Column definitions live here only; changing a header label, adding,
        or removing a column automatically updates both representations.

        Returns
        -------
        tuple
            ``(headers, rows, alignments)`` where:

            - **headers** : list of str — column header labels.
            - **rows** : list of list of str — one sub-list per data source,
              each containing the string representation of every column.
            - **alignments** : list of {'left', 'right'} — per-column
              text alignment hint.
        """
        # (header_label,  alignment)
        column_defs = [
            ("Sample name",             "left"),
            ("Mass spectra number",     "right"),
            ("Continious",              "left"),
            ("Peaklists",               "left"),
            ("Processed mass spectra",  "left"),
            ("Peaks density",           "left"),
            ("Directory",               "left"),
            ("Previous configs",        "left")
        ]

        headers = [col[0] for col in column_defs]
        alignments = [col[1] for col in column_defs]

        rows = []
        for sample, source in self.sources.items():
            dir_path = os.path.dirname(source.file_path)
            config_file = os.path.join(dir_path, 'processed_pelmesha', f'{sample}_processing_recipe.yaml')
            row = [
                    sample,                                                                                                                 #Sample name
                    Indexator(np.vstack(source.roi_metadata['idxroi'].to_numpy())).count,                                                   #Mass spectra number
                    source._datasource.dcont,                                                                                               #Continuous
                    os.path.exists(os.path.join(dir_path,'processed_pelmesha', f'{sample}_peaklists.hdf5')),                                #Peaklists
                    os.path.exists(os.path.join(dir_path,'processed_pelmesha', f'{sample}_processed_spectra.hdf5')),                        #Processed mass spectra
                    os.path.exists(source._datasource.peaks_density_path) if source._datasource.peaks_density_path is not None else "No",  #Peaks density
                    dir_path,                                                                                                               #Directory
                    config_file if os.path.exists(config_file) else "—",                                                                    #Previous configs
                    ]
            for i, cell in enumerate(row):
                if isinstance(cell, bool):
                    row[i] = "Yes" if cell else "No"
                elif not isinstance(cell, str):
                    row[i] = str(cell)
            rows.append(row)
        return headers, rows, alignments

    def __repr__(self):
        """
        Text-table representation of the DataSet for the console.

        Columns are dynamically sized; text is left-aligned and numeric
        columns are right-aligned.  Uses ``_generate_table_data()`` as
        the single source of truth.

        Returns
        -------
        str
            A plain-text table with ``|`` separators.
        """
        if not self.sources:
            return "DataSet (empty)"

        headers, rows, alignments = self._generate_table_data()

        # --- dynamic column widths ----------------------------------- #
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # --- separator line ------------------------------------------ #
        sep = " | ".join("-" * w for w in col_widths)

        # --- header line --------------------------------------------- #
        header_parts = []
        for i, header in enumerate(headers):
            if alignments[i] == "right":
                header_parts.append(header.rjust(col_widths[i]))
            else:
                header_parts.append(header.ljust(col_widths[i]))
        header_line = " | ".join(header_parts)

        # --- data lines ---------------------------------------------- #
        lines = [header_line, sep]
        for row in rows:
            parts = []
            for i, cell in enumerate(row):
                if alignments[i] == "right":
                    parts.append(cell.rjust(col_widths[i]))
                else:
                    parts.append(cell.ljust(col_widths[i]))
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    def _repr_html_(self):
        """
        HTML-table representation of the DataSet for Jupyter Notebook.

        Uses ``_generate_table_data()`` as the single source of truth.
        Features:

        - Inline CSS for borders, padding, and a dark header.
        - Zebra-stripe alternating row colours.
        - Monospace font for the numeric "Mass spectra number" column.
        - Hyperlinks for the "Directory" and "Previous configs" columns.

        Returns
        -------
        str
            A valid HTML ``<table>`` element as a single string.
        """
        if not self.sources:
            return "<i>DataSet (empty)</i>"

        headers, rows, alignments = self._generate_table_data()

        # Locate special columns by header label (avoids hard-coding indices)
        col_map = {h: i for i, h in enumerate(headers)}
        path_idx = col_map.get("Directory")
        cfg_idx  = col_map.get("Previous configs")
        num_idx  = col_map.get("Mass spectra number")

        html = [
            '<table style="border-collapse: collapse; '
            'font-family: sans-serif; font-size: 13px;">'
        ]

        # --- thead --------------------------------------------------- #
        html.append("  <thead>")
        html.append("    <tr>")
        for header in headers:
            html.append(
                '      <th style="border: 1px solid #999; padding: 6px 10px; '
                'background-color: #4a4a4a; color: #fff; font-weight: bold; '
                'text-align: left; white-space: nowrap;">'
                f'{self._escape_html(header)}</th>'
            )
        html.append("    </tr>")
        html.append("  </thead>")

        # --- tbody --------------------------------------------------- #
        html.append("  <tbody>")
        for row_idx, row in enumerate(rows):
            bg = "#f9f9f9" if row_idx % 2 == 0 else "#ffffff"
            html.append(f'    <tr style="background-color: {bg};">')

            for col_idx, cell in enumerate(row):
                # -- base style --------------------------------------- #
                style = (
                    "border: 1px solid #ccc; padding: 4px 10px; "
                    "white-space: nowrap;"
                )
                if col_idx == num_idx:
                    style += " font-family: monospace; text-align: right;"
                else:
                    style += " text-align: left;"

                # -- cell content ------------------------------------- #
                if cell == "—":
                    content = "—"
                elif col_idx == path_idx:
                    content = (
                        f'<a href="{self._file_url(cell)}" target="_blank" '
                        f'style="color: #1a73e8; text-decoration: none;">'
                        f'{self._escape_html(cell)}</a>'
                    )
                elif col_idx == cfg_idx:
                    if cell != "—":
                        content = (
                            f'<a href="{self._file_url(cell)}" target="_blank" '
                            f'style="color: #1a73e8; text-decoration: none;">'
                            f'{"Open"}</a>'
                        )
                    else:
                        content = self._escape_html(cell)
                else:
                    content = self._escape_html(cell)

                html.append(f'      <td style="{style}">{content}</td>')

            html.append("    </tr>")
        html.append("  </tbody>")
        html.append("</table>")

        return "\n".join(html)

    def keys(self):
        """
        Return the keys (sample names) of the internal sources dict.

        Returns
        -------
        dict_keys
            View of the sample names.
        """
        return self.sources.keys()

    def values(self):
        """
        Return the values (prepared data sources) of the internal dict.

        Returns
        -------
        dict_values
            View of the prepared data sources.
        """
        return self.sources.values()

    def items(self):
        """
        Return (sample_name, prepared data source) pairs.

        Returns
        -------
        dict_items
            View of the ``(sample_name, data source)`` pairs.
        """
        return self.sources.items()

    def close(self, samples = None):
        """
        Close one or all data sources.

        :param samples: Sample name or list of sample names to close.
            If ``None``, closes all sources.
        :type samples: str or list or None
        """
        if samples is not None:
            if isinstance(samples, list):
                for s in samples:
                    if s in self.sources:
                        self.sources[s].close()
            else:
                if samples in self.sources:
                    self.sources[samples].close()
        else:
            for s in self.sources.values():
                s.close()



class Drawer():
    def __init__(self, datasource: "str | DataSource | PreparedDataSource"):
        """
        Initialize a Drawer for a single data source.

        Accepts a raw :class:`DataSource`, a file path string, or an already
        prepared :class:`PreparedDataSource`. If a raw source with a known
        ``configs_path`` is given, a :class:`PreparedDataSource` is created
        for processing; otherwise only raw-data drawing is possible.

        Parameters
        ----------
        datasource : str | DataSource | PreparedDataSource
            Path to the source file, a raw data source, or a prepared data
            source.

        Raises
        ------
        ValueError
            If no processed spectra, peaklists, or configs can be found for a
            raw data source.
        """
        if isinstance(datasource, (DataSource, str)):
            if isinstance(datasource, str):
                datasource = DataSource(datasource)
            self.datasource = datasource
            if datasource.configs_path is not None:
                self.prepdata = PreparedDataSource(datasource, datasource.configs_path)
            else:
                self.prepdata = None
                if self.processed_spectra_path is None and self.peaklists_path is None:
                    raise ValueError("No processed spectra, peaklist or configs found. Only raw datasource")
        elif isinstance(datasource, PreparedDataSource):
            self.prepdata = datasource
            self.datasource = datasource._datasource
        self.peaks_density_path = self.datasource.peaks_density_path
        self.processed_spectra_path = self.datasource.processed_spectra_path
        self.peaklists_path = self.datasource.peaklists_path

    def _draw_raw(self,
                  roi: str | None = None,
                  mz_range: tuple[float, float] | None = None,
                  spectrum_idx: int | None = None,
                  axes: plt.Axes | None = None):
        datasource = self.datasource
        if axes is None:    
            plt.figure().set_figwidth(25)
            plt.gcf().set_figheight(5)
            axes = plt.gca()
            plt.xlabel("m/z")
            plt.ylabel("Intensity")
            plt.minorticks_on()
            plt.grid(visible=True,which="both")
            if roi is None:
                roi = datasource._get_roi(spectrum_idx)
            plt.title(f"Sample: {datasource.sample_name}, roi: {roi}")
        else:
            plt.sca(axes)
        diapcalc = lambda mz, plot_mz_range: (np.array(mz>plot_mz_range[0]) & np.array(mz<plot_mz_range[1])) if plot_mz_range is not None else range(len(mz))

        # Draw raw
        mz_raw, intens_raw = datasource.get_spectrum(spectrum_idx)

        diap_raw = diapcalc(mz_raw, mz_range)
        plt.plot(mz_raw[diap_raw], intens_raw[diap_raw],alpha=0.75, label = f"Raw mass spectrum N{spectrum_idx}")
        # plt.legend()
        plt.xlim(mz_range)
        
    def _draw(self,
              mz: np.ndarray,
              intens: np.ndarray,
              peaklist: np.ndarray | None = None,
              headers: list[str] | None = None,
              roi: str | None = None,
              mz_range: tuple[float, float] | None = None,
              spectrum_idx: int | None = None,
              axes: plt.Axes | None = None,
              draw_raw: bool = True):
        """
        Draw raw and processed mass spectra on a single figure.

        Overlays the raw spectrum, the processed spectrum, and (optionally)
        the detected peaks together with their left/right peak bases within
        the requested m/z window. Draws on *axes* if provided, otherwise on a
        new figure.

        Parameters
        ----------
        mz : numpy.ndarray
            m/z values of the processed spectrum.
        intens : numpy.ndarray
            Intensity values of the processed spectrum.
        peaklist : numpy.ndarray | None, optional
            Peak list; an array of rows with ``headers`` or a DataFrame.
            Default ``None``.
        headers : list[str] | None, optional
            Column headers for a raw peaklist array. Default ``None``.
        roi : str | None, optional
            ROI name, used for the plot title. Default ``None``.
        mz_range : tuple[float, float] | None, optional
            m/z window to display. Default ``None`` (full range).
        spectrum_idx : int | None, optional
            Spectrum index, used for the plot title and peak queries.
            Default ``None``.
        axes : matplotlib.axes.Axes | None, optional
            Axes to draw on. If ``None``, a new figure is created.
            Default ``None``.
        """
        datasource = self.datasource
        if axes is None:    
            plt.figure().set_figwidth(25)
            plt.gcf().set_figheight(5)
            axes = plt.gca()
            plt.xlabel("m/z")
            plt.ylabel("Intensity")
            plt.minorticks_on()
            plt.grid(visible=True,which="both")
            plt.title(f"Sample: {datasource.sample_name}, roi: {roi}")
        else:
            plt.sca(axes)
        diapcalc = lambda mz, plot_mz_range: (np.array(mz>plot_mz_range[0]) & np.array(mz<plot_mz_range[1])) if plot_mz_range is not None else range(len(mz))

        # Draw raw
        if draw_raw:
            self._draw_raw(roi = roi, spectrum_idx = spectrum_idx, axes = axes, mz_range = mz_range)
        
        mz_raw, _ = datasource.get_spectrum(spectrum_idx)

        if mz is None:
            mz = mz_raw
        if mz_range is None:
            mz_range = (mz[0], mz[-1])

        # Draw processed
        diap = diapcalc(mz, mz_range)
        plt.plot(mz[diap], intens[diap],alpha=0.75, label = f"Processed mass spectrum N{spectrum_idx}")
        
        # Draw peaklist
        if peaklist is not None:
            if isinstance(peaklist, np.ndarray):
                peaklist = pd.DataFrame(peaklist, columns = headers)
            peaklist = peaklist.astype({"spectra_ind": int})
            # .plot(x="mz", y="Intensity",ax = plt.gca(),)
            plt.plot(*peaklist.query("mz>@mz_range[0] and mz<@mz_range[1] and spectra_ind == @spectrum_idx").loc[:,['mz','Intensity']].values.T, "o", markersize=9, fillstyle='none', mew = 1.25, color = "k", label = "Peaks")
            left_intens=[]
            for left_base in peaklist.query("PextL>@mz_range[0] and PextL<@mz_range[1] and spectra_ind == @spectrum_idx")['PextL']:
                left_intens.append(intens[mz>=left_base][0])
            right_intens = []
            for right_base in peaklist.query("PextR>@mz_range[0] and PextR<@mz_range[1] and spectra_ind == @spectrum_idx")['PextR']:
                right_intens.append(intens[mz<=right_base][-1])
            plt.plot(peaklist.query("PextL>@mz_range[0] and PextL<@mz_range[1] and spectra_ind == @spectrum_idx")['PextL'],
            left_intens,'^', color = 'g', label = 'Left peak base')
            plt.plot(peaklist.query("PextR>@mz_range[0] and PextR<@mz_range[1] and spectra_ind == @spectrum_idx")['PextR'],
            right_intens,'v', color = 'r', label = 'Right peak base')
        old_legend = axes.get_legend()
        if old_legend:
            old_handles = old_legend.legend_handles
            old_labels = [text.get_text() for text in old_legend.get_texts()]
        else:
            old_handles, old_labels = [], []
        handles, labels = axes.get_legend_handles_labels()
        old_handles += handles 
        old_labels += labels
         
        axes.legend(old_handles, old_labels)
        plt.xlim(mz_range)

    def audit_processing(self,
                         roi: str | list | None = None,
                         draw_mz_range: tuple[float, float] | None = None,
                         draw_spectrum_idx: int | None = None,
                         dtypeconv: np.dtype | None = None,
                         axes: plt.Axes | None = None,
                         configs_path: str | None = None,
                         draw_raw: bool = True):
        """
        Audit the processing of the data source by re-running the pipeline.

        Re-processes the requested spectrum (or a randomly chosen one per ROI)
        and draws the raw vs processed result to visually verify the effect of
        the processing configuration.

        Parameters
        ----------
        roi : str | list | None, optional
            ROI name or list of ROI names. Default ``None`` (all ROIs).
        draw_mz_range : tuple[float, float] | None, optional
            m/z range for drawing. Default ``None``.
        draw_spectrum_idx : int | None, optional
            Spectrum index to audit. Default ``None`` (random per ROI).
        dtypeconv : numpy.dtype | None, optional
            Data type conversion for processing. Default ``None``.
        axes : matplotlib.axes.Axes | None, optional
            Axes to draw on. Default ``None``.
        configs_path : str | None, optional
            Path to configs to use instead of the stored ones. Default ``None``.
        """
        if roi is None:
            roi = list(self.datasource.roi_metadata.index)
        elif isinstance(roi, str):
            roi = [roi]
        elif isinstance(roi, list):
            pass
        else:
            raise ValueError("Invalid roi")
        if configs_path:
            prepdata = PreparedDataSource(self.datasource, configs_source= configs_path)
        else:
            prepdata = self.prepdata
        pipeline = Pipeline(prepdata)
        datasource = self.datasource
        roi_metadata = datasource.roi_metadata
        for r in roi:
            if draw_spectrum_idx is None:
                rmeta = roi_metadata.loc[r]
                idxs = Indexator(rmeta["idxroi"])
                spectrum_idx = list(idxs)[np.random.randint(0,idxs.count)]
            else:
                spectrum_idx = draw_spectrum_idx

            processing_stream = pipeline._multistream_pipeline(Pipeline._alldata_wrapper, r, idxs = spectrum_idx, dtypeconv=dtypeconv)
            _, headers = next(processing_stream)
            mz_dict, intensity_dict, peaklist = next(processing_stream)
            mz = mz_dict[spectrum_idx]
            mz_crop_range  = prepdata[r]["modify_raw_spectrum"]["mz_crop_range"]
            if mz_crop_range is not None and draw_mz_range is None:
                low_mz, high_mz = mz_crop_range
                if low_mz is None:
                    low_mz = mz.min()
                if high_mz is None:
                    high_mz = mz.max()
                roi_draw_mz_range = (low_mz, high_mz)
            else:
                roi_draw_mz_range = draw_mz_range
            self._draw(mz_dict[spectrum_idx], intensity_dict[spectrum_idx].squeeze(), peaklist, headers, r, roi_draw_mz_range, spectrum_idx, axes, draw_raw)

    def draw_processed_data(self,
                            roi: str | list | None = None,
                            draw_mz_range: tuple[float, float] | None = None,
                            draw_spectrum_idx: int | None = None,
                            axes: plt.Axes | None = None):
        """
        Draw processed spectra (and peaklists if available) for the ROIs.

        Loads processed spectra from the stored HDF5 files, or re-processes
        them on the fly if the files are missing, and draws each requested
        spectrum via :meth:`_draw`.

        Parameters
        ----------
        roi : str | list | None, optional
            ROI name or list of ROI names. Default ``None`` (all ROIs).
        draw_mz_range : tuple[float, float] | None, optional
            m/z range for drawing. Default ``None``.
        draw_spectrum_idx : int | None, optional
            Spectrum index to draw. Default ``None`` (random per ROI).
        axes : matplotlib.axes.Axes | None, optional
            Axes to draw on. Default ``None``.

        Raises
        ------
        ValueError
            If no processed spectra file or configs are available.
        """
        if roi is None:
            roi = list(self.prepdata.roi_metadata.index)
        elif isinstance(roi, str):
            roi = [roi]
        elif isinstance(roi, list):
            pass
        else:
            raise ValueError("Invalid roi")
        datasource = self.datasource

        for r in roi:
            headers = None
            if draw_spectrum_idx is None:
                rmeta = datasource.roi_metadata.loc[r]
                idxs = Indexator(rmeta["idxroi"])
                spectrum_idx = list(idxs)[np.random.randint(0,idxs.count)]
            else:
                spectrum_idx = draw_spectrum_idx
            
            if os.path.exists(self.processed_spectra_path):
                with File(self.processed_spectra_path, "r") as hdf5:
                    data_int = hdf5[r]["int"][datasource._get_local_roi_idx(spectrum_idx), :]
                    mz = hdf5[r]["mz"][:]
                    roi_draw_mz_range = (mz.min(), mz.max())
            else:
                if self.prepdata is None:
                    raise ValueError("No processed spectra path or configs to get processed spectrum")
                else:
                    pipeline = Pipeline(self.prepdata)
                    stream = pipeline._multistream_pipeline(Pipeline._alldata_wrapper,r, cpu_num=1, idxs = spectrum_idx)
                    mz, headers = next(stream)
                    mz_dict, intensity_dict, _ = next(stream)
                    mz = mz_dict[spectrum_idx]
                    data_int = intensity_dict[spectrum_idx].squeeze()
                    mz_crop_range  = self.prepdata[r]["modify_raw_spectrum"]["mz_crop_range"]
                    if mz_crop_range is not None and draw_mz_range is None:
                        low_mz, high_mz = mz_crop_range
                        if low_mz is None:
                            low_mz = mz.min()
                        if high_mz is None:
                            high_mz = mz.max()
                        roi_draw_mz_range = (low_mz, high_mz)
                    else:
                        roi_draw_mz_range = draw_mz_range
            if os.path.exists(self.peaklists_path):
                with File(self.peaklists_path, "r") as hdf5:
                    headers = hdf5[r].attrs["Column headers"]
                    peaklist = pd.DataFrame(hdf5[r][:], columns = headers).astype({"spectra_ind": int}).query('spectra_ind == @spectrum_idx')
            else:
                peaklist = None

            self._draw(mz, data_int, peaklist, headers, r, roi_draw_mz_range, spectrum_idx, axes)
            plt.show()
    def draw_peak_density(self,
                           roi: str | list | None = None,
                           draw_mz_borders: tuple[float, float] | None = None):
        """
        Draw the KDE peak density around a randomly chosen peak.

        Loads the KDE m/z and density from the peaks-density HDF5 file,
        applies the m/z correction to the peaklist within the borders around
        a random peak, and plots the probability density function together
        with the corrected peaks and the mean spectrum.

        Parameters
        ----------
        roi : str | list | None, optional
            ROI name or list of ROI names. Default ``None`` (all ROIs).
        draw_mz_borders : tuple[float, float] | None, optional
            m/z window (in m/z units) around the random peak to display.
        """
        if roi is None:
            roi = self.prepdata.rois
        if isinstance(roi, str):
            roi = [roi]
        sample_name = self.datasource.sample_name
        formatter = AbsoluteFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        for r in roi:
            with File(self.peaks_density_path, "r") as hdf5:
                kde_mz = hdf5[r]["mz"][:]
                kde_density = hdf5[r]["peaks_density"][:]
                if kde_density.size == 0:
                    continue
                kde_density = normalize( kde_density.reshape(1, -1), norm='l1' ).squeeze()
            peaklists = self.datasource.peaklists(r)
            rand_num = np.random.randint(0,peaklists.shape[0])
            peak_mz = peaklists.at[rand_num, "mz"]

            rand_spec = peaklists.at[rand_num, "spectra_ind"]
            mz_borders = (peak_mz-draw_mz_borders,peak_mz+draw_mz_borders)

            dots_bord = (np.array(kde_mz)>=mz_borders[0]) & (np.array(kde_mz)<=mz_borders[1])
            local_kde_mz = np.array(kde_mz)[dots_bord]
            local_kde_density = np.array(kde_density)[dots_bord]
            plt.figure(figsize=(25, 6), dpi=600)
            kde_line, = plt.plot(local_kde_mz, local_kde_density*(-1), color="k",alpha=0.85)
            graphs = [kde_line]
            leg = ["Probability density function"]
            plt.xlim(mz_borders)
            
            quered_peaklists = peaklists.query("mz>=@mz_borders[0] and mz<=@mz_borders[1]")
            quered_peaklists['uncor_mz'] = quered_peaklists['mz'].copy()

            corrected_peaklists = apply_kde_mzcorrection(quered_peaklists, kde_mz, kde_density)
            
            Peaks_list=corrected_peaklists["mz"].sort_values().unique()
            for peak in Peaks_list:
                temp_query = corrected_peaklists.query("mz == @peak")
                color = plt.gca()._get_lines.get_next_color()
                peak_dot, = plt.plot([peak],[0],'|', markersize=12,alpha=1, color = color, mew =3)
                graphs+=[peak_dot]
                leg+=[f"Peak {peak:.3f} m/z: On adjust/Original"]
                plt.plot(temp_query['uncor_mz'], [0]*temp_query.shape[0],'|', markersize=6,alpha=0.33, color = color)
            plt.xlabel('m/z')
            plt.ylabel("Probability Density")
            plt.gca().set_title(f"Probability density by KDE around {peak_mz:.3f} m/z. Sample: {sample_name}. roi: {r}.")
            plt.minorticks_on()
            plt.grid(visible=True,which="both")
            axes_prob = plt.gca()
            axes_prob.yaxis.set_major_formatter(formatter)
            ylim_min, ylim_max = axes_prob.get_ylim() 
            axes_prob.set_ylim((-max((abs(ylim_min), ylim_max)), max((abs(ylim_min), ylim_max))) )
            axes = plt.gca().twinx()
            legend1 = axes.legend(graphs, leg, loc = 'upper left',framealpha=0.95)
            axes.plot(*self.datasource.get_mean_spectrum(roi = r, mz_range = mz_borders), color="r",alpha=0.85)
            leg_twin=[f'Mean spectrum. Sample: {sample_name}, ROI: {r}']
            axes.set_ylabel("Intensity")
            axes.add_artist(legend1)
            axes.legend(leg_twin, loc = 'upper right')
            self.audit_processing(r,mz_borders, rand_spec, axes = axes, configs_path = self.datasource.configs_path)
            ylim_min, ylim_max = axes.get_ylim() 
            limit = max(abs(ylim_min), abs(ylim_max))
            axes.set_ylim( (-limit, limit) )
            
            plt.show()

    @staticmethod
    def draw_mzcorrection(feature_matrix: pd.DataFrame,
                            kde_mz: np.ndarray,
                            kde_density: np.ndarray,
                            countf_rel: float | None = None,
                            countf: int | None = None,
                            filter_mz_mask: np.ndarray[bool] = None,
                            draw_mz_borders = None,
                            flipped_kde_density: bool = False,
                            axes: plt.Axes | None = None) -> plt.Axes:
        
        formatter = AbsoluteFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        
        if draw_mz_borders is None:
            draw_mz_borders = (kde_mz.min(), kde_mz.max())
            local_kde_density = kde_density
            local_kde_mz = kde_mz
        else:
            kde_borders_mask = (kde_mz >= draw_mz_borders[0]) & (kde_mz <= draw_mz_borders[1])
            local_kde_mz = kde_mz[kde_borders_mask]
            local_kde_density = kde_density[kde_borders_mask]
        # Create the figure.
        if axes is None:    
            plt.figure(figsize=(25, 6), dpi=600)
            plt.xlim(draw_mz_borders)
        else:
            plt.sca(axes)

        # Draw the KDE curve.
        kde_line, = plt.plot(local_kde_mz, local_kde_density*(-1 if flipped_kde_density else 1), color="k",alpha=0.85)
        graphs = [kde_line]
        leg = ["Probability density function"]
        
        if filter_mz_mask is None:
            if countf_rel or countf:
                filter_mz_mask = _peaks_filtration(feature_matrix['mz'], countf_rel, countf)
            else:
                filter_mz_mask = np.ones(len(feature_matrix['mz']), dtype=bool)
        
        # Identify the columns holding the uncorrected peaks: those that
        # contain 'mz' but are not the corrected 'mz' column itself.
        probably_uncor_cols = [col for col in feature_matrix.columns if 'mz' in col and col != 'mz']
        if len(probably_uncor_cols) > 0:
            # Use the first such column as a simple safeguard.
            uncor_col = probably_uncor_cols[0]
        # feature_matrix.loc[filter_mz_mask, ["mz",uncor_col]]
        
        # Markers for the peak positions on the m/z axis.
        uncor_legend, = plt.plot([], [], '|', markersize=6, alpha=0.55, color = 'grey')  # Empty plot used only for the legend.
        graphs.append(uncor_legend)
        leg += [uncor_col] 

        Peaks_list = feature_matrix.loc[
        filter_mz_mask & feature_matrix["mz"].between(draw_mz_borders[0], draw_mz_borders[1]), 
        "mz"].sort_values().unique()
        for peak in Peaks_list:
            # assert peak in excluded_peaks_list 
            temp_query = feature_matrix.query("mz == @peak")
            color = plt.gca()._get_lines.get_next_color()
            peak_dot, = plt.plot([peak],[0],'|', markersize=12,alpha=1, color = color, mew =3)
            graphs+=[peak_dot]
            leg+=[f"{peak:.3f} m/z"]
            plt.plot(temp_query[uncor_col], [0]*temp_query.shape[0],'|', markersize=6,alpha=0.33, color = color)

        Excluded_peaks=feature_matrix.loc[~filter_mz_mask, ["mz",uncor_col]]
        Excluded_peaks_list = Excluded_peaks['mz'].sort_values().unique()
        excl_dots, = plt.plot(Excluded_peaks_list, [0]*len(Excluded_peaks_list),'x', markersize=12,alpha=1, color = 'k', mew=3)
        graphs+=[excl_dots]
        leg+=[f"Excluded peaks"]
        plt.plot(Excluded_peaks[uncor_col], [0]*Excluded_peaks.shape[0],'x', markersize=6,alpha=0.5, color = 'k')
        
        plt.xlabel('m/z')
        plt.ylabel("Probability Density")
        plt.gca().set_title(f"Dataset m/z correction results in range {draw_mz_borders[0]:.3f} - {draw_mz_borders[1]:.3f} m/z.")
        plt.legend(graphs, leg, loc = 'upper left')
        plt.minorticks_on()
        plt.grid(visible=True,which="both")
        axes_prob = plt.gca()
        axes_prob.yaxis.set_major_formatter(formatter)
        if flipped_kde_density:
            ylim_min, ylim_max = axes_prob.get_ylim()
            limit = max(abs(ylim_min), abs(ylim_max))
            axes_prob.set_ylim((-limit, limit))
        axes_prob.legend(graphs, leg, loc = 'upper left')
        return axes_prob

    @staticmethod
    def _draw_datasets_mzcorrection(dataset: DataSet,
                                    kde_mz: np.ndarray,
                                    kde_density: np.ndarray,
                                    peak_mz: float,
                                    samples: list[str] | None = None,
                                    rois: str | list[str] | None = None, 
                                    sample_rois_map: dict[str, list[str]] | None = None,
                                    draw_mz_borders: float = 2.5,
                                    countf_rel: float | None = None,
                                    countf: int | None = None,
                                    duplicates_drop: bool = True,
                                    cpu_num: int = 1):
        """Build a verification plot of the m/z correction for one random source.

        Uses a randomly chosen source and ROI from the dataset to visualise
        the m/z correction applied during feature-matrix construction: the
        KDE probability density, the corrected and original peak positions,
        the excluded peaks, and the raw/processed mean spectrum.
        """
        formatter = AbsoluteFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        mz_borders = (peak_mz-draw_mz_borders,peak_mz+draw_mz_borders)
        # datasources = dataset.sources
        multiindex_keys = []
        feature_matrix = []
        
        sample_rois_map = dataset._resolve_sample_rois_map(samples, rois, sample_rois_map)
        for sample, rois in sample_rois_map.items():
            datasource = dataset.sources[sample]
            if os.path.exists(datasource.peaklists_path):
                for roi in rois:
                    multiindex_keys.append((sample, roi))
                    uncor_peaklist = datasource.peaklists(roi)
                    uncor_quered_peaklist = uncor_peaklist.query("mz>=@mz_borders[0] and mz<=@mz_borders[1]")
                    uncor_quered_peaklist['uncor_mz'] = uncor_quered_peaklist['mz'].copy()

                    feature_matrix.append(uncor_quered_peaklist)
        feature_matrix = pd.concat(feature_matrix, keys=multiindex_keys, names = ['sample', 'roi'])
        feature_matrix = apply_kde_mzcorrection(feature_matrix, kde_mz, kde_density, cpu_num)
        if duplicates_drop:
            feature_matrix = _consesusing_peaks(feature_matrix)
        
        if countf or countf_rel:
            if countf is None and countf_rel is not None:
                countf = countf_rel*uncor_peaklist.shape[0]
            print(f"[DEBUG] BEFORE merge: feature_matrix.index.names={feature_matrix.index.names!r}, "
                  f"len(index)={len(feature_matrix.index)}")
            feature_matrix = feature_matrix.merge(feature_matrix['mz'].value_counts().to_frame(name='count'),left_on="mz",right_index=True)
            print(f"[DEBUG] AFTER merge: feature_matrix.index.names={feature_matrix.index.names!r}, "
                  f"index={feature_matrix.index[:5].tolist()}")

            excluded_peaks = feature_matrix.loc[feature_matrix['count'] < countf]
            feature_matrix = feature_matrix.loc[feature_matrix['count'] >= countf]
            # corrected_peaklist = _peaks_filtration(corrected_peaklist, countf_loc, countf_rel_loc)
            feature_matrix.drop(columns=['count'], inplace=True)
            excluded_peaks.drop(columns=['count'], inplace=True)
            
        else:
            excluded_peaks = pd.DataFrame(columns=feature_matrix.columns)
        
        dots_bord = (np.array(kde_mz)>=mz_borders[0]) & (np.array(kde_mz)<=mz_borders[1])
        local_kde_mz = np.array(kde_mz)[dots_bord]
        local_kde_density = np.array(kde_density)[dots_bord]

        plt.figure(figsize=(25, 6), dpi=600)
        kde_line, = plt.plot(local_kde_mz, local_kde_density*(-1), color="k",alpha=0.85)
        graphs = [kde_line]
        leg = ["Probability density function"]
        plt.xlim(mz_borders)        
        
        Peaks_list=feature_matrix["mz"].sort_values().unique()
        excluded_peaks_list=excluded_peaks["mz"].sort_values().unique()
        for peak in Peaks_list:
            # assert peak in excluded_peaks_list 
            temp_query = feature_matrix.query("mz == @peak")
            color = plt.gca()._get_lines.get_next_color()
            peak_dot, = plt.plot([peak],[0],'|', markersize=12,alpha=1, color = color, mew =3)
            graphs+=[peak_dot]
            leg+=[f"Peak {peak:.3f} m/z: Adjusted / Original"]
            plt.plot(temp_query['uncor_mz'], [0]*temp_query.shape[0],'|', markersize=6,alpha=0.33, color = color)
        excl_dots, = plt.plot(excluded_peaks_list, [0]*len(excluded_peaks_list),'x', markersize=12,alpha=1, color = 'k', mew=3)
        graphs+=[excl_dots]
        leg+=[f"Excluded peaks"]
        plt.plot(excluded_peaks['uncor_mz'], [0]*excluded_peaks.shape[0],'x', markersize=6,alpha=0.5, color = 'k')
        plt.xlabel('m/z')
        plt.ylabel("Probability Density")
        plt.gca().set_title(f"Dataset m/z correction results around {peak_mz:.3f} m/z.")
        plt.legend(graphs, leg, loc = 'upper left')
        plt.minorticks_on()
        plt.grid(visible=True,which="both")
        axes_prob = plt.gca()
        axes_prob.yaxis.set_major_formatter(formatter)
        ylim_min, ylim_max = axes_prob.get_ylim()
        limit = max(abs(ylim_min), abs(ylim_max))
        axes_prob.set_ylim((-limit, limit))
        axes_prob.legend(graphs, leg, loc = 'upper left')
        # leg_twin=[f'Mean spectrum']
        axes = plt.gca().twinx()
        legend1 = axes.legend(graphs, leg, loc = 'upper left',framealpha=0.95)
        # axes.plot(*exampled_datasource.get_mean_spectrum(roi,mz_range = mz_borders), color="r",alpha=0.85)

        axes.set_ylabel("Intensity")

        axes.add_artist(legend1)
        # axes.legend(leg_twin, loc = 'upper right')
        datasources = dataset.sources
        rand_ds_name = np.random.choice(list(sample_rois_map.keys()))
        rand_ds = datasources[rand_ds_name]
        sample_rois = sample_rois_map[rand_ds_name]
        rand_roi_int = np.random.randint(0, len(sample_rois))
        rand_roi = sample_rois[rand_roi_int]
        print(f"[DEBUG] At .loc: feature_matrix.index.names={feature_matrix.index.names!r}")
        print(f"[DEBUG] At .loc: rand_ds_name={rand_ds_name!r} (type {type(rand_ds_name).__name__}), "
              f"rand_roi={rand_roi!r} (type {type(rand_roi).__name__})")
        if not isinstance(feature_matrix.index, pd.MultiIndex):
            print(f"[DEBUG] At .loc: INDEX IS NOT A MULTIINDEX -> index sample values: "
                  f"{feature_matrix.index[:5].tolist()}")
        else:
            print(f"[DEBUG] At .loc: index roi level (level 1) unique sample = "
                  f"{feature_matrix.index.get_level_values(1).unique()[:10].tolist()}, "
                  f"has_rand_roi={rand_roi in feature_matrix.index.get_level_values(1)}")
        rand_spec = feature_matrix.loc[(rand_ds_name, rand_roi)].query("mz == @peak_mz")['spectra_ind']
        print(f"[DEBUG] feature_matrix: {feature_matrix}")
        print(f"[DEBUG] feature_matrix.shape: {feature_matrix.shape}")
        print(f"[DEBUG] peak_mz: {peak_mz}")
        if rand_spec.empty:
            print(f"[DEBUG] At .loc: rand_spec is empty")
            rand_spec = feature_matrix.loc[(rand_ds_name, rand_roi)].query("mz >= (@peak_mz - 1e-6) and mz <= (@peak_mz + 1e-6)")['spectra_ind']
            print(f"[DEBUG] At .loc: rand_spec is empty after query with tol {rand_spec}")
        while rand_spec.empty:
            del sample_rois[rand_roi_int]
            rand_roi_int = np.random.randint(0, len(sample_rois))
            rand_roi = sample_rois[rand_roi_int]
            rand_spec = feature_matrix.loc[(rand_ds_name, rand_roi)].query("mz == @peak_mz")['spectra_ind']
                 
        rand_spec = np.random.choice(rand_spec)

        axes.plot(*rand_ds.get_mean_spectrum(rand_roi,mz_range = mz_borders), color="r",alpha=0.85)
        leg_twin=[f'Mean spectrum. Sample: {rand_ds_name}, ROI: {rand_roi}']
        legend2 = axes.legend(leg_twin, loc = 'upper right')
        Drawer(rand_ds).audit_processing(rand_roi,mz_borders, rand_spec, axes = axes, configs_path = rand_ds.configs_path)
        ylim_min, ylim_max = axes.get_ylim() 
        limit = max(abs(ylim_min), abs(ylim_max))
        axes.set_ylim( (-limit, limit) )
        legend2 = axes.get_legend()
        legend2.get_texts()[1].set_text(f'Raw mass spectrum. Sample: {rand_ds_name}, ROI: {rand_roi}, N{rand_spec}')
        legend2.get_texts()[2].set_text(f'Processed mass spectrum. Sample: {rand_ds_name}, ROI: {rand_roi}, N{rand_spec}' )
        plt.show()

class AbsoluteFormatter(ticker.ScalarFormatter):
    """
    Matplotlib formatter that displays the absolute value of the tick.

    Subclasses :class:`matplotlib.ticker.ScalarFormatter` and renders the
    magnitude of each tick value, which is useful for symmetric density plots
    drawn around zero (e.g. mirrored probability densities).
    """
    def _set_format(self):
        """
        Set the tick format using the parent's logic.

        Called internally by matplotlib when configuring the formatter.
        """
        # This method is called internally to set up the format.
        super()._set_format()
        
    def __call__(self, x, pos=None):
        """
        Format the tick at position *pos* using ``abs(x)``.

        Parameters
        ----------
        x : float
            Tick value.
        pos : int | None, optional
            Tick position. Default ``None``.

        Returns
        -------
        str
            Formatted string for the absolute value of *x*.
        """
        # Main method: take abs(x) and pass it to the standard parent.
        return super().__call__(abs(x), pos)

class Pipeline:
    """
    Unified orchestrator for running MSI data processing.

    Wraps a :class:`PreparedDataSource` and runs the processing, peak
    picking, and KDE density estimation steps using the pipeline functions
    stored in the per-ROI configs. Also builds feature matrices across a
    whole :class:`DataSet`.
    """
    def __init__(self,
                 prepdata: "PreparedDataSource"):
        """
        Initialize a Pipeline for a prepared data source.

        Pipeline is a thin orchestrator that accepts a
        :class:`PreparedDataSource` and runs the processing, peak-picking,
        and KDE density-estimation steps using the per-ROI pipeline functions
        stored in the configuration object.

        Parameters
        ----------
        prepdata : PreparedDataSource
            The prepared data source with its datasource and pipeline
            functions.

        Raises
        ------
        ValueError
            If *prepdata* is not a :class:`PreparedDataSource`.
        """
        if isinstance(prepdata, PreparedDataSource):
            self.prepdata = prepdata
            self.roi_configs = prepdata.roi_configs
            self._configs_source_path = prepdata._configs_source_path
            self._datasource = prepdata._datasource
        else:
            raise ValueError(
                "Provide a PreparedDataSource."
            )

    def process(self,
                free_cpus: int = 1, 
                draw: bool = False, 
                draw_mz_range: tuple[float, float] | None = None,
                draw_spectrum_idx: int | None = None,
                Ram_GB_limit: float = 2,
                h5chunk_size_MB: int = 10,
                dtypeconv: np.dtype | str | None = None):
        """
        Process the spectra of every ROI and write them to an HDF5 file.

        Streams the raw spectra through the processing pipeline and writes
        the processed intensities into ``<sample>_processed_spectra.hdf5``,
        one dataset per ROI. Continuous data are written as a common m/z
        scale plus an intensity matrix; discontinuous data require resampling
        before they can be written and are skipped with a warning. Finally,
        the per-ROI configs are saved to disk.

        Parameters
        ----------
        free_cpus : int, optional
            Number of CPUs to leave free (default 1).
        draw : bool, optional
            Whether to draw the processed spectra (default False).
        draw_mz_range : tuple[float, float] | None, optional
            m/z range used for drawing (default None).
        draw_spectrum_idx : int | None, optional
            Spectrum index used for drawing (default None).
        Ram_GB_limit : float, optional
            RAM limit in GB for batch processing (default 2).
        h5chunk_size_MB : int, optional
            HDF5 chunk size in MB (default 10).
        dtypeconv : np.dtype | str | None, optional
            Data type conversion (default None).
        """
        datasource = self._datasource
        
        hdf5_save_path = self._default_save_path("processed_spectra.hdf5")
        if os.path.exists(hdf5_save_path):
            os.remove(hdf5_save_path)

        cpu_num = cpu_count()-free_cpus
        roi_metadata = datasource.roi_metadata

        if dtypeconv is None:
            dtypeconv = datasource.metadata.iloc[0]["dtype_raw"]
        dtypeconv = np.dtype(dtypeconv)
        bytes_flsize = dtypeconv.itemsize
        chunk_size_by_elements = int(max(1,np.ceil(h5chunk_size_MB*(1024**2)/bytes_flsize)))
        for roi in roi_metadata.index:
            processing_stream = self._multistream_pipeline(self._procfunc_wrapper,
                                                           roi = roi,
                                                           cpu_num = cpu_num,
                                                           Ram_GB_limit = Ram_GB_limit,
                                                           dtypeconv = dtypeconv)
            gen_mz, headers_list = next(processing_stream)
            if gen_mz is None:
                processing_stream.close()
                warnings.warn(f"Discontinuous data detected for sample '{datasource.sample_name}' (ROI: {roi}). "  
                              "Resampling is required before writing to HDF5.")
                
                continue
            
            with File(hdf5_save_path,"a") as hdf5:
                dots_num = len(gen_mz)
                hdf5.create_dataset(roi+"/int", (Indexator(roi_metadata.loc[roi,"idxroi"]).count, dots_num), chunks=(int(chunk_size_by_elements/dots_num), dots_num), dtype = dtypeconv)
                hdf5.create_dataset(roi+"/mz", data = gen_mz, dtype = dtypeconv)
                
                for loc_idxs, proc_intensity in processing_stream:
                
                    hdf5[roi]["int"][next(loc_idxs.__iter__()),:] = proc_intensity
                    # hdf5[roi]["int"][loc_idxs,:] = proc_intensity

            if draw:
                Drawer(self.prepdata).draw_processed_data(roi, draw_mz_range, draw_spectrum_idx)

        self.prepdata.save()
        

    def peakpick(self,
                 free_cpus: int = 1,
                 draw: bool = False,
                 draw_mz_range: tuple[float, float] | None = None,
                 draw_spectrum_idx: int | None = None,
                 Ram_GB_limit: float = 2,
                 h5chunk_size_MB: int = 10,
                 dtypeconv: np.dtype | str | None = None):
        """
        Run peak picking for the data source.

        Streams the raw spectra through the processing and peak-picking
        pipeline and writes the resulting peaklists (plus the column headers)
        into a per-ROI dataset of ``<sample>_peaklists.hdf5``.

        Parameters
        ----------
        free_cpus : int, optional
            Number of CPUs to leave free (default 1).
        draw : bool, optional
            Whether to draw the processed spectra after peak picking
            (default False).
        draw_mz_range : tuple[float, float] | None, optional
            m/z range for drawing (default None).
        draw_spectrum_idx : int | None, optional
            Spectrum index for drawing (default None).
        Ram_GB_limit : float, optional
            RAM limit in GB for batch processing (default 2).
        h5chunk_size_MB : int, optional
            HDF5 chunk size in MB (default 10).
        dtypeconv : numpy.dtype | str | None, optional
            Data type conversion for processing (default None).
        """
             
        datasource = self._datasource
        hdf5_save_path = self._default_save_path("peaklists.hdf5")
        if os.path.exists(hdf5_save_path):
            os.remove(hdf5_save_path)
        os.makedirs(os.path.dirname(hdf5_save_path), exist_ok=True)
        peaks_density_path = self._default_save_path("peaks_density.hdf5")
        if os.path.exists(peaks_density_path):
            os.remove(peaks_density_path)


        if dtypeconv is None:
            dtypeconv = datasource.metadata.iloc[0]["dtype_raw"]
        dtypeconv = np.dtype(dtypeconv)
        cpu_num = cpu_count()-free_cpus
        bytes_flsize = dtypeconv.itemsize
        chunk_size_by_elements = int(max(1,np.ceil(h5chunk_size_MB*(1024**2)/bytes_flsize)))
        
        roi_metadata = datasource.roi_metadata
        for roi in roi_metadata.index:
            peakpicking_stream = self._multistream_pipeline(self._peakpick_wrapper,
                                                           roi = roi,
                                                           cpu_num = cpu_num,
                                                           Ram_GB_limit = Ram_GB_limit,
                                                           dtypeconv = dtypeconv)
            _, headers_list = next(peakpicking_stream)
            
            with File(hdf5_save_path,"a") as hdf5:
                n_heads = len(headers_list)
                hdf5.create_dataset(roi,(0, n_heads), maxshape = (None, n_heads), chunks=(chunk_size_by_elements/n_heads, n_heads), dtype=dtypeconv)
                hdf5[roi].attrs["Column headers"] = headers_list
                for peaklists in peakpicking_stream:
                    list_size = len(peaklists)
                    if list_size == 0:
                        continue
                    hdf5[roi].resize((hdf5[roi].shape[0] + list_size, n_heads))
                    hdf5[roi][-list_size:,:] = peaklists
            if draw:
                Drawer(self.prepdata).draw_processed_data(roi, draw_mz_range, draw_spectrum_idx)

        self.prepdata.save()

    def estimate_peak_density_kde(self,
                                  free_cpus: int = 1,
                                  draw: bool = True,
                                  draw_borders: float = 1.5):
        """
        Estimate the peak density using KDE.

        Computes the kernel density estimate over the peaklists of each ROI
        and writes the resulting m/z and density grids into
        ``<sample>_peaks_density.hdf5``. Optionally draws the density around
        a random peak.

        Parameters
        ----------
        free_cpus : int, optional
            Number of CPUs to leave free (default 1).
        draw : bool, optional
            Whether to draw the resulting peak densities (default True).
        draw_borders : float, optional
            m/z window (in m/z units) around the random peak used for
            drawing (default 1.5).

        Raises
        ------
        FileNotFoundError
            If the peaklists file does not exist; run :meth:`peakpick` first.
        """
        prepdata = self.prepdata
        source = prepdata._datasource
        save_path = prepdata._default_save_path('peaks_density.hdf5')
        
        if os.path.exists(save_path):
            os.remove(save_path)
        
        peaklists_path = prepdata._default_save_path('peaklists.hdf5')
        
        if not os.path.exists(peaklists_path):
            raise FileNotFoundError(f"Peaklists file {peaklists_path} does not exist. Please run peakpicking first")
        roi_metadata = source.roi_metadata
        cpu_num = cpu_count() - free_cpus
        roi_kde_configs = prepdata.roi_kde_configs
        with File(save_path, 'w') as f:
            for roi in prepdata.rois:
                peaklist = source.peaklists(roi)
                roi_peak_density = _compute_KDE(peaklist, roi_metadata.loc[roi,"discret_coeffs"], cpu_num, **roi_kde_configs[roi].to_dict)
                f.create_dataset(f"{roi}/mz", data=roi_peak_density[0])
                f.create_dataset(f"{roi}/peaks_density", data=roi_peak_density[1])
        prepdata.dump_kde_configs()
        for roi in prepdata.rois:
            if draw:
                Drawer(self.prepdata).draw_peak_density(roi, draw_borders)
    @staticmethod
    def feature_matrix(dataset: DataSet,
                       samples: list[str] | None = None,
                        rois: str | list[str] | None = None,
                        sample_rois_map: dict[str, list[str]] | None = None,
                        countf: int = 10,
                        countf_rel: float | None = None,
                        duplicates_drop: bool = True,
                        pivot_values: str | list[str] | None = None,  # 'Intensity', ['Intensity', 'Area']
                        fill_values = 0.0,
                        free_cpus: int = 1,
                        save_path: str = None,
                        draw_borders: float = 2.5,
                        draw: bool = True,
                        local_roi_idx: bool = True,
                        merge_with_coords: bool = False) -> pd.DataFrame:
        """
        Build the feature matrix across all sources of a DataSet.

        Aggregates the peaklists of the requested samples/ROIs, applies the
        combined KDE m/z correction, optionally filters by occurrence count,
        merges duplicates into consensus peaks, and optionally pivots and
        saves the result (optionally together with coordinates).

        Parameters
        ----------
        dataset : DataSet
            The dataset whose sources are aggregated.
        rois : dict[str, list[str]], optional
            Mapping of ``{sample_name: [roi_name, ...]}``. Samples missing
            from the dict use all of their ROIs. Default ``{}``.
        countf : int, optional
            Minimum absolute occurrence count for a peak to be kept
            (default 10).
        countf_rel : float | None, optional
            Minimum relative occurrence count (fraction of spectra) for a
            peak to be kept. Default ``None``.
        duplicates_drop : bool, optional
            Whether to merge duplicate peaks into consensus peaks
            (default True).
        pivot_values : str | list[str] | None, optional
            Column(s) to pivot on, e.g. ``'Intensity'``. Default ``None``.
        fill_values : float, optional
            Value used to fill missing cells when pivoting (default 0.0).
        free_cpus : int, optional
            Number of CPUs to leave free (default 1).
        save_path : str | None, optional
            Path to save the feature matrix as Parquet (default None).
        draw_borders : float, optional
            m/z window used when drawing the correction results (default 2.5).
        draw : bool, optional
            Whether to draw the m/z correction verification plot
            (default True).
        local_roi_idx : bool, optional
            Whether to use local (per-ROI) spectrum indices (default True).
        merge_with_coords : bool, optional
            Whether to merge the coordinates into the feature matrix
            (default False).

        Returns
        -------
        pandas.DataFrame
            The aggregated (and optionally pivoted) feature matrix.

        Raises
        ------
        FileNotFoundError
            If peaklists or peaks-density files are missing; run
            :meth:`peakpick` and :meth:`estimate_peak_density_kde` first.
        """
        # Get the combined probability density of the peaks.
        kde_mz_list = []
        kde_density_list = []
        cpu_num = cpu_count() - free_cpus
        datasources = dataset.sources
        sample_rois_map = dataset._resolve_sample_rois_map(samples, rois, sample_rois_map)

        for sample, rois in sample_rois_map.items():
            datasource = datasources[sample]
            if os.path.exists(datasource.peaks_density_path):

                with File(datasource.peaks_density_path, 'r') as f:
                    for roi in rois: 
                        roi_kde_mz = f[roi]['mz'][:]
                        if roi_kde_mz.size == 0:
                            continue
                        kde_mz_list.append(roi_kde_mz)
                        kde_density_list.append( normalize( f[roi]['peaks_density'][:].reshape(1, -1), norm='l1').squeeze() )
            else:
                raise FileNotFoundError(f"Peaks density file {datasource.peaks_density_path} does not exist.\nPlease run `estimate_peak_density_kde` method first")

        kde_mz, kde_density = _summerize_kde_mz(kde_mz_list, kde_density_list)

        multiindex_keys = []
        feature_matrix = []
        for sample, rois in sample_rois_map.items():
            datasource = datasources[sample]
            
            if os.path.exists(datasource.peaklists_path):
                for roi in rois:
                    multiindex_keys.append((sample, roi))
                    peaklists = datasource.peaklists(roi).set_index('spectra_ind',drop = True)
                    if not local_roi_idx:
                        idx_map = Indexator(datasource.roi_metadata.loc[roi, 'idxroi'])
                        idx_map = dict(zip(range(idx_map.count), idx_map))
                        peaklists.index = peaklists.index.map(idx_map)
                    feature_matrix.append(peaklists)
            else:
                raise FileNotFoundError(f"Peaklists file {datasource.peaklists_path} does not exist.\nPlease run `peakpick` method first")
        nunique_stats = []
        feature_matrix = pd.concat(feature_matrix, keys=multiindex_keys, names = ['sample', 'roi'])
        if draw:
            feature_matrix['uncorrected_mz'] = feature_matrix['mz'].copy()
        nunique_stats.append(_nunique_summary(feature_matrix['mz'].droplevel('spectra_ind'),'before correction'))
        feature_matrix = apply_kde_mzcorrection(feature_matrix, kde_mz, kde_density, cpu_num)
        nunique_stats.append(_nunique_summary(feature_matrix['mz'].droplevel('spectra_ind'),'after correction'))

        # Duplicates manipulations
        # Counting duplicates
        dupl_stats = _consensus_peaks_summary(feature_matrix['mz'])
        # Merging duplicates and create consensus peaks

        # Peaks filtration
        if countf or countf_rel:
            filter_bool = _frequency_filtration(feature_matrix['mz'], countf, countf_rel)
            nunique_stats.append(_nunique_summary(feature_matrix['mz'].loc[filter_bool].droplevel('spectra_ind'),'after filtration'))
            # Duplicates recounting with merging
            dupl_stats_filtration = _consensus_peaks_summary(feature_matrix['mz'])
            all_columns = dupl_stats.columns.union(dupl_stats_filtration.columns)
            dupl_stats = dupl_stats.reindex(columns=all_columns, fill_value=0)
            dupl_stats_filtration = dupl_stats_filtration.reindex(columns=all_columns, fill_value=0)
            dupl_stats = pd.concat([dupl_stats, dupl_stats_filtration],axis=1, keys=["before filtration", "after filtration"])
        
        if draw:
            peaks_num = filter_bool.sum()
            if peaks_num > 0:
                # Randomizer: pick a random sample and ROI first, then a random peak that
                # is guaranteed to exist within that ROI's filtered data, so the later mz
                # query can never run empty (fixes ValueError: high <= 0 from randint(0, 0)).
                rand_ds_name_list = set(feature_matrix[filter_bool].index.get_level_values(0)) & set(sample_rois_map.keys())
                rand_ds_name = np.random.choice(list(rand_ds_name_list))
                rand_ds = datasources[rand_ds_name]

                sample_rois = sample_rois_map[rand_ds_name]
                rand_roi_int = np.random.randint(0, len(sample_rois))
                rand_roi = sample_rois[rand_roi_int]

                roi_filtered = feature_matrix[filter_bool].sort_index().loc[(rand_ds_name, rand_roi)]
                if roi_filtered.empty:
                    raise ValueError(
                        f"No filtered peaks for sample={rand_ds_name!r}, roi={rand_roi!r}; "
                        "cannot draw a random peak"
                    )
                rand_num = np.random.randint(0, roi_filtered.shape[0])
                peak_mz = roi_filtered['mz'].values[rand_num]
                draw_mz_borders = (peak_mz - draw_borders, peak_mz + draw_borders)
                axes_prob = Drawer.draw_mzcorrection(feature_matrix, kde_mz, kde_density, draw_mz_borders = draw_mz_borders, countf_rel=countf_rel, countf=countf, filter_mz_mask = filter_bool, flipped_kde_density = True)
                graphs, leg = axes_prob.get_legend_handles_labels()
                axes_spectrum = axes_prob.twinx()
                legend1 = axes_spectrum.legend(graphs, leg, loc = 'upper left',framealpha=0.95)
                # axes.plot(*exampled_datasource.get_mean_spectrum(roi,mz_range = mz_borders), color="r",alpha=0.85)

                axes_spectrum.set_ylabel("Intensity")
                axes_spectrum.add_artist(legend1)
                
                rand_spec = feature_matrix.sort_index().loc[(rand_ds_name, rand_roi)].query("mz == @peak_mz").index.get_level_values('spectra_ind').tolist()
                
                # if not rand_spec:
                    
                #     rand_spec = feature_matrix.sort_index().loc[(rand_ds_name, rand_roi)].query("mz >= (@peak_mz - 1e-6) and mz <= (@peak_mz + 1e-6)").index.get_level_values('spectra_ind').tolist()
                while not rand_spec:
                    del sample_rois[rand_roi_int]
                    rand_roi_int = np.random.randint(0, len(sample_rois))
                    rand_roi = sample_rois[rand_roi_int]
                    rand_spec = feature_matrix.loc[(rand_ds_name, rand_roi)].query("mz == @peak_mz")['spectra_ind']
                rand_spec = np.random.choice(rand_spec)

                axes_spectrum.plot(*rand_ds.get_mean_spectrum(rand_roi,mz_range = draw_mz_borders), color="r",alpha=0.85)
                leg_twin=[f'Mean spectrum. Sample: {rand_ds_name}, ROI: {rand_roi}']
                legend2 = axes_spectrum.legend(leg_twin, loc = 'upper right')
                Drawer(rand_ds).audit_processing(rand_roi, draw_mz_borders, rand_spec, axes = axes_spectrum, configs_path = rand_ds.configs_path)
                ylim_min, ylim_max = axes_spectrum.get_ylim() 
                limit = max(abs(ylim_min), abs(ylim_max))
                axes_spectrum.set_ylim( (-limit, limit) )
                legend2 = axes_spectrum.get_legend()
                legend2.get_texts()[1].set_text(f'Raw mass spectrum. Sample: {rand_ds_name}, ROI: {rand_roi}, N{rand_spec}')
                legend2.get_texts()[2].set_text(f'Processed mass spectrum. Sample: {rand_ds_name}, ROI: {rand_roi}, N{rand_spec}' )
                plt.show()
            else:
                print(f"No peaks after filtration with countf_rel={countf_rel} and countf={countf}")

        feature_matrix = feature_matrix[filter_bool]
        if duplicates_drop:
            feature_matrix = _consesusing_peaks(feature_matrix)

        nunique_stats = pd.concat(nunique_stats, axis=1)
        nunique_stats.columns =  pd.MultiIndex.from_tuples( [('Number of unique peaks', c, '') for c in nunique_stats.columns] )
        dupl_stats.columns =  pd.MultiIndex.from_tuples( [('Number of unique consensus peaks', *c) for c in dupl_stats.columns] )
        show_df(pd.concat([nunique_stats, dupl_stats], axis=1), 'Peaks statistics')
        # if draw:
        #     rand_num = np.random.randint(0,feature_matrix[filter_bool].shape[0])
        #     col_idx = feature_matrix.columns.get_loc("mz")
        #     peak_mz = feature_matrix.iat[rand_num, col_idx]
  
        if pivot_values is not None:
            feature_matrix = feature_matrix.pivot_table(index= [name for name in feature_matrix.index.names if name is not None], 
                                                        columns='mz', 
                                                        values=pivot_values, 
                                                        fill_value = fill_values)
        if save_path or merge_with_coords: 
            coords = dataset.coordinates(sample_rois_map=sample_rois_map, local_roi_idx=local_roi_idx)
        if merge_with_coords:
            nlevels = feature_matrix.columns.nlevels
            if nlevels != 1:
                base_cols = coords.columns.get_level_values(0)
                n_cols = len(base_cols)
                empty_layers = [[""]*n_cols] * (nlevels - 2)
                coords.columns = pd.MultiIndex.from_arrays([['Coordinates'] * n_cols, *empty_layers, base_cols])

            feature_matrix = feature_matrix.merge(coords, left_index=True, right_index=True)

        if save_path is not None:
            dirpath = os.path.dirname(save_path)
            os.makedirs(dirpath, exist_ok=True)
            if not save_path.endswith('.parquet'):
                save_path += '.parquet'
            feature_matrix.to_parquet(save_path)
            if not merge_with_coords:
                coords.to_parquet(os.path.join(dirpath, 'coordinates.parquet'))
                
            with open(os.path.join(dirpath, 'aggregation_configs.yaml'), 'w', encoding='utf-8') as f:
                sample_dict = {}
                for sample, roi in multiindex_keys:
                    sample_dict[sample] = sample_dict.get(sample, []) + [roi]
                yaml.dump({'samples': sample_dict,
                           'countf': countf, 
                           'countf_rel': countf_rel, 
                           'duplicates_drop': duplicates_drop, 
                           'pivoting values': pivot_values, 
                           'fill values on pivoting': fill_values, 
                           'merge_with_coords': merge_with_coords, 
                           'local_roi_idx': local_roi_idx}, f, allow_unicode=True, sort_keys=False)
        # if draw:
        #     draw_mz_borders = (peak_mz - draw_borders, peak_mz + draw_borders)
            # Drawer._draw_datasets_mzcorrection(dataset, kde_mz, kde_density, sample_rois_map = sample_rois_map, draw_mz_borders = draw_mz_borders, countf_rel=countf_rel, countf=countf, duplicates_drop=duplicates_drop, cpu_num=cpu_num)

            
        return feature_matrix
    
    def _multistream_pipeline(self,
                              process_wrapper: Callable,
                              roi: str,
                              cpu_num: int = 1,
                              Ram_GB_limit: int = 2,
                              dtypeconv:  np.dtype | str | None = None,
                              idxs: Indexator | SliceIndexator | int | None = None):
        """Generator that runs a processing step over a ROI using multiprocessing.

        First resolves the per-ROI pipeline functions from the configuration
        and yields the common m/z scale together with the peak-list column
        headers. It then splits the requested spectrum indices into batches
        that fit within the RAM budget and yields the result of
        ``process_wrapper`` for every batch, processed in parallel.

        Parameters
        ----------
        process_wrapper : Callable
            One of ``_procfunc_wrapper``, ``_peakpick_wrapper``, or
            ``_alldata_wrapper``, defining which step to run.
        roi : str
            The ROI to process.
        cpu_num : int, optional
            Number of processes to use. Default ``1``.
        Ram_GB_limit : int, optional
            RAM limit in GB for batch sizing. Default ``2``.
        dtypeconv : numpy.dtype or str or None, optional
            Data type conversion for processing. Default ``None``.
        idxs : Indexator | SliceIndexator | int | None, optional
            Spectrum indices to process. If ``None``, all the indices of the
            ROI are used. Default ``None``.

        Yields
        ------
        object
            The first yielded value is ``(mz, headers_list)``; subsequent
            values are the per-batch results returned by ``process_wrapper``.
        """
        datasource = self._datasource
        if dtypeconv is None:
            dtypeconv = datasource.metadata.iloc[0]["dtype_raw"]
        dtypeconv = np.dtype(dtypeconv)
        roi_metadata = datasource.roi_metadata
        rmeta = roi_metadata.loc[roi]
        if idxs is None:
            idxs = rmeta["idxroi"]
        # Get per-ROI PipelineConfigurator pipeline functions and its configs from PreparedDataSource
        roi_configs = self.roi_configs[roi]
        preprocess_function = roi_configs._preprocess_function
        
        mz = None
        headers_list = None
        size_per_spec = None
        if preprocess_function:
            mz, headers_list, internal_configs = preprocess_function(datasource, roi, rmeta, roi_configs)
            if mz is None and datasource.loader.dcont:
                mz = datasource.loader.mz_scale_cont
        else:
            if datasource.loader.dcont:
                mz = datasource.loader.mz_scale_cont

        if process_wrapper.__name__ == "_procfunc_wrapper":
            internal_configs['process_pipeline']= roi_configs._process_function
            wrapper_configs = roi_configs.get_step_configs("process")
        elif process_wrapper.__name__ in ["_peakpick_wrapper","_alldata_wrapper"]:
            internal_configs['process_pipeline']  = roi_configs._process_function
            internal_configs['peakpick_function']  = roi_configs._peakpick_function
            wrapper_configs = {"peakpick":roi_configs.get_step_configs("peakpick"), 
                               "process":roi_configs.get_step_configs("process")}
        else:
            raise ValueError("Unknown process_wrapper function")

        yield mz, headers_list # mz common mz to spectra, if mz is not common, None is returned. Headers list for peaklists.

        partial_worker = partial(process_wrapper,
            datasource = datasource,
            configs = wrapper_configs,
            dtypeconv = dtypeconv,
            **internal_configs
            )
        

        if isinstance(idxs, (int,np.integer)):
            yield partial_worker(np.asarray([idxs,idxs+1]))
        else:
            if mz is not None:
                size_per_spec = len(mz)
            idxs_batches = datasource.split_idxs(idxs = idxs,cpu_count=cpu_num, Ramcap_GB = Ram_GB_limit, size_per_spec = size_per_spec)
            with Pool(cpu_num) as p:
                for data in tqdm(p.imap_unordered(partial_worker, idxs_batches), total=len(idxs_batches), unit = 'batch', desc = f'Processing ROI {roi}'):
                    yield data

    def _default_save_path(self, suffix = "", prefix = ""):
        """
        Delegate to the underlying datasource's default save-path helper.

        Parameters
        ----------
        suffix : str, optional
            File suffix/name appended to the base path (default ``""``).
        prefix : str, optional
            File prefix prepended to the base path (default ``""``).

        Returns
        -------
        str
            The resolved output file path.
        """
        return self._datasource._default_save_path(suffix, prefix)
    @staticmethod
    def _procfunc_wrapper(idxs: Indexator | SliceIndexator | tuple| np.ndarray,
                          datasource: "DataSource",
                          configs: "dict | Configs | PipelineConfigurator",
                          dtypeconv: np.dtype | None = None,
                          **internal_configs
                          ):
        """
        Worker that processes a batch of spectra for :meth:`process`.

        Streams the spectra in *idxs*, applies the configured process
        function to each one, and returns the local row indices together with
        the stacked processed intensities.

        Parameters
        ----------
        idxs : Indexator | SliceIndexator | tuple | numpy.ndarray
            Global spectrum indices of the batch.
        datasource : DataSource
            The data source to read spectra from.
        configs : dict | Configs | PipelineConfigurator
            Per-step processing configs.
        dtypeconv : numpy.dtype | None, optional
            Data type conversion. Default ``None``.
        **internal_configs
            Internal per-step configs and the process pipeline function.

        Returns
        -------
        tuple
            ``(local_row_indices, processed_intensity_matrix)``.
        """
        process_function = internal_configs.pop("process_pipeline")
        internal_process_configs = internal_configs['process']

        row_idxs = SliceIndexator(datasource._get_local_roi_idx(idxs))
        processed_spectra: list[np.ndarray] = [None] * row_idxs.count
        batch_iter = datasource.get_spectra_stream(Indexator(idxs))
        for n, (_mz, raw_intensity) in enumerate(batch_iter):
            _, proc_intensity = process_function(_mz, np.asarray(raw_intensity, dtype=dtypeconv), configs, **internal_process_configs)
            processed_spectra[n] = proc_intensity
        return row_idxs, np.vstack(processed_spectra)
    
    @staticmethod
    def _peakpick_wrapper(idxs: Indexator | SliceIndexator | tuple| np.ndarray,
                          datasource: "DataSource",
                          configs: "dict | Configs | PipelineConfigurator",
                          dtypeconv: np.dtype | None = None,
                          **internal_configs
                          ):
        """
        Worker that processes and peak-picks a batch of spectra.

        Streams the spectra in *idxs*, applies the process function and then
        the peak-picking function to each one, and returns the stacked peak
        list for the batch.

        Parameters
        ----------
        idxs : Indexator | SliceIndexator | tuple | numpy.ndarray
            Global spectrum indices of the batch.
        datasource : DataSource
            The data source to read spectra from.
        configs : dict | Configs | PipelineConfigurator
            Per-step processing and peak-picking configs.
        dtypeconv : numpy.dtype | None, optional
            Data type conversion. Default ``None``.
        **internal_configs
            Internal per-step configs and the pipeline functions.

        Returns
        -------
        numpy.ndarray
            The stacked peaklist for the batch.
        """
        
        process_function = internal_configs.pop("process_pipeline")
        peakpick_function = internal_configs.pop("peakpick_function")
        proc_configs = configs['process']
        internal_proc_configs = internal_configs['process']
        peakpick_configs = configs['peakpick']
        internal_peakpick_configs = internal_configs['peakpick']

        peaklists = {}
        idxs_list = list(Indexator(idxs))
        batch_iter = datasource.get_spectra_stream(Indexator(idxs))
        for n, (_mz, raw_intensity) in enumerate(batch_iter):
            _mz, proc_intensity = process_function(_mz, raw_intensity, proc_configs, **internal_proc_configs)
            peaklists[n] = peakpick_function(_mz, np.asarray(proc_intensity, dtype=dtypeconv).squeeze(), idxs_list[n], peakpick_configs, **internal_peakpick_configs)
        return np.vstack(tuple(peaklists.values()))
    
    @staticmethod
    def _alldata_wrapper(idx: Indexator | SliceIndexator | tuple| np.ndarray | int,
                              datasource: "DataSource",
                              configs: "dict | Configs | PipelineConfigurator",
                              dtypeconv: np.dtype | None = None,
                              **internal_configs
                              ):
        """
        Worker that returns processed spectra, intensities, and peaklists.

        Processes and peak-picks a single spectrum (or a batch), returning
        dictionaries keyed by spectrum index along with the stacked peak list.
        Used by the drawing/auditing workflows.

        Parameters
        ----------
        idx : Indexator | SliceIndexator | tuple | numpy.ndarray | int
            Spectrum index or batch of indices to process.
        datasource : DataSource
            The data source to read spectra from.
        configs : dict | Configs | PipelineConfigurator
            Per-step processing and peak-picking configs.
        dtypeconv : numpy.dtype | None, optional
            Data type conversion. Default ``None``.
        **internal_configs
            Internal per-step configs and the pipeline functions.

        Returns
        -------
        tuple
            ``(proc_mz, proc_intensity, stacked_peaklist)`` where the first
            two are dicts keyed by spectrum index.
        """
        process_function = internal_configs.pop("process_pipeline")
        peakpick_function = internal_configs.pop("peakpick_function")
        
        proc_configs = configs['process']
        internal_proc_configs = internal_configs['process']
        
        peakpick_configs = configs['peakpick']
        internal_peakpick_configs = internal_configs['peakpick']

        peaklists = {}
        if isinstance(idx, int):
            idx = np.asarray([idx,idx+1], dtype=np.int64)
        idxs_list = list(Indexator(idx))
        batch_iter = datasource.get_spectra_stream(Indexator(idx))
        proc_mz = {}
        proc_intensity = {}
        for n, (_mz, raw_intensity) in enumerate(batch_iter):
            i = idxs_list[n]
            _mz, _intensity = process_function(_mz, raw_intensity, proc_configs, **internal_proc_configs)
            proc_mz[i] = _mz
            proc_intensity[i] = _intensity
            peaklists[i] = peakpick_function(_mz, np.asarray(_intensity, dtype=dtypeconv).squeeze(), idxs_list[n], peakpick_configs, **internal_peakpick_configs)
        return proc_mz, proc_intensity, np.vstack(tuple(peaklists.values()))
