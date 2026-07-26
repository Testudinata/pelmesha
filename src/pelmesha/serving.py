from pelmesha.configs import Configs, Pipeline, PreparedDataSource, PipelineConfigurator
from pelmesha.filling import DataSource
from pelmesha.dough import Indexator
import numpy as np
import os
import yaml
import warnings
from urllib.parse import quote
class DataSet:
    # TODO:
    # INITIALIZATION, CONFIGS AND PATHS:
    # DONE 1. Инициализация класса с возможностью опционального добавления словаря по типу: {path: config}, если конфиг None или просто список path, то пытается использовать записанный конфиг файл в папке
    # DONE 2. Метод add_source(path, config или {path: config}), который вносит в список путей
    # DONE 3. Метод ({path: config}), также вносит в список путей, как и инит или add_source
    # DONE 4. Добавить защиту от добавления уже какого-то пути, который есть
    # DONE 5. Добавить метод add_sources_from_paths(path_list, extensions, config), который ищет определённые расширения в пути и вносит в список. Минус:  При этом нужно исключить сразу те названия, которые уже связаны с обработкой (типа определённых hdf5)
    # DONE 6. Создать хранение конфигов в DataSource (путь и сам выгруженный конфиг класс в подпапке) и, собственно, выгрузку при инициализации пути к файлу. Если файл конфигов yaml отсутствует - создать и внести поверх базовой основы новые конфиги
    # 7. (Сделать последним)Добавить проверку об изменениях параметров обработки или их соответствию старым, если они есть. Отметка, что старых не было тоже
    # 8. (Сделать последним, а пока сделать заглушку в виде удаления)Добавить возможность автоматической сверки наличия обработанных данных и их конфигов обработки с поставляемыми, если совпадают - повторная обработка не производится.
    # New TODO (TODO выше написана нейросеткой и на ревизии, ниже результаты ревизии)
    # Done 1. Придумать способ и реализовать исключения совпадений названий образцов (с одинаковыми sample названиями).
    # 2. Добавить отдельный класс для создания референсного списка пиков????????
    # Done 3. Добавить удобную репрезентацию, чтобы отображало кратко какие датасеты находятся внутри. 
    # Done Отображение: 
    # Done 1) sample name, 
    # Done 2) кол-во спектров, 
    # Done 3) cont/discont, 
    # Done 4) Путь с гиперссылкой?(html вариант), 
    # Done 5) ?гиперссылка на конфиги локальные?(html вариант)
    # Done 6) Есть ли пиклист
    # Done 7) Есть ли обработанный вариант масс-спектров
    # Not implemented 8) Есть ли pgrouping вариант?
    # 9) (Сделать последним) Совпадает ли старый конфиг и новый?   
    # 4. Удаление путей??????

    # INDIVIDUAL PROCESSING AND REFERENCE
    # Done 9. Написать метод, который бы запустил обработку данных записанных в списке путей и на основе индивидиуальных конфигов (с вариантами: чисто пик-пикинг (по сути последний метод с изменёнными конфигами), 
    # чисто обработка спектров, пик-пикинг с обработкой и сохранение в hdf5 спектров). При этом, проводится проверка наличия атрибута рефернсного файла. Обработка "индивидуальная"
    # 10. Написать атрибут, который хранил бы путь выбранного "референсного файла" для создания референсного пиклиста, также хранил бы и свой особый конфиг для обработки 
    # (отличный от того, когда файл используется для обработки и анализа, а не референсный).
    # 11. Написать атрибут, со списком референсных пиков от референсного файла
    # 12. Если менять конфиг рефернсного атрибута, то список рефернсных пиков удаляется/None-ится.
    # upd. 13. Добавить автоматическое создание базового пайплайна с базовыми настройками при инициализации. Если хочется изменить пайплайн - использовать метод setPipeline (написать)
    #New TODO:
    # 1. Добавить вместе с отображением ds, также какие-то данные о референсном/рефернсных датасетах.
    # 2. Добавить возможность разом удалить все уже обарботанные данные датасетов, добавленных в экземпляр DataSet. Обязательно наличие yes/no подтверждения (лучший вариант, не требующих новых зависимостей) 
    # MERGING AND PGROUPING
    # 13. Мерджинг координат, метадаты и датасетов с сохранением принадлежности к определённому файлу и roi (Мультииндекс в датафрейме?). 
    # Подумать насчёт сохранения в атрибут результата. Либо каждый раз доставать заново
    # 14. Создать метод финальной коррекции m/z методом Pgrouping_KD для объединённых датасетов. Подумать над тем, как результирующие данные сохранять. По сути ведь лишь появляется новая колонка, 
    # но стоит как-то зафиксировать как она была получена (какие датасеты смерджены и какие были параметры их обработки?)
    # 15. Метод мерджинга основного датасета с координатами? По сути пользователи и сами смогут сделать, но тут фишка в упрощении действа, так как большинство моментов спратаны в классе.
    #New TODO:
    # 1. Обдумать и добавить отображение для результата Pgrouping_KD в сочетании с какими датасетами он формировался.
 
    """
    Central class for managing multiple mass spectrometry data sources.

    Combines datasets from multiple files into unified tables with metadata,
    coordinates, and processing pipeline support. Handles config management,
    duplicate detection, reference peaks, and batch processing.

    Основные возможности:
    1. Объединяет датасеты для группировки фич в одну таблицу, с сохранением всех метаданных и ссылок на них.
    2. При этом освобождает RAM от индивидуальных подгрузок.
    3. Ищет источники по списку путей, если найденный файл не имеет обработанного рядом результата - просит конфиг для обработки. Если есть обработанный, то сравнивает конфиги, если он передан в аргумент.
       И если они не совпадают производит новую обработку по новому конфигу.
    4. Должен исключать конфликты в названиях sample.
    5. По "призыву" функции по мердженгу: создаёт и мерджит в единый DF все датасеты внутри.
    6. На выходе функции по смёрдживанию датасетов (пункт 5) не только основная датасет таблица, но и метадаты с координатами.

    :param sources: Optional initial sources. Can be:
        - A list of file paths (configs loaded from disk if available)
        - A dict ``{path: config_or_None}`` where config is a  or path to a YAML file
        - ``None`` (empty DataSet, add sources later)
    :param RamGb_limit_usage: RAM limit in GB for batch processing. Default ``2``.

    :type sources: list or dict or None
    :type RamGb_limit_usage: int or float
    """
    _EXCLUDED_EXTENSIONS = {'ingredients.hdf5', 'specdata.hdf5', 'peaklists.hdf5'}

    def __init__(self, 
                 sources = None, 
                 RamGb_limit_usage = 2):
        self.sources = {}
        self.RamGb_limit_usage = RamGb_limit_usage
        self._reference_file_path = None
        self._reference_peaks = None
        self._merged_result = None

        if sources is not None:
            self.add_sources(sources)
        print(f"DataSet is initialized. Current data samples:")
        print(self)

    def add_sources(self, source, config = None, extensions = None):
        """
        Add one or more data sources to the DataSet.

        :param source: A single file path, a list of paths, or a dict ``{path: config_or_None}``.
        :param config: Config dict or path to a YAML file. Ignored if ``source`` is a dict.
            If ``None``, attempts to load a saved config from the source directory.

        :type source: str or list or dict
        :type config: dict or str or None

        :raises FileNotFoundError: If a source path does not exist.
        :raises ValueError: If a source path has already been added.
        """
        if isinstance(source, dict):
            for path, cfg in source.items():
                if os.path.isdir(path):
                    self.add_sources_from_paths([path],extensions,cfg)
                else:
                    self._add_single_source(path, cfg)
        elif isinstance(source, (list, tuple)):
            self.add_sources_from_paths(source,extensions,config) # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
            # for path in source: 
            #     if os.path.isdir(path):
            #         self.add_sources_from_paths([path],extensions,config)
            #     else:
            #         self._add_single_source(path, config)
        elif isinstance(source, str):
            if os.path.isdir(source):
                self.add_sources_from_paths([source],extensions,config)
            else:
                self._add_single_source(source, config)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def __call__(self, source, config = None, extensions = None):
        """
        Callable interface — same as :meth:`add_source`.

        Usage: ``dataset(path, config)`` or ``dataset({path: config})``.
        """
        self.add_sources(source, config, extensions)
        return self

    def _add_single_source(self, path, config = None):
        """Internal: add a single source with duplicate protection."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source path does not exist: {path}")
        sample_name = os.path.splitext(os.path.basename(path))[0]
        folder_name = os.path.basename(os.path.dirname(path))
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
        source = PreparedDataSource(source, resolved_config)
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

    def add_sources_from_paths(self, path_list, extensions = None, config = None):
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
                self._add_single_source(path, config)
            except (ValueError, FileNotFoundError) as e:
                warnings.warn(f"Skipping {path}: {e}")

    def save_configs(self): 
        """
        Save the current config for each source to its ``raw_pelmesha/config.yaml`` file.
        """
        for ds in self.sources.values():
            ds.save()

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
                **kwargs,
            )


    @property
    def reference_file_path(self): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """Path to the reference data file used for reference peak list generation."""
        return self._reference_file_path

    @reference_file_path.setter
    def reference_file_path(self, path): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """Set reference file path. Resets reference peaks if config changes."""
        if path != self._reference_file_path:
            self._reference_file_path = path
            self._reference_peaks = None

    @property
    def reference_config(self): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """Config dict for reference file processing (may differ from analysis config)."""
        return self._reference_config

    @reference_config.setter
    def reference_config(self, config): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """Set reference config. Resets cached reference peaks."""
        self._reference_config = config
        self._reference_peaks = None

    @property
    def reference_peaks(self): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """
        List of reference peaks (m/z values) from the reference file.

        Lazily computed on first access. Returns ``None`` if no reference file is set.
        """
        if self._reference_peaks is None and self._reference_file_path is not None:
            self._compute_reference_peaks()
        return self._reference_peaks

    def _compute_reference_peaks(self): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """
        Compute reference peaks from the reference file using its config.

        Processes the reference file through the peak-picking pipeline and stores
        the resulting peak m/z values.
        """
        if self._reference_file_path is None:
            return

        from pelmesha.pspectra import add_procc_data, peaks_prop_array

        ref_config = self._reference_config or {}
        ref_name = os.path.splitext(os.path.basename(self._reference_file_path))[0]

        ref_ds = DataSource(self._reference_file_path, RamGb_limit_usage = self.RamGb_limit_usage)
        ref_ds.config = ref_config

        hdf5_path = create_file_path(
            os.path.dirname(self._reference_file_path),
            slide_name = ref_name,
            hdf5_end = '_specdata'
        )

        add_procc_data(
            {hdf5_path: [[ref_name, None]]},
            func = [peaks_prop_array],
            configs_source = {ref_name: ref_config},
            dataset_name = "peaklists"
        )

        with File(hdf5_path, 'r') as f:
            peaks = []
            for sample in f.keys():
                for roi in f[sample].keys():
                    if 'peaklists' in f[sample][roi]:
                        headers = list(f[sample][roi]['peaklists'].attrs['Column headers'])
                        if 'Peak' in headers:
                            peak_idx = headers.index('Peak')
                            peaks.extend(f[sample][roi]['peaklists'][:, peak_idx])

        self._reference_peaks = np.unique(peaks) if peaks else np.array([])

    def merge(self, extr_columns = None, extract_coords = True, pivoting4val = None,
              processed_feat = False, force = False): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """
        Merge all sources into a single DataFrame with multi-index ``(slide, sample, roi)``.

        Results are cached. Use ``force=True`` to recompute.

        :param extr_columns: Columns to extract. See :func:`IMGfeats_concat`.
        :param extract_coords: If ``True``, also return coordinates DataFrame.
        :param pivoting4val: If set, pivot the result table.
        :param processed_feat: If ``True``, use feature data instead of peaklists.
        :param force: If ``True``, recompute even if cached.

        :type extr_columns: list or None
        :type extract_coords: bool
        :type pivoting4val: list or None
        :type processed_feat: bool
        :type force: bool

        :return: Merged DataFrame (and optionally coordinates DataFrame).
        :rtype: pd.DataFrame or tuple
        """
        if self._merged_result is not None and not force:
            return self._merged_result

        paths = {}
        for sample_name, ds in self.sources.items():
            hdf5_dir = os.path.dirname(ds.file_path)
            paths[hdf5_dir] = [[sample_name, None]]

        result = IMGfeats_concat(
            paths,
            extr_columns = extr_columns,
            extracts_coords = extract_coords,
            processed_feat = processed_feat
        )

        self._merged_result = result
        return result

    def correct_mz(self, ftable = None, **kwargs): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """
        Apply Pgrouping_KD m/z correction to the merged dataset.

        Groups closely-spaced m/z values across spectra using kernel density estimation.

        :param ftable: Input feature table. If ``None``, uses the merged result from :meth:`merge`.
        :param kwargs: Additional parameters passed to :func:`Pgrouping_KD`.

        :type ftable: pd.DataFrame or None

        :return: Corrected feature table with a new ``Peak`` column.
        :rtype: pd.DataFrame
        """
        from pelmesha.pfeats import Pgrouping_KD

        if ftable is None:
            merged = self.merge(extract_coords = False)
            if isinstance(merged, tuple):
                ftable = merged[0]
            else:
                ftable = merged

        result = Pgrouping_KD(ftable, **kwargs)
        return result

    def merge_with_coords(self, extr_columns = None, pivoting4val = None,
                          processed_feat = False, force = False): # TODO Проверить !!!!!!!!!!!!!!!!!!!!!!
        """
        Merge all sources and return a combined DataFrame with coordinates joined.

        The result has a multi-index ``(slide, sample, roi, spectra_ind)`` and includes
        ``x``, ``y`` (and optionally ``z``) columns alongside the feature data.

        :param extr_columns: Columns to extract from feature data.
        :param pivoting4val: If set, pivot the feature table.
        :param processed_feat: If ``True``, use feature data instead of peaklists.
        :param force: If ``True``, recompute even if cached.

        :type extr_columns: list or None
        :type pivoting4val: list or None
        :type processed_feat: bool
        :type force: bool

        :return: DataFrame with feature columns and ``x``, ``y`` coordinates.
        :rtype: pd.DataFrame
        """
        features, coords = self.merge(
            extr_columns = extr_columns,
            extract_coords = True,
            pivoting4val = pivoting4val,
            processed_feat = processed_feat,
            force = force
        )

        combined = features.join(coords, how = 'left')
        return combined

    def __getitem__(self, sample):
        return self.sources[sample]

    def __getattr__(self, sample):
        if sample.startswith('_'):
            raise AttributeError(sample)
        if sample in self.sources:
            return self.sources[sample]
        raise AttributeError(f"DataSet has no source '{sample}'")

    def __iter__(self):
        return iter(self.sources.values())

    def __len__(self):
        return len(self.sources)

    def __contains__(self, item):
        return item in self.sources

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
                    sample,                                                                                             #Sample name
                    Indexator(np.vstack(source.roi_metadata['idxroi'].to_numpy())).count,                               #Mass spectra number
                    source._datasource.dcont,                                                                           #Continuous
                    os.path.exists(os.path.join(dir_path,'processed_pelmesha', f'{sample}_peaklists.hdf5')),            #Peaklists
                    os.path.exists(os.path.join(dir_path,'processed_pelmesha', f'{sample}_processed_spectra.hdf5')),    #Processed mass spectra
                    dir_path,                                                                                           #Directory
                    config_file if os.path.exists(config_file) else "—",                                                #Previous configs
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
        - Monospace font for the numeric "кол-во спектров" column.
        - Hyperlinks for the "Путь" and "Конфиги" columns.

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
        return self.sources.keys()

    def values(self):
        return self.sources.values()

    def items(self):
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