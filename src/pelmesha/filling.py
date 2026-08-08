import itertools
import os
import re
import numpy as np
from h5py import File
import pandas as pd 
import xarray as xr
from pyteomics import mzxml
from pyimzml.ImzMLParser import ImzMLParser
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import math
from abc import ABC, abstractmethod
from itertools import  pairwise
from functools import cached_property
from pelmesha.dough import Indexator, SliceIndexator
from pelmesha.utensils import del_hdf5

class DataSource: 
    """
    WIP
    Класс основного управления обработкой данных от ОДНОГО файла. Использует класс DataManager для получения источника данных с унифицированным интерфейсом подгрузки.
    СОдержит в себе метаданные источника данных. Сохранные метаданные и конфиги данных.
    Сохраняет всю обработку рядом с источником данных. (может даже для удобства в одной папке с файлом источника данных)
    Подаёдтся непосредственное обращение к источнику данных.
    WIP Отработать подгрузку континуальных и неконтинуальных данных.
    """

    def __init__(self, path, respec = False, RamGb_limit_usage = 2):
        """
        Инициализация источника данных.

        :param path: путь к файлу данных
        :param respec: If ``True``, force re-creation of metadata. Default ``False``.
        :param RamGb_limit_usage: RAM limit in GB for batch processing. Default ``2``.
        :param config: Processing config dict for this source. If ``None``,
            defaults to ``{}`` (will be resolved later by DataSet).
        :type path: str
        :type respec: bool
        :type RamGb_limit_usage: int or float
        :type config: dict or None
        """

        self.Ramcap = RamGb_limit_usage * (1024**3)
        self.file_path = path
        sample_name = os.path.splitext(os.path.basename(path))[0]
        folder_name = os.path.basename(os.path.dirname(path))
        if folder_name != sample_name:
            sample_name = folder_name + "_" + sample_name
        self.sample_name = sample_name
        self.loader = DataManager().get_loader(path)(path, respec = respec) # Composition pattern
        for attr_name in dir(self.loader):
            if not attr_name.startswith('_'): # Игнорируем приватные методы
                attr = getattr(self.loader, attr_name)
                if callable(attr):
                    setattr(self, attr_name, attr)


        dirpath = os.path.split(path)[0]
        processed_dirname = 'processed_pelmesha'
        configs_name = f'{sample_name}_processing_recipe'
        if os.path.exists(os.path.join(dirpath, processed_dirname, configs_name + '.yaml')):
            self.configs_path = os.path.join(dirpath, processed_dirname, configs_name + '.yaml')
        else:
            self.configs_path = None

        spectra_hdf5_name = sample_name + '_processed_spectra.hdf5'
        if os.path.exists(os.path.join(dirpath, processed_dirname, spectra_hdf5_name)):
            self.processed_spectra_path = os.path.join(dirpath, processed_dirname, spectra_hdf5_name)
        else:
            self.processed_spectra_path = None

        peaklist_hdf5_name = sample_name + '_peaklists.hdf5'
        if os.path.exists(os.path.join(dirpath, processed_dirname, peaklist_hdf5_name)):
            self.peaklist_path = os.path.join(dirpath, processed_dirname, peaklist_hdf5_name)
        else:
            self.peaklist_path = None
    
        # Выгрузка метаданных
        self.meta_file_path = os.path.join(os.path.dirname(self.file_path),'raw_pelmesha',sample_name + '_ingredients.hdf5')
        if not os.path.exists(self.meta_file_path):
            raise FileNotFoundError("Metadata file not found")
        with File(self.meta_file_path,"r") as hdf5:
            self.metadata = pd.DataFrame([dict(hdf5['metadata'].attrs.items())], index = [self.sample_name])
            columns_list = set()
            metadata_dict = {}
            for roi in hdf5['metadata']:
                columns_list.update(hdf5[f'metadata/{roi}'].attrs.keys())
                metadata_dict[roi] = dict(hdf5[f'metadata/{roi}'].attrs.items())
            self.roi_metadata = pd.DataFrame.from_dict(metadata_dict, orient='index', columns=list(columns_list))
    
    # def __getattr__(self, name):
    #     return getattr(self.loader, name)
    # ------------------------------------------------------------------ #
    #  Representations (__repr__ / _repr_html_)                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _file_url(path):
        """Convert a local path to a ``file:///`` URL (opens in OS file manager)."""
        from urllib.parse import quote
        return "file:///" + quote(path.replace("\\", "/"), safe="/:")

    def _repr_html_(self):
        """
        HTML summary of the DataSource for Jupyter Notebook.

        Shows file info, data type, and a table of ROIs with index ranges
        and m/z ranges.  The file path is a clickable link to the parent
        folder in the OS file manager.

        Returns
        -------
        str
            A valid HTML block.
        """
        dir_url = self._file_url(os.path.dirname(self.file_path))
        dcont_str = "Yes" if self.dcont else "No"
        dtype_str = self.metadata.get("dtype_raw", ["—"]).iloc[0]

        html = [
            '<div style="font-family: sans-serif; font-size: 13px;">',
            # --- header line: file name + folder link ----------------- #
            f'  <b>DataSource:</b> {self._escape_html(self.sample_name)}',
            f'  &nbsp;—&nbsp;'
            f'<a href="{dir_url}" target="_blank" '
            f'style="color: #1a73e8; text-decoration: none; font-size: 12px;">'
            f'📁 {self._escape_html(os.path.dirname(self.file_path))}</a>',
            # --- info table ------------------------------------------- #
            '  <table style="border-collapse: collapse; margin-top: 6px; '
            'font-size: 13px;">',
            '    <thead>',
            '      <tr>',
            '        <th style="border: 1px solid #999; padding: 4px 8px; '
            'background-color: #4a4a4a; color: #fff; text-align: left;">Property</th>',
            '        <th style="border: 1px solid #999; padding: 4px 8px; '
            'background-color: #4a4a4a; color: #fff; text-align: left;">Value</th>',
            '      </tr>',
            '    </thead>',
            '    <tbody>',
            f'      <tr style="background-color: #f9f9f9;">'
            f'<td style="border: 1px solid #ccc; padding: 3px 8px;">Continuous</td>'
            f'<td style="border: 1px solid #ccc; padding: 3px 8px;">{dcont_str}</td></tr>',
            f'      <tr style="background-color: #ffffff;">'
            f'<td style="border: 1px solid #ccc; padding: 3px 8px;">Raw dtype</td>'
            f'<td style="border: 1px solid #ccc; padding: 3px 8px; font-family: monospace;">'
            f'{self._escape_html(dtype_str)}</td></tr>',
            '    </tbody>',
            '  </table>',
        ]

        # --- ROI table ------------------------------------------------ #
        if len(self.roi_metadata) > 0:
            html.append(
                '  <table style="border-collapse: collapse; margin-top: 8px; '
                'font-size: 13px;">'
            )
            html.append('    <thead>')
            html.append('      <tr>')
            for col in ["ROI", "Index range", "m/z range"]:
                html.append(
                    f'        <th style="border: 1px solid #999; padding: 4px 8px; '
                    f'background-color: #4a4a4a; color: #fff; text-align: left;">'
                    f'{col}</th>'
                )
            html.append('      </tr>')
            html.append('    </thead>')
            html.append('    <tbody>')

            for roi_idx, (roi_name, roi_row) in enumerate(self.roi_metadata.iterrows()):
                bg = "#f9f9f9" if roi_idx % 2 == 0 else "#ffffff"
                idxroi = roi_row.get("idxroi")
                if isinstance(idxroi, np.ndarray):
                    if idxroi.ndim == 2 and idxroi.shape[1] == 2:
                        ranges = ", ".join(f"[{s}, {e})" for s, e in idxroi)
                    else:
                        ranges = str(idxroi)
                else:
                    ranges = str(idxroi)

                mz_range = roi_row.get("mz_range")
                if isinstance(mz_range, (tuple, list, np.ndarray)) and len(mz_range) == 2:
                    mz_str = f"{mz_range[0]:.4f} – {mz_range[1]:.4f}"
                else:
                    mz_str = str(mz_range)

                html.append(
                    f'      <tr style="background-color: {bg};">'
                    f'<td style="border: 1px solid #ccc; padding: 3px 8px; '
                    f'font-family: monospace;">{self._escape_html(str(roi_name))}</td>'
                    f'<td style="border: 1px solid #ccc; padding: 3px 8px; '
                    f'font-family: monospace;">{self._escape_html(ranges)}</td>'
                    f'<td style="border: 1px solid #ccc; padding: 3px 8px; '
                    f'font-family: monospace;">{self._escape_html(mz_str)}</td>'
                    f'</tr>'
                )

            html.append('    </tbody>')
            html.append('  </table>')

        html.append('</div>')
        return "\n".join(html)

    def __repr__(self):
        """
        Text summary of the DataSource for the console.

        Shows file name, data type, raw dtype, and a table of ROIs with
        index ranges and m/z ranges.

        Returns
        -------
        str
        """
        dcont_str = "Yes" if self.loader.dcont else "No"
        dtype_str = self.metadata.get("dtype_raw", ["—"]).iloc[0]

        lines = [
            f"DataSource: {self.sample_name}",
            f"  File:      {self.file_path}",
            f"  Continuous: {dcont_str}",
            f"  Raw dtype: {dtype_str}",
        ]

        if len(self.roi_metadata) > 0:
            # Build a mini text table for ROIs
            roi_rows = []
            for roi_name, roi_row in self.roi_metadata.iterrows():
                idxroi = roi_row.get("idxroi")
                if isinstance(idxroi, np.ndarray):
                    if idxroi.ndim == 2 and idxroi.shape[1] == 2:
                        ranges = ", ".join(f"[{s}, {e})" for s, e in idxroi)
                    else:
                        ranges = str(idxroi)
                else:
                    ranges = str(idxroi)

                mz_range = roi_row.get("mz_range")
                if isinstance(mz_range, (tuple, list, np.ndarray)) and len(mz_range) == 2:
                    mz_str = f"{mz_range[0]:.4f} – {mz_range[1]:.4f}"
                else:
                    mz_str = str(mz_range)

                roi_rows.append((str(roi_name), ranges, mz_str))

            # Dynamic column widths
            col_widths = [max(len(r[i]) for r in roi_rows) for i in range(3)]
            col_widths[0] = max(col_widths[0], len("ROI"))
            col_widths[1] = max(col_widths[1], len("Index range"))
            col_widths[2] = max(col_widths[2], len("m/z range"))

            hdr = (
                f"  {'ROI'.ljust(col_widths[0])} | "
                f"{'Index range'.ljust(col_widths[1])} | "
                f"{'m/z range'.ljust(col_widths[2])}"
            )
            sep = "  " + "-" * (sum(col_widths) + 2 * len(col_widths) - 1)
            lines.append("")
            lines.append(hdr)
            lines.append(sep)
            for r in roi_rows:
                lines.append(
                    f"  {r[0].ljust(col_widths[0])} | "
                    f"{r[1].ljust(col_widths[1])} | "
                    f"{r[2].ljust(col_widths[2])}"
                )

        return "\n".join(lines)

    @staticmethod
    def _escape_html(text):
        """Escape HTML special characters."""
        table = {
            "&": "&",
            '"': '"',
            "'": "'",
            ">": ">",
            "<": "<",
        }
        return "".join(table.get(c, c) for c in str(text))

    def _normalize_indices(self, idxs):
        if idxs is None:
            idxs = np.concatenate(self.roi_metadata['idxroi'].to_numpy())
        elif isinstance(idxs, (Indexator, SliceIndexator)):
            idxs = idxs.view(np.ndarray)
        elif isinstance(idxs, str):
            idxs = self.roi_metadata.loc[idxs, 'idxroi'].to_numpy()
        elif isinstance(idxs,slice):
            idxs = np.array((idxs.start,idxs.stop))
        elif isinstance(idxs,int):
            idxs = np.array([idxs,idxs+1])
        elif isinstance(idxs,(list, tuple)):
            idxs = np.array(idxs)

        if idxs.ndim == 1:
            idxs = idxs[np.newaxis, :]
        return idxs

    def _get_local_roi_idx(self, idxs, roi_name = None):
        """
        Convert global spectrum indices to local indices within a ROI,
        accounting for gaps (discontinuous segments) in the ROI.

        For a ROI with segments ``[[0, 21158], [30000, 30100]]``,
        global index ``30000`` maps to local index ``21158``
        (the cumulative offset of the first segment).

        Parameters
        ----------
        idxs : int or np.ndarray
            Global index or array of index segments (shape ``(n, 2)``).
            Accepts :class:`Indexator`, :class:`SliceIndexator`, ``int``,
            ``list``, ``tuple``, or a raw :class:`np.ndarray`.
        roi_name : str, optional
            ROI name. If ``None``, the ROI is auto-detected from the
            index range of *idxs*.

        Returns
        -------
        int or np.ndarray
            Local index (if *idxs* was an ``int``) or array of local index
            segments with the same shape as the input.

        Raises
        ------
        ValueError
            If *roi_name* is ``None`` and the index does not belong to any ROI.
        KeyError
            If *roi_name* is not found in ``roi_metadata``.
        """

        # --- Resolve ROI segments -------------------------------------- #
        if roi_name is None:
            roi_name = self._get_roi(self._normalize_indices(idxs))
        roi_segments = self.roi_metadata.loc[roi_name, 'idxroi']
        # roi_segments is an Indexator (ndarray subclass) with shape (M, 2)
        roi_segments = np.asarray(roi_segments, dtype=np.int64)
        if roi_segments.ndim == 1:
            roi_segments = roi_segments[np.newaxis, :]

        # --- Handle single int ----------------------------------------- #
        if isinstance(idxs, (int, np.integer)):
            # Find which ROI segment this index belongs to
            for seg_i, (start, end) in enumerate(roi_segments):
                if start <= idxs < end:
                    # Cumulative local start of this segment
                    local_start = np.sum(
                        np.diff(roi_segments[:seg_i], axis=1)
                    )
                    return int(idxs - (start - local_start))
            raise ValueError(
                f"Global index {idxs} does not belong to ROI '{roi_name}' "
                f"with segments {roi_segments.tolist()}"
            )
        # --- Normalise array input ------------------------------------- #
        idxs = self._normalize_indices(idxs)  # -> (N, 2)
        # --- Compute cumulative sizes and offsets per ROI segment ----- #
        sizes = np.diff(roi_segments, axis=1).ravel()                # (M,)
        cum_sizes = np.zeros(len(roi_segments), dtype=np.int64)
        cum_sizes[1:] = np.cumsum(sizes[:-1])                        # (M,) — local start of each segment
        # offset = global_start - local_start
        offsets = roi_segments[:, 0] - cum_sizes                     # (M,)

        # --- Map each batch segment to its ROI segment ---------------- #
        starts = idxs[:, 0, None]                                    # (N, 1)
        mask = (starts >= roi_segments[:, 0]) & \
               (starts <  roi_segments[:, 1])                        # (N, M)
        seg_idx = np.argmax(mask, axis=1)                            # (N,)

        # --- Subtract the corresponding offset ------------------------ #
        local_segments = idxs - offsets[seg_idx, None]

        return local_segments

    def get_coords(self, idxs = None, extract = None):
        idxs = self._normalize_indices(idxs)

        with File(self.meta_file_path,"r") as hdf5:
            coords = []
            if extract is None:
                extract = hdf5['coords'].keys()
            for idx in SliceIndexator(idxs):
                coords_slice = np.vstack((range(idx.start, idx.stop), *(hdf5[f'coords/{coord}'][idx] for coord in extract))).T
                coords.append(coords_slice)
            coords = pd.DataFrame(np.vstack(coords), columns = ['spectra_ind', *extract])
        coords['spectra_ind'] = coords['spectra_ind'].astype(np.int64)
        
        return coords.set_index('spectra_ind')

    def _get_roi(self, idx): 
        """
        WIP
        Метод возвращает первый roi, относящийся индексу или точь-в-точь тому полного сегмента/отрезка (пока не умеет определять нахождение в промежутке). 
        Функция сильно упрощённая для решения простой задачи - определить принадлежность к ПЕРВОМУ roi

        :param idx: Spectrum index or array of index segments.
        :type idx: int or np.ndarray or Indexator

        :return: ROI name (string) matching the given index.
        :rtype: str

        :raises ValueError: If no ROI matches the given index.
        """ # TODO Доработать для использования в объединённых датасетах (с мультиинексом, где ещё и sample)
        # in_bool = np.zeros(len(self.roi_metadata), dtype = bool) # TODO: Оставить до момента решения судьбы функции на вопрос её усложнения и возвращения нескольких roi
        
        for roi_num, indexes in enumerate(self.roi_metadata['idxroi'].values):
            if isinstance(idx, np.ndarray):
                for rng in idx:
                    range_bool = (indexes == rng).sum(axis = 1) == 2
                    if range_bool.any():
                        # in_bool[roi_num] = True
                        return self.roi_metadata.index[roi_num]
                for rng in idx:
                    if (rng[0]>=indexes[0,0]) & (rng[1]<=indexes[0,1]):
                        return self.roi_metadata.index[roi_num]
                # if range_bool.any(): 
                #     in_bool[roi_num] = True
                #     continue
            if isinstance(idx, int):
                if indexes.ndim ==1:
                    indexes = indexes[np.newaxis, :]
                if any(start <= idx < end for start, end in indexes):
                    return self.roi_metadata.index[roi_num]

        raise ValueError(f"ROI with index/indexes {idx} not found in {self.roi_metadata['idxroi'].values}")

    def get_mean_spectrum(self, idxs = None, mz_range = None):
        """
        Compute the mean (average) mass spectrum over the specified indices.

        Handles both continuous (TOF) and discontinuous (Orbitrap) data types,
        automatically detecting the data type and applying appropriate interpolation.

        :param idxs: Spectrum indices or ROI name. If ``None`` and only one ROI exists, uses that ROI.
        :param mz_range: Optional ``(mz_min, mz_max)`` tuple to restrict the m/z range.
        :type idxs: str or np.ndarray or None
        :type mz_range: tuple or None

        :return: Tuple ``(mz_scale, mean_intensity)`` where both are 1-D numpy arrays.
        :rtype: tuple
        """
        if isinstance(idxs, str):
            idxs = self.roi_metadata.loc[idxs, 'idxroi'] # TODO: Проверить работоспособность
        elif idxs is None:
            if len(self.roi_metadata) == 1:
                idxs = self.roi_metadata['idxroi'].iloc[0]
            else:
                raise ValueError(
                    f"Multiple regions (ROIs) found: {self.roi_metadata.index.tolist()}. In this case 'idxs' must be a ROI name (str) "
                    f"or an np.ndarray of continuous segments with shape (n, 2)"
                )
        if mz_range is not None:
            mz_min, mz_max = mz_range
        if self.dcont:
            d = 1
        else:
            d = 2

        if self.dcont:
            mz = self.mz_scale_cont

            if mz_range is not None:
                start_idx = np.searchsorted(mz, mz_min, side='left')
                stop_idx = np.searchsorted(mz, mz_max, side='right')
                mz_range_slice = slice(start_idx, stop_idx)
            else:
                mz_range_slice = slice(None)
            mz = mz[mz_range_slice]
            stats_sum = np.zeros(len(mz))
            stats_count = np.zeros(len(mz))
            indexes = Indexator(idxs)

            for intens in self.get_intensities_stream(indexes):
                stats_sum += intens[mz_range_slice]
            stats_count = indexes.count()
            # plt.plot(mz, stats_sum/stats_count)
            # plt.show()
            return mz, stats_sum/stats_count
        else:
            idxs_batches = self.split_idxs(idxs = idxs, cpu_count = 1, d = d)
            mz_range_slice = slice(None)
            pilot_batch = idxs_batches.pop()
            if np.diff(pilot_batch) < 5: # Проверка на то, что pilot_batch содержит мало масс спектров, на основе которых невозможно будет оценить надёжно орбитрепные это данные или нет. 5 спектров минимум - эмпирическое
                pilot_batch = np.array([idxs_batches.pop()[0], pilot_batch[1]])

            batch_mz = []
            batch_intens = []
            for mz, intens in self.get_batch(Indexator(pilot_batch)):
                if mz_range is not None:
                    start_idx = np.searchsorted(mz, mz_min, side='left')
                    stop_idx = np.searchsorted(mz, mz_max, side='right')
                    mz_range_slice = slice(start_idx, stop_idx)
                    mz = mz[mz_range_slice]
                    intens = intens[mz_range_slice]
                batch_mz.append(mz)
                batch_intens.append(intens)

            values, counts = np.unique(np.concatenate(batch_mz, axis=0), return_counts=True)
            if len(values[counts > 1])/len(values) < 0.33: # Если общих точек дискретизации m/z, которые совпадают хотя бы раз со всех масс спектров пробного батча, менее 33%, то считаем, что полученные данные с орбитрепа.
                roi = self._get_roi(pilot_batch)
                if mz_range is None:
                    mz_min, mz_max = self.roi_metadata['mz_range'][roi]
                discret_coeffs = self.roi_metadata.loc[roi, 'discret_coeffs']
                mz_discretion_model = np.poly1d(discret_coeffs)
                mz_scale = [mz_min]
                mz_end = mz_min
                while mz_end <= mz_max:
                    mz_end += mz_discretion_model(mz_end)
                    mz_scale.append(mz_end)
                mz_scale = np.array(mz_scale)
                stats_sum = np.zeros(len(mz_scale))
                stats_count = np.zeros(len(mz_scale))
                
                for mz_loc, intens_loc in zip(batch_mz, batch_intens):
                    if len(mz_loc) == 0:
                        continue
                    stats_sum += np.interp(mz_scale, mz_loc, intens_loc, left=0, right=0)
                    stats_count[np.unique(np.searchsorted(mz_scale, mz_loc, side='left'))] += 1
                del batch_mz, batch_intens

                for idxs_batch in idxs_batches:
                    for mz_loc, intens_loc in self.get_batch(Indexator(idxs_batch)):
                        if mz_range is not None:
                            start_idx = np.searchsorted(mz_loc, mz_min, side='left')
                            stop_idx = np.searchsorted(mz_loc, mz_max, side='right')
                            mz_range_slice = slice(start_idx, stop_idx)
                            mz_loc = mz_loc[mz_range_slice]
                            if len(mz_loc) == 0:
                                continue
                            intens_loc = intens_loc[mz_range_slice]
                        stats_sum += np.interp(mz_scale, mz_loc, intens_loc, left=0, right=0)
                        stats_count[np.unique(np.searchsorted(mz_scale, mz_loc, side='left'))] += 1
                nonzero_count_bool = stats_count != 0
                mean_intensity = np.zeros(len(mz_scale))
                mean_intensity[nonzero_count_bool] = stats_sum[nonzero_count_bool]/stats_count[nonzero_count_bool]
                return mz_scale, mean_intensity

            else:
                stats = pd.DataFrame(np.vstack([np.concatenate(batch_mz, axis=0), np.concatenate(batch_intens, axis=0)]).T, columns=['mz','intensities']).groupby('mz')['intensities'].agg(['sum', 'count'])
                batch_mz = []
                batch_intens = []
                for idxs_batch in idxs_batches:
                    for mz, intens in self.get_batch(Indexator(idxs_batch)):
                        if mz_range is not None:
                            start_idx = np.searchsorted(mz, mz_min, side='left')
                            stop_idx = np.searchsorted(mz, mz_max, side='right')
                            mz_range_slice = slice(start_idx, stop_idx)
                            mz = mz[mz_range_slice]
                            intens = intens[mz_range_slice]
                        batch_mz.append(mz)
                        batch_intens.append(intens)
                    stats = stats.add(pd.DataFrame(np.vstack([np.concatenate(batch_mz, axis=0), np.concatenate(batch_intens, axis=0)]).T, columns=['mz','intensities']).groupby('mz')['intensities'].agg(['sum', 'count']), fill_value=0)
                return stats.index, stats['sum']/stats['count']
            
    def split_idxs(self, idxs = None, d = None, cpu_count = 1, Ramcap_GB = None, size_per_spec = None):
        """
        Split spectrum indices into batches based on RAM cap configs.

        :param idxs: Index segments to split. If ``None``, uses all available spectra.
        :param d: Data dimensionality factor (1 for continuous, 2 for discontinuous), обозначает количество используемых в батчах векторов. 
        Для континуальных данных d = 1, так как достаточно обрабатывать один вектор 'y' (интенсивность), а mz у всех общее и под него выделить память - единожды, 
        но для неконтинуальных данных необходимо выделить память для mz и y, так как для каждого спектра свой mz. Default ``None`` - функция выберет автоматически.
        :param cpu_count: Number of CPU cores to account for in RAM budgeting. Default ``1``.
        :param Ramcap_GB: RAM capacity in gigabytes. Default ``None`` - uses ``self.Ramcap``.
        :param size_per_spec: Size of each spectrum in dots. Used for batching processing with resampling spectra. Default ``None`` - uses ``self.spectrum_sizes``.
        
        :type idxs: np.ndarray or None
        :type d: int
        :type cpu_count: int
        :type Ramcap_GB: float
        :type size_per_spec: int

        :return: List of index segment arrays, each fitting within the RAM budget.
        :rtype: list
        """
        spectrum_sizes = self.get_spectrum_sizes(idxs)
        idxs_batches = []
        remainder = []
        if d is None:
            if self.dcont:
                d = 1
            else:
                d = 2
        if Ramcap_GB is None:
            Ramcap = self.Ramcap 
        else:
            Ramcap = Ramcap_GB * (1024 ** 3)
        Ram_usage_per_batch = Ramcap/ (cpu_count * d)
        
        idxs = self._normalize_indices(idxs)

        length_per_batch = Ram_usage_per_batch // np.dtype(self.metadata['dtype_raw'].iloc[0]).itemsize
        if self.dcont or size_per_spec:
            if size_per_spec:
                dots_num_per_spec = size_per_spec
            else:
                dots_num_per_spec = spectrum_sizes[0]
            chunk_size = length_per_batch // dots_num_per_spec
            remainder_count = 0
            
            for idx in idxs:
                
                segment_size = np.diff(idx)
                if remainder:
                    if remainder_count+segment_size >= chunk_size:
                        idx_rem_stop = idx[0] + chunk_size - remainder_count
                        idxs_batches.append(np.vstack(remainder.append([idx[0], idx_rem_stop])))
                        idx = np.array([idx_rem_stop, idx[1]])
                        segment_size = remainder_count+segment_size - chunk_size
                        assert segment_size == np.diff(idx)
                        remainder_count = 0
                        remainder = []
                    else:
                        remainder.append(idx)
                        remainder_count = remainder_count + segment_size
                if segment_size >= chunk_size:
                    for idx_batch in pairwise(np.arange(idx[0], idx[1], chunk_size)):
                        idxs_batches.append(np.array(idx_batch))
                    if idx[1] != idx_batch[1]:
                        remainder.append([idx_batch[1], idx[1]])
                        remainder_count = remainder_count + idx[1] - idx_batch[1]
                else:
                    remainder.append(idx)
                    remainder_count = remainder_count + segment_size
            if remainder:
                idxs_batches.append(np.vstack(remainder))
                remainder = []
                remainder_count = 0
        else:
            length_usage = 0
            idxs = Indexator(idxs)
            idx_start = idxs[0][0]
            idx_last = idx_start - 1
            for n, idx in enumerate(idxs):
                if idx - idx_last > 1: # Учёт сегментного разрыва в индексации
                    remainder.append(np.array([idx_start, idx_last + 1]))
                    idx_start = idx
                    
                if length_usage + spectrum_sizes[n] >= length_per_batch:
                    if remainder:
                        idxs_batches.append(np.vstack(remainder))
                        remainder = []
                    else:   
                        idxs_batches.append(np.array([idx_start, idx + 1]))
                    idx_start = idx
                    length_usage = 0
                else:
                    length_usage += spectrum_sizes[n]
                idx_last = idx

            if length_usage != 0:
                if remainder:
                    idxs_batches.append(np.vstack(remainder))
                    remainder = []
                else:
                    idxs_batches.append(np.array([idx_start, idx + 1]))

        return idxs_batches
    @property
    def config_path(self):
        path = os.path.join(os.path.dirname(self.file_path), 'processed_pelmesha', 'processing_recipe.yaml')
        return path if os.path.exists(path) else None
    @property
    def dcont(self):
        """get dcont from loader"""
        return self.loader.dcont
    @property
    def close(self):
        """
        Close the data source.
        """
        self.loader.close()

SUPPORTED_FILE_EXTENSIONS = ['.imzml', '.cdf']

class BaseLoader(ABC): #TODO @задачка: Базовый абстрактный класс
    """
    Базовый класс для всех подгрузчиков данных с абстрактными и общими методами.
    Ни один конкретный подгрузчик не запустится, если в нем не будут определены ряд абстрактных методов, индивидуальные для них.
    """
    def create_metafile(self, draw = True, respec = False, chunk_Mbsize = 10):
        """
        Create or update the metadata HDF5 file (``<sample_name>_ingredients.hdf5``) for the data source.

        Stores metadata attributes, ROI metadata, and coordinate data (xy, z or another) in a structured
        HDF5 file under the ``raw_pelmesha`` subdirectory.

        :param draw: If ``True``, display diagnostic plots during metadata extraction. Default ``True``.
        :param respec: If ``True``, force re-creation of the metadata file even if it exists. Default ``False``.
        :param chunk_Mbsize: Chunk size in MB for HDF5 dataset storage. Default ``10``.

        :type draw: bool
        :type respec: bool
        :type chunk_Mbsize: int
        """ 
        file_path = self.file_path                                             
        base_folder_path = os.path.dirname(file_path)
        meta_file_folder = os.path.join(base_folder_path,'raw_pelmesha')
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        folder_name = os.path.basename(os.path.dirname(file_path))
        if folder_name != sample_name:
            sample_name = folder_name + "_" + sample_name

        meta_file_name = sample_name +'_ingredients.hdf5'
        meta_file_path = os.path.join(base_folder_path,'raw_pelmesha', meta_file_name)
        meta_file_exist = os.path.exists(meta_file_path)
        if not os.path.exists(meta_file_folder):
            os.makedirs(meta_file_folder)
        if (not meta_file_exist) or respec:
            print('new_metadata_file!!!')
            chunk_Mbsize = chunk_Mbsize * (1024**2)
            metadata, roi_metadata, coords = self.get_metadata(draw) # TODO @задачка: Создать в новом классе выгрузки данных из cdf метод get_metadata, чтобы сработал этот метод. 
                                                                        #             Отмечу: coords в cdf - это RT, а roi - это по планам каналы 0-3 (где разные масс анализаторы и моды) 

            del_hdf5(meta_file_path) # del hdf5 if it exists
            
            with File(meta_file_path, 'w') as f:
                f.create_group('metadata')
                for key, value in metadata.items():
                    f['metadata'].attrs[key] = value

                for roi in roi_metadata.keys():
                    print(roi)#TODO
                    f['metadata'].create_group(roi)
                    for key, value in roi_metadata[roi].items():
                        f[f'metadata/{roi}'].attrs[key] = value
                
                f.create_group('coords')
                for coord_key, data in coords.items():
                    val_size = data.shape
                    if len(val_size) == 1:
                        f.create_dataset(f"coords/{coord_key}", data=data, dtype=data.dtype, chunks = True)
                    else:
                        auto_row = int(chunk_Mbsize/(data.itemsize * val_size[1])) 
                        auto_row = val_size[0] if auto_row > val_size[0] else auto_row
                        f.create_dataset(f'coords/{coord_key}', data=data, dtype=data.dtype, chunks = (auto_row, val_size[1]))

    def get_mz_range(self, idxs): 
        """
        Determine the minimum and maximum m/z values across the specified spectra.

        :param idxs: Index segments to scan.
        :type idxs: np.ndarray

        :return: Tuple ``(mz_min, mz_max)``.
        :rtype: tuple
        """
        if self.dcont:
            mz = self.mz_scale_cont
            return (mz[0], mz[-1])
        
        min_mz = np.inf
        max_mz = -np.inf

        for mz in self.get_mz_stream(idxs):
            min_mz = min(min_mz, mz[0])
            max_mz = max(max_mz, mz[-1])
        return (min_mz, max_mz)
    
    def get_mz_discretion_coeffs(self, idxs, degree = 3, mz_range = None, draw = True):
        """
        Fit polynomial coefficients describing the m/z discretisation step size.

        Uses a least-squares regression on the median-filtered m/z differences.

        :param idxs: Index segments to use for fitting.
        :param degree: Polynomial degree for the fit. Default ``3``.
        :param mz_range: Optional ``(mz_min, mz_max)`` to restrict the fitting range.
        :param draw: If ``True``, display a diagnostic plot. Default ``True``.

        :type idxs: np.ndarray
        :type degree: int
        :type mz_range: tuple or None
        :type draw: bool

        :return: Polynomial coefficients (highest degree first).
        :rtype: np.ndarray
        """
        nsize_matrix = degree + 1
        sum_XTX = np.zeros((nsize_matrix, nsize_matrix))
        sum_XTy = np.zeros(nsize_matrix)
        if self.dcont: #Достаточно одного спектра
            start_idx = idxs[0][0]
            idxs = Indexator([start_idx, start_idx+1])
        if mz_range:
            mz_min, mz_max = mz_range
        else:
            mz_min, mz_max = self.get_mz_range(idxs)
        for mz in self.get_mz_stream(idxs):
            mz_vander = np.vander(mz[:-1], nsize_matrix)
            sum_XTX += mz_vander.T @ mz_vander
            sum_XTy += mz_vander.T @ medfilt(np.diff(mz),5)
        discret_coeffs = np.linalg.solve(sum_XTX, sum_XTy)

        #перепроверка на равномерность дискретизации шкалы
        mz_discret = np.poly1d(discret_coeffs)(np.linspace(mz_min, mz_max, 100000))
        discret_mean = np.mean(mz_discret)
        discret_std = np.std(mz_discret,ddof=1)
        if discret_std/discret_mean < 0.025: #равномерная дискретизация
            discret_coeffs = np.linalg.solve(sum_XTX[degree:,:1], sum_XTy[:1])

        if draw:
            plt.figure(figsize = (25,4))
            plt.plot(mz[:-1], medfilt(np.diff(mz),5))
            plt.plot(mz[:-1], np.poly1d(discret_coeffs)(mz[:-1]))
            plt.legend(["median filtered m/z discretization example", "m/z discretization regression"])
            plt.xlabel('m/z')
            plt.ylabel('m/z discretion')
        return discret_coeffs

    @abstractmethod
    def get_spectrum(self, idx):
        """Обязательный метод для загрузки одного масс спектра"""
        pass

    @abstractmethod
    def get_metadata(self):
        """Обязательный метод для извлечения метаданных"""
        pass 
    
    @abstractmethod
    def get_mz(self, idx):
        """Обязательный метод для извлечения mz шкалы масс спектра"""
        pass

    @abstractmethod
    def get_intensity(self, idx):
        """Обязательный метод для извлечения интенсивности масс спектра"""
        pass

    @abstractmethod
    def get_batch(self, idxs): # TODO Возможно нужно просто удалить и заменить там где используется на get_spectra_stream, так как попытки написать универсальную работу как матрицей спектров, так и индивидиульаных - слишком усложняет код, что 1) ломает гибкость 2) усложняет написание и сам код 3) не факт что сильно эффективнее выходит
        """Обязательный метод для загрузки пачки масс спектров либо в ленивом формате, если данные неконтинуальные, либо матрицей, если данные континуальные"""
        pass
    
    @abstractmethod
    def get_mz_stream(self, idxs = None):
        """Обязательный метод для извлечения нескольких mz шкал масс спектра ленивым методом"""
        pass

    @abstractmethod
    def get_intensities_stream(self, idxs = None):
        """Обязательный метод для извлечения нескольких интенсивностей масс спектра ленивым методом"""
        pass
    @abstractmethod
    def get_spectra_stream(self, idxs = None):
        """Обязательный метод для загрузки пачки масс спектров в ленивом формате"""
        pass
    @abstractmethod
    def get_spectrum_sizes(self, idxs = None):
        """Обязательный метод для извлечения размеров масс спектров"""
        pass

###################### IMZML
class loader_imzml(BaseLoader): #TODO @задачка: Класс выгрузки как пример 
                                #TODO @задачка: Совет: есть особый класс Indexator, упрощающий индексацию и хранение индексов. Просто пишу краткое пояснение к нему:
                                #  Индексы разбиваются на непрерывные сегменты и сегмент обозначен начальным и конечным индексом. Класс может считать кол-во индексов в сложных случаях (когда много сегментов) (свойство .count).
                                #  Класс используется часто для удобного итерирования. У него есть "родственник" SliceIndexator: Тоже самое, но возвращает при итерации слайсы  
    """                         
    Loader for imzML mass spectrometry imaging data.

    Automatically detects whether the data is continuous (TOF) or discontinuous (Orbitrap)
    and selects the appropriate batch-loading strategy.
    """
    def __init__(self, file_path, respec = False):
        self.file_path = file_path
        source = ImzMLParser(file_path)
        # Назначение функции подгрузки в зависимости от континуальности данных 
        mzoffsets = source.mzOffsets
        if mzoffsets[0] == mzoffsets[1]:
            self.dcont = True
            self.mz_scale_cont = source.getspectrum(0)[0]
        #     self.get_batch = self._get_batch_cont
        else:
            self.dcont = False
        #     self.get_batch = self._get_batch_discont
        self.create_metafile(respec = respec)
    @cached_property
    def source(self):
        return ImzMLParser(self.file_path)
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('source', None) 
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
    def get_spectra_stream(self, idxs):
        """
        Lazily yield individual discontinuous spectra.

        :param idxs: Index segments to load.
        :type idxs: Indexator

        :yields: Tuple ``(mz_array, intensity_array)`` for each spectrum.
        """
        for idx in idxs:
            yield self.source.getspectrum(idx)
    def _get_batch_cont(self, idxs):
        """
        Load a batch of continuous spectra as a single matrix.

        :param idxs: Index segments to load.
        :type idxs: Indexator

        :yields: Tuple ``(mz_scale, intensity_matrix)``.
        """
        yield self.mz_scale_cont, np.vstack(tuple(self.get_intensities_stream(idxs)))
    
    def _get_batch_discont(self, idxs):
        """
        Lazily yield individual discontinuous spectra.

        :param idxs: Index segments to load.
        :type idxs: Indexator

        :yields: Tuple ``(mz_array, intensity_array)`` for each spectrum.
        """
        for idx in idxs:
            yield self.source.getspectrum(idx)

    def _poslog_parser(self, poslog_path,specnum):
        """
        Parse a Bruker ``_poslog.txt`` file to extract ROI assignments and coordinates.

        :param poslog_path: Path to the ``_poslog.txt`` file.
        :param specnum: Total number of spectra expected.

        :type poslog_path: str
        :type specnum: int

        :return: Tuple ``(roi_list, roi_idx, poslog_specdata)`` where ``roi_idx`` maps ROI names
            to :class:`Indexator` segments and ``poslog_specdata`` is a list of ``(roi, x, y, z)`` tuples.
        :rtype: tuple
        """
        idx=0
        roi_idx = {} 
        roi_list = []
        current_roi = None
        poslog_specdata = [None]*specnum
        roi_pattern = re.compile(r'R(.+?)X')

        with open(poslog_path) as f:
            data = f.readlines()[2:]
        for line in data:
            line_search = roi_pattern.search(line)
            if not line_search:
                continue
            roi_num = line_search.group(1)
            coords =  line.split(' ')[-3:]
            try:
                x, y, z = map(float, coords)
            except (ValueError, IndexError):
                continue 
            if roi_num != current_roi:
                if current_roi is not None:
                    roi_idx[current_roi] = Indexator([start_idx, idx])  
                current_roi = roi_num                                  
                start_idx = idx                                         
                roi_list.append(roi_num)
            poslog_specdata[idx]=(x, y, z)
            idx += 1
        
        # Final ROI update
        if current_roi:
            roi_idx[current_roi] = Indexator((start_idx, idx))
        # return roi_list, roi_idx, poslog_specdata # 160626 перепись легаси функции, теперь координаты выгружаются как общие метаданные в виде единого массива. Суть, выгружать конкретные - позже, 
        # так как есть целая таблица с индексацией
        return roi_list, roi_idx, np.vstack(poslog_specdata)
    
    def get_mz_stream(self, idxs = None):
        if idxs is None:
            idxs = range(len(self.source.mzLengths))
        if self.dcont:
            for idx in idxs:
                yield self.mz_scale_cont
        else:
            for idx in idxs:
                yield self.source.getspectrum(idx)[0]
    def get_intensities_stream(self, idxs):
        for idx in idxs:
            yield self.source.getspectrum(idx)[1]
    
    def get_spectrum(self, idx):
        return self.source.getspectrum(idx)
    
    def get_mz(self, idx = None):
        if self.dcont:
            return self.mz_scale_cont
        else:
            return self.source.getspectrum(idx)[0]
        
    def get_intensity(self, idx):
        return self.source.getspectrum(idx)[1]
    
    def get_batch(self, idxs):
        """Заглушка от @abstractmethod, get_batch назначится динамически после __init__"""
        pass
    def get_metadata(self, draw = True): 
        metadata = {}
        roi_metadata = {}
        specdata = {}
        file_path = self.file_path
        sample_name = os.path.splitext(os.path.basename(file_path))[0] # Имя файла без расширения
        folder_path = os.path.dirname(file_path)

        base_path = os.path.join(folder_path, sample_name) # Базовый стринг пути к файлам без расширения
        poslog_path = os.path.join(folder_path, sample_name) + "_poslog.txt"
        if os.path.exists(base_path+"_info.txt"):
            with open(base_path+"_info.txt") as f:
                data_info = f.readlines()
                specnum = int(data_info[2].split(' ')[-1]) # Информация по кол-ву спектров в sample
        else:
            dpoints = self.source.mzLengths
            specnum = len(dpoints)
        if os.path.exists(poslog_path):
            roi_list, roi_idx, poslog_specdata = self._poslog_parser(poslog_path, specnum)
            specdata["x"] = poslog_specdata[:,0]
            specdata['y'] = poslog_specdata[:,1]
            specdata["z"] = poslog_specdata[:,2]
        else: # If there is no poslog file in the folder, take coordinates from imzml
            #Get base info and coordinates
            roi = "00" # Only one roi
            roi_list = [roi]
            roi_idx = {}
            roi_idx[roi] = Indexator((0,specnum))
            specdata = {}
            try:
                coords = np.vstack(list(self._get_physical_coordinates(range(specnum)))) 
                
            except:
                coords =  np.array(self.source.coordinates)[:,[0,1]]
            specdata["x"] = coords[:,0]
            specdata['y'] = coords[:,1]
            specdata["z"] = np.array([0]*specnum)
        for roi in roi_list:
            roi_metadata[roi] = {}
            roi_metadata[roi]["idxroi"] = roi_idx[roi]
        metadata["continuous"] = self.dcont
        metadata["dtype_raw"] = self.get_spectrum(0)[1].dtype.name

        for roi in roi_list:
            # находим коэффициенты для интерполяции дискретизации (для экономии хранения в метаданных, так как mz_scale может выйти на некоторых приборах в несколько ГБ)
            roi_metadata[roi]["mz_range"] = self.get_mz_range(roi_idx[roi])
            roi_metadata[roi]["discret_coeffs"] = self.get_mz_discretion_coeffs(roi_idx[roi], degree = 3, mz_range =  roi_metadata[roi]["mz_range"], draw = draw)
            if draw:
                plt.title(f'm/z discretization in sample {os.path.basename(file_path)} and roi {roi}')
                plt.show()
                
        # return metadata, roi_metadata, roi_data
        return metadata, roi_metadata, specdata

    def get_spectrum_sizes(self, idxs = None) -> list:
        if idxs is None:
            return self.source.mzLengths
        else:
            if isinstance(idxs, (np.ndarray, SliceIndexator)):
                if isinstance(idxs, np.ndarray):
                    idxs = SliceIndexator(idxs)
                mz_lengths = []
                for idxs_slice in idxs:
                    mz_lengths.extend(self.source.mzLengths[idxs_slice])
                return mz_lengths
            elif isinstance(idxs, int):
                return [self.source.mzLengths[idxs]]
            else:
                raise ValueError("Invalid index type")
    def _get_physical_coordinates(self, idxs):
        for idx in idxs:
            yield self.source.get_physical_coordinates(idx)
class loader_hdf5(BaseLoader): #TODO: Написать
        """
        Загрузка данных из источника.
        В зависимости от формата файла, данные загружаются разными способами.
        """
        def __init__(self, source, dtypeconv):
            self.source = source
            self.dtypeconv = dtypeconv
            self.mz_scale = source.getspectrum(0)[0]
            self.sample_metadata = {}

class loader_mzxml(BaseLoader): # TODO дописать.
    pass


class loader_cdf(BaseLoader):  # TODO дописать. #TODO @задачка: Собственно сам класс, который надо написать
    def __init__(self, file_path, respec = False):
        self.file_path = file_path
        source = xr.load_dataset(self.file_path)
        # фактически данные изначально всегда неконтинуальные
        self.dcont = False

        self.create_metafile(respec=respec)  # фактически метаданные находятся в самом cdf
    @cached_property
    def source(self):
        return xr.load_dataset(self.file_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('source', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _load_spectra(self, idx):
        """собираем один спектр по его индексу"""

        start_idx = self.source['scan_index'][idx].item()
        end_idx = self.source['point_count'][idx].item() + start_idx
        dot_selection = slice(start_idx, end_idx,1)
        mz_single = self.source['mass_values'][dot_selection].values
        intens_single = self.source['intensity_values'][dot_selection].values

        return mz_single, intens_single

    def get_spectra_stream(self, idxs = None):
        for idx in idxs:
            yield self._load_spectra(idx)

    def get_mz_stream(self, idxs = None):
        for idx in idxs:
            yield self._load_spectra(idx)[0]

    def get_intensities_stream(self, idxs = None):
        for idx in idxs:
            yield self._load_spectra(idx)[0]

    def get_spectrum(self, idx):
        return self._load_spectra(idx)

    def get_mz(self, idx):
        return self._load_spectra(idx)[0]

    def get_intensity(self, idx):
        return self._load_spectra(idx)[1]

    def get_batch(self, idxs):
        """заглушка"""
        pass

    def get_metadata(self, draw=True):
        metadata = {}
        roi_metadata = {}
        specdata = {}
        file_path = self.file_path

        # roi - каналы записи - scan_filters
        roi_list = np.unique(self.source["scan_filters"])
        roi_idx = {int(i): Indexator(_to_index(np.where(self.source['scan_filters'].values == i)[0])) for i in
                   roi_list}



        # в cdf нет информации о координатах спектров

        for roi in roi_list:
            roi_metadata[str(roi)] = {}
            roi_metadata[str(roi)]['idxroi'] = roi_idx[roi]



        # данные в cdf discontinuous
        metadata['continuous'] = self.dcont
        metadata['dtype_raw'] = self._load_spectra(0)[1].dtype.name

        # дискретизация
        for roi in roi_list:
            try:
                roi_range = np.array(self.source['scan_filters'].values == roi, dtype=bool)
                max_by_spec = self.source['scan_highmz'][roi_range].values
                min_by_spec = self.source['scan_lowmz'][roi_range].values
                roi_metadata[str(roi)]["mz_range"] = (min(min_by_spec), max(max_by_spec))
            except:
                roi_metadata[str(roi)]["mz_range"] = self.get_mz_range(roi_idx[roi])


            roi_metadata[str(roi)]["discret_coeffs"] = self.get_mz_discretion_coeffs(roi_idx[roi], degree=3,
                                                                                mz_range=roi_metadata[str(roi)]["mz_range"],
                                                                                draw=draw)
            if draw:
                plt.title(f'm/z discretization in sample {os.path.basename(file_path)} and roi {roi}')
                plt.show()


        return metadata, roi_metadata, specdata

    def get_spectrum_sizes(self, idxs = None):
        if isinstance(idxs, (np.ndarray, SliceIndexator)):
            if isinstance(idxs, np.ndarray):
                idxs = SliceIndexator(idxs)
            spec_lengths = []
            for idx_slice in idxs:
                spec_lengths.extend(self.source['point_count'].values[idx_slice])
            return spec_lengths
        elif isinstance(idxs, int):
            return [idxs]
        else:
            raise ValueError("Invalid index type")


class DataManager():
    """
    WIP
    для работы с различными источниками масс-спектрометрических данных (IMZML, HDF5, MZXML). Подгружает необходимый класс для загрузки данных и
    Обеспечивает унифицированный интерфейс для получения данных m/z шкалы и интенсивностей спектра и метаданных."""
    def __init__(self):
        self._loaders = {
            'imzml': loader_imzml,
            'hdf5': loader_hdf5,
            'mzxml': loader_mzxml,
            'cdf': loader_cdf        
        }
    def get_loader(self, file_path):
        """
        Return the appropriate loader class for the given file path based on its extension.

        :param file_path: Path to the data file.
        :type file_path: str

        :return: Loader class corresponding to the file extension.
        :rtype: class

        :raises ValueError: If the file extension is not supported.
        """
        file_ext = os.path.splitext(file_path)[1][1:]
        loader = self._loaders.get(file_ext.lower(), None)
        if not loader:
            raise ValueError(f'Format {file_ext} is not supported')
        return loader

############### Utility functions

def get_mz_discretion_coeffs_legacy(mz, min_discret = None, draw = True):
    """
    Legacy function to estimate m/z discretisation coefficients from a single m/z array.

    :param mz: 1-D array of m/z values.
    :param min_discret: Optional minimum discretisation threshold for filtering.
    :param draw: If ``True``, display a diagnostic plot. Default ``True``.

    :type mz: np.ndarray
    :type min_discret: float or None
    :type draw: bool

    :return: Polynomial coefficients describing the m/z step size.
    :rtype: np.ndarray
    """
    dots_distance = np.diff(mz)
    
    if min_discret:
        float_error_bool = dots_distance >= min_discret - math.sqrt(np.finfo(float).eps)
        if not float_error_bool[0]:
            first_mz = True
        else:
            first_mz = False
        float_error_bool = np.append(first_mz,float_error_bool)
        mz = mz[float_error_bool]
        dots_distance = np.diff(mz)
    distance_diff = np.diff(dots_distance)
    std_diff = np.std(dots_distance, ddof=1) / np.sqrt(len(dots_distance))
    dots_distance_bool = np.abs(distance_diff) <= std_diff
    start_diff = (dots_distance[0] - dots_distance[1:][dots_distance_bool][0])
    if abs(start_diff) <= std_diff:
        dots_distance_bool = np.append(True, dots_distance_bool)
    else:
        dots_distance_bool = np.append(False, dots_distance_bool)
    
    mz_discret = np.array(medfilt(dots_distance[dots_distance_bool],5))
    mz = mz[:-1][dots_distance_bool]

    discret_mean = np.mean(mz_discret)
    discret_std = np.std(mz_discret,ddof=1)
    if discret_std/discret_mean < 0.025:
        discret_coeffs = np.polyfit(mz, mz_discret, deg = 0)
    else:
        discret_coeffs = np.polyfit(mz, mz_discret, deg = 3)
        if draw:
            plt.figure(figsize = (25,4))
            plt.plot(mz, mz_discret)
            plt.plot(mz, np.poly1d(discret_coeffs)(mz))
            plt.legend(["m/z discretization", "m/z discretization regression"])
            plt.xlabel('m/z')
            plt.ylabel('m/z discretion')
    return discret_coeffs
def _batch_mz_discret_props_legacy(source, idxs):
    """
    Legacy function to collect unique m/z values and minimum discretisation step from a batch of spectra.

    :param source: Data loader with a ``get_mz_stream`` method.
    :param idxs: Index segment ``(start, end)`` to scan.

    :type source: BaseLoader
    :type idxs: tuple

    :return: Tuple ``(unique_mz_array, min_discret_step)``.
    :rtype: tuple
    """
    mzs = []
    min_discret_batch = np.inf
    for mz in source.get_mz_stream(range(*idxs)):
        mzs.append(mz)
        min_discret_batch = min(min_discret_batch, np.diff(mz).min())
    mz_batch = np.unique(np.hstack(mzs))
    return mz_batch, min_discret_batch

def _to_index(arr):
    ranges = []
    for _, group in itertools.groupby(enumerate(arr), lambda pair: pair[1] - pair[0]):
        group_list = list(group)
        start = group_list[0][1]
        end = group_list[-1][1] + 1
        ranges.append((start, end))
    return ranges