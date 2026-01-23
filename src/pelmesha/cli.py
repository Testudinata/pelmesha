import os

import numpy as np

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import copy
import math
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Dict,Any
import typer
import h5py
import scipy.stats as stats
import yaml
from KDEpy import FFTKDE
from diptest import diptest
from numba import njit
import pelmesha.alignment as alignment
from pelmesha.data_classes import *
from scipy.special import rel_entr
import pandas as pd
import tqdm
from rich.console import Console
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

console = Console(stderr=True)
app = typer.Typer(help="An utility for assessing the quality of MSI data alignment")


"""classes declaration"""

class DatasetHeaders:
    """
    Helper to access HDF5 dataset attributes by name or index.

    Parameters
    ----------
    attrs : Sequence[str]
        List of attribute names as provided by the HDF5 dataset.

    Attributes
    ----------
    index : dict
        Mapping from attribute name to its integer index.
    name : list
        List of attribute names in positional order.
    """
    def __init__(self,attrs):
        """
        Build the name-to-index and index-to-name mappings.

        Parameters
        ----------
        attrs : Sequence[str]
            Attribute names from the dataset.
        """
        self.index = {}
        self.name = [0]*len(attrs)
        for index, name in enumerate(attrs):
            self.name.append(name)
            self.index[name]=index

    def __call__(self,index_value):
        """
        Convert between names and indices for single values or lists.

        Parameters
        ----------
        index_value : int, str, list[int], or list[str]
            Input specification to convert.

        Returns
        -------
        int, str, or list
            The converted value(s): index for name input, name for index input.
        """

        if isinstance(index_value,list):
            list_ind = [0]*len(self.name)
            if isinstance(index_value[0],int):
                for i,ind in enumerate(index_value):
                    list_ind[i] = self.name[ind]
            elif isinstance(index_value[0],str):
                for i,ind in enumerate(index_value):
                    list_ind[i]=self.index[ind]
            return list_ind

        else:
            if isinstance(index_value,int):
                return self.name[index_value]
            elif isinstance(index_value,str):
                return self.index[index_value]


class Dataset(LinkedList):
    """
    LinkedList with an optional reference value attached.

    Parameters
    ----------
    input_array : array_like
        Primary data.
    linked_array : array_like or None, optional
        Secondary linked data.
    reference : float or None, optional
        Reference m/z value associated with the dataset.

    Attributes
    ----------
    reference : float or None
        The attached reference value.
    """
    def __new__(cls, input_array, linked_array=None, reference=None):
        """
        Create a Dataset and attach an optional reference value.

        Parameters
        ----------
        input_array : array_like
            Primary data.
        linked_array : array_like or None, optional
            Secondary linked data.
        reference : float or None, optional
            Reference value to attach.
        """
        obj = super().__new__(cls, input_array, linked_array)
        obj.reference = reference
        return obj

    def __array_finalize__(self, obj):
        """
        Propagate `reference` and linked array when creating views.

        Parameters
        ----------
        obj : ndarray or None
            Source object for the view.
        """
        super().__array_finalize__(obj)
        if obj is None: return
        self.reference = getattr(obj, 'reference', None)

    def __setitem__(self, index, value):
        """
        Assign items in the primary array and mirror to the linked array.
        """
        super().__setitem__(index, value)

class File:
    """
    Thin wrapper around an HDF5 file to read datasets and their headers.

    Parameters
    ----------
    file_name : str or Path
        Path to the HDF5 file.

    Attributes
    ----------
    real_path : Path
        Resolved path to the file.
    """
    def __init__(self, file_name):
        """
        Initialize the file wrapper.

        Parameters
        ----------
        file_name : str or Path
            Path to the HDF5 file.
        """
        self.real_path = Path(file_name)

    def exist(self):
        """
        Check whether the file exists.

        Returns
        -------
        bool
            True if the file exists.
        """
        return self.real_path.exists()

    def read(self, dataset):
        """
        Read a dataset and its column headers from the HDF5 file.

        Parameters
        ----------
        dataset : str
            HDF5 path to the dataset to read.

        Returns
        -------
        tuple
            A tuple ``(data, attr)`` where ``data`` is a NumPy array and
            ``attr`` is a list/array of column headers. Returns None on error.

        """
        try:
            if not self.exist():
                console.print("[bold red]Error: file doesn't exist[/bold red]")
                raise typer.Exit(code=1)

            with h5py.File(self.real_path, 'r') as f:
                if dataset in f:
                    data = f[dataset][:]
                    attr = f[dataset].attrs["Column headers"]

                    if len(attr) != f[dataset].shape[0] and len(attr) == f[dataset].shape[1]:
                        data = data.T
                    elif len(attr) != f[dataset].shape[0] and len(attr) != f[dataset].shape[1]:
                        console.print("[bold red]Error: the number of columns does not match the number of headers[/bold red]")
                        raise typer.Exit(code=1)
                    return data, attr
        except Exception as err:
            console.print(f'Error raised while reading: {err}ex')
            raise typer.Exit(code=1)

def construct_output(p_value, var_raw, var_aln,alpha = 0.05):
    """
        Detect peaks in a KDE curve and return their centers and boundaries.

        Parameters
        ----------
        p_value : np.ndarray
            Array with all p-values
        var_raw : np.ndarray
            Array with dispersion for all peaks in raw data
        var_aln : np.ndarray
            Array with dispersion for all peaks in aln data
        alpha: float
            Confidence level. Default is 0.05.
        Returns
        -------
        result_type: float
            type of result: -1 is negative, +1 is positive, 0 is not statistically significant.
        result_text: str
            exact text of result message which will be displayed.
        """
    s_val, simes_significance = simes(p_value, alpha)
    delta_var_all= np.mean(var_raw - var_aln)
    delta_var_significant = np.mean((var_raw - var_aln)[np.where(p_value<=alpha)])

    if simes_significance: #if significant
        if delta_var_significant>0:
            result_type = 1
            result_text = f"[bold green]Alignment is better than raw data[/bold green]\n[green]Simes = {s_val:.2e}\nVar(total) = {delta_var_all:.2e}\nVar(significant) = {delta_var_significant:.2e}[/green]"
        else:
            result_type = -1
            result_text = f"[bold red]Alignment is worse than raw data[/bold red]\n[red]Simes = {s_val:.2e}\nVar(total) = {delta_var_all:.2e}\nVar(significant) = {delta_var_significant:.2e}[/red]"
    else:
        result_type = 0
        result_text = f"[bold]The differences are not significant[/bold]\nSimes = {s_val:.2e}\nVar(total) = {delta_var_all:.2e}\nVar(significant) = {delta_var_significant:.2e}"
    return result_text, result_type


def peak_picking(X, Y, oversegmentation_filter=None, peak_location=1):
    """
    Detect peaks in a KDE curve and return their centers and boundaries.

    Parameters
    ----------
    X : ndarray
        Monotonic array of X coordinates (e.g., m/z grid).
    Y : ndarray
        Corresponding density/height values.
    oversegmentation_filter : float or None, optional
        Minimal allowed separation between adjacent peaks; when provided, peaks
        closer than this threshold are merged.
    peak_location : float, optional
        Fraction of the peak height to compute a barycentric center; used in
        boundary calculations as a threshold. Default is 1.

    Returns
    -------
    pk_x : ndarray
        Estimated peak centers (X positions). May contain NaNs if a region has
        no samples above the threshold.
    left : ndarray
        Left boundary (valley position) for each peak.
    right : ndarray
        Right boundary (valley position) for each peak.
    """
    n = X.size
    # Robust valley finding
    valley_dots = np.concatenate((np.where(np.diff(Y) != 0)[0], [n-1]))
    loc_min = np.diff(Y[valley_dots])
    loc_min = (np.array([True,*(loc_min < 0)])) & np.array(([*(loc_min > 0),True]))
    left_min = np.concatenate([[-1],valley_dots[:-1]])[loc_min][:-1] + 1
    right_min = valley_dots[loc_min][1:]
    # Compute max and min for every peak
    size = left_min.shape
    val_max = np.empty(size)
    pos_peak = np.empty(size)
    for idx, [lm, rm] in enumerate(zip(left_min, right_min)):
        pp = lm + np.argmax(Y[lm:rm])
        vm = np.max(Y[lm:rm])
        val_max[idx] = vm
        pos_peak[idx] = pp

    # Remove over-segmented peaks
    if oversegmentation_filter:
        while True:
            peak_threshold = val_max * peak_location - math.sqrt(np.finfo(float).eps)
            pk_x = np.empty(left_min.shape)

            for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_threshold)):
                mask = Y[lm:rm] >= th
                if np.sum(mask) == 0:
                    pk_x[idx]=np.nan
                else:
                    pk_x[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
            dpk_x = np.concatenate(([np.inf], np.diff(pk_x), [np.inf]))

            j = np.where((dpk_x[1:-1] <= oversegmentation_filter) & (dpk_x[1:-1] <= dpk_x[:-2]) & (dpk_x[1:-1] < dpk_x[2:]))[0]
            if j.size == 0:
                break
            left_min = np.delete(left_min, j + 1)
            right_min = np.delete(right_min, j)
            val_max[j] = np.maximum(val_max[j], val_max[j + 1])
            val_max = np.delete(val_max, j + 1)
    else:
        peak_threshold = val_max * peak_location - math.sqrt(np.finfo(float).eps)
        pk_x = np.empty(left_min.shape)

        for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_threshold)):
            mask = Y[lm:rm] >= th
            if np.sum(mask) == 0:
                pk_x[idx]=np.nan
            else:
                pk_x[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
    return pk_x, X[left_min], X[right_min]

@njit()
def sort_dots_numba(ds: np.ndarray, left: np.ndarray, right: np.ndarray) -> list:
    """
    Group values into bins defined by paired left/right boundaries.

    Parameters
    ----------
    ds : ndarray
        Values to be grouped.
    left : ndarray
        Left boundaries for each bin.
    right : ndarray
        Right boundaries for each bin.

    Returns
    -------
    flat_grouped_values : ndarray
        Concatenated values from all bins.
    split_indices : ndarray
        Indices to split `flat_grouped_values` into original bins.
    """

    num_intervals = left.size
    num_ds = ds.size

    # 1. Сначала подсчитываем, сколько элементов попадает в каждый интервал
    counts = np.zeros(num_intervals, dtype=np.int64)
    for i in range(num_intervals):
        # np.sum() на булевом массиве работает в nopython режиме
        counts[i] = np.sum((ds >= left[i]) & (ds <= right[i]))

    # 2. Вычисляем индексы для разделения. Это будет [0, count_0, count_0+count_1, ...]
    split_indices = np.zeros(num_intervals + 1, dtype=np.int64)
    split_indices[1:] = np.cumsum(counts)

    # 3. Создаем плоский массив для всех найденных значений
    total_elements = split_indices[-1]
    flat_grouped_values = np.empty(total_elements, dtype=ds.dtype)

    # Создаем копию индексов, чтобы отслеживать, куда вставлять следующий элемент для каждого интервала
    current_indices = split_indices.copy()

    for i in range(num_ds):
        val = ds[i]
        for j in range(num_intervals):
            if left[j] <= val <= right[j]:
                # Находим позицию для вставки и вставляем значение
                idx_to_insert = current_indices[j]
                flat_grouped_values[idx_to_insert] = val
                # Увеличиваем индекс для следующего элемента в этом интервале
                current_indices[j] += 1
                # Поскольку интервалы не пересекаются, можно прервать внутренний цикл
                break

    return flat_grouped_values, split_indices


def sort_dots(ds: np.ndarray, left: np.ndarray, right: np.ndarray) -> list:
    """
    Wrapper above sort_dots_numba to return grouped values as a list.

    Parameters
    ----------
    ds : ndarray
        Values to be grouped.
    left : ndarray
        Left boundaries for each bin.
    right : ndarray
        Right boundaries for each bin.

    Returns
    -------
    list of ndarray
        For each interval [left[i], right[i]], the subset of `ds` within it.
    """
    if len(left) == 0:
        return []

    flat_values, split_ind = sort_dots_numba(ds, left, right)

    ret = np.split(flat_values, split_ind[1:-1])
    return ret


def get_long_and_short(arr_1: np.ndarray, arr_2: np.ndarray) -> (np.ndarray, np.ndarray, bool):
    """
    Return the longer and shorter of two arrays and a flag indicating order.

    Parameters
    ----------
    arr_1, arr_2 : ndarray
        Arrays to compare by first-dimension length.

    Returns
    -------
    long : ndarray
        The longer array.
    short : ndarray
        The shorter array.
    flag : bool
        True if ``arr_1`` is the longer array, else False.
    """
    size1, size2 = arr_1.shape[0], arr_2.shape[0]
    if size1 > size2:
        return arr_1, arr_2, True
    else:
        return arr_2, arr_1, False


def get_opt_strip(arr_long: Dataset, arr_short: Dataset, flag: bool) -> (Dataset, Dataset):
    """
    Align two sequences by shifting the longer to minimize mean squared error.

    Parameters
    ----------
    arr_long : Dataset
        Longer dataset.
    arr_short : Dataset
        Shorter dataset.
    flag : bool
        True if `arr_long` corresponds to the original first argument from
        ``get_long_and_short``.

    Returns
    -------
    Dataset, Dataset
        Sliced/shifted versions with equal length, ordered to match the flag.
    """
    size = arr_short.shape[0]
    long_size = arr_long.shape[0]
    max_shift = long_size - size + 1
    shift_array = np.arange(max_shift)
    score_array = np.zeros(max_shift)
    for i in shift_array:
        fit_score = np.mean((arr_short - arr_long[i:i + size]) ** 2)
        score_array[i] = fit_score
    opt_shift = np.where(score_array == score_array.min())[0][0]
    opt_long = arr_long[opt_shift:opt_shift + size]
    if flag:
        return opt_long, arr_short
    else:
        return arr_short, opt_long


def verify_datasets(data_1: LinkedList, data_2: LinkedList, threshold=1.0) -> (LinkedList, LinkedList):
    """
    Verify and co-trim two sorted datasets so that element-wise differences are bounded.

    The function optionally removes one outlier (by index) and re-aligns to
    satisfy the threshold, returning two arrays of equal length.

    Parameters
    ----------
    data_1, data_2 : LinkedList
        Input datasets to verify.
    threshold : float or str, optional
        Maximum allowed absolute difference between paired values. If
        ``'dist_based'``, the mean difference is used as the threshold.

    Returns
    -------
    LinkedList, LinkedList
        Verified (possibly trimmed) datasets of equal size.
    """
    if data_1.size != data_2.size:
        data1_new, data2_new = get_opt_strip(*get_long_and_short(data_1, data_2))
    else:
        data1_new = data_1
        data2_new = data_2

    dist_array = data1_new - data2_new
    if threshold == 'dist_based':
        threshold = np.mean(dist_array)
    score_fit = np.max(np.abs(dist_array))

    if score_fit > threshold:
        cut_index = np.array([np.where(np.abs(dist_array) >= threshold)]).min()
        if data1_new[cut_index] < data2_new[cut_index]:
            data1_new2 = data1_new.sync_delete(cut_index)
            data2_new2 = data2_new
        else:
            data2_new2 = data2_new.sync_delete(cut_index)
            data1_new2 = data1_new
        return get_opt_strip(*get_long_and_short(data1_new2, data2_new2))
    return data1_new, data2_new


_DATA_RAW = None
_DATA_ALN = None
_IDX = None          # (mz_idx_raw, intensity_idx_raw, spectra_idx_raw, mz_idx_aln, intensity_idx_aln, spectra_idx_aln)
_REF_DEV = None      # (REF, DEV)


def pool_initializer(data_raw, data_aln, idx_tuple, ref, dev):
    """
    Pool initializer: store global references to datasets, indices, and params.

    Parameters
    ----------
    data_raw : ndarray
        Raw dataset array loaded from HDF5.
    data_aln : ndarray
        Aligned dataset array loaded from HDF5.
    idx_tuple : tuple[int, int, int, int, int, int]
        ``(mz_idx_raw, intensity_idx_raw, spectra_idx_raw, mz_idx_aln,
        intensity_idx_aln, spectra_idx_aln)`` indices into the datasets.
    ref : float
        Reference m/z value for ``find_ref``.
    dev : float
        Allowed deviation (±) around ``ref`` for reference search.

    Notes
    -----
    Stores the arguments into module-level globals (``_DATA_RAW``, ``_DATA_ALN``,
    ``_IDX``, ``_REF_DEV``) to avoid repeated pickling and argument passing to
    worker processes.
    """
    global _DATA_RAW, _DATA_ALN, _IDX, _REF_DEV
    _DATA_RAW = data_raw
    _DATA_ALN = data_aln
    _IDX = idx_tuple
    _REF_DEV = (ref, dev)


def process_spectrum(task):
    """
    Process a single spectrum task and return datasets for raw and aligned.

    Parameters
    ----------
    task : tuple[int, int, int, int, int]
        ``(spec_id, r0, r1, a0, a1)`` where ``[r0:r1]`` and ``[a0:a1]`` are
        inclusive slices for raw and aligned blocks belonging to ``spec_id``.

    Returns
    -------
    tuple
        ``(spec_id, arr_raw, arr_aln)`` where ``arr_raw`` and ``arr_aln`` are
        NumPy arrays representing ``Dataset`` instances for the spectrum.
    """
    spec_id, r0, r1, a0, a1 = task
    mz_idx_raw, intensity_idx_raw, _s_idx_r, mz_idx_aln, intensity_idx_aln, _s_idx_a = _IDX
    REF, DEV = _REF_DEV

    # извлечь и отсортировать по m/z
    data_raw_unsorted = _DATA_RAW[[mz_idx_raw, intensity_idx_raw], r0:r1 + 1]
    data_aln_unsorted = _DATA_ALN[[mz_idx_aln, intensity_idx_aln], a0:a1 + 1]

    order_raw = np.argsort(data_raw_unsorted[0])
    order_aln = np.argsort(data_aln_unsorted[0])
    data_raw = data_raw_unsorted[:, order_raw]
    data_aln = data_aln_unsorted[:, order_aln]

    data_raw_mz, data_aln_mz = data_raw[0], data_aln[0]
    data_raw_int, data_aln_int = data_raw[1], data_aln[1]

    data_raw_linked = Dataset(data_raw_mz, data_raw_int)
    data_aln_linked = Dataset(data_aln_mz, data_aln_int)

    checked_raw, checked_aln = verify_datasets(data_raw_linked, data_aln_linked, 1)

    _, ref_aln = find_ref(checked_aln, REF, DEV)
    _, ref_raw = find_ref(checked_raw, REF, DEV)

    checked_raw.reference = ref_raw
    checked_aln.reference = ref_aln

    return spec_id, np.array(checked_raw), np.array(checked_aln)


def find_ref(dataset: Dataset, approx_mz: float, deviation=1.0) -> [float, float]:
    """
    Locate a reference peak near an approximate m/z within a deviation window.

    Parameters
    ----------
    dataset : Dataset
        Sorted m/z values (primary) with intensities as linked data.
    approx_mz : float
        Approximate m/z for the reference.
    deviation : float, optional
        Allowed deviation around `approx_mz` for candidate search.

    Returns
    -------
    tuple
        Pair ``(index, mz)`` of the selected reference peak.
    """
    condition_1 = approx_mz - deviation <= dataset
    condition_2 = approx_mz + deviation >= dataset

    where_construct = np.where(condition_1 & condition_2)
    if where_construct[0].size:
        ref_index = where_construct[0][np.argmax(dataset.linked_array[where_construct])]
    else:
        ref_index = np.argmin(np.abs(dataset - approx_mz))

    return ref_index, dataset[ref_index]


def read_dataset( dataset_raw: np.ndarray, attrs_raw: list, dataset_aln: np.ndarray,
                 attrs_aln: list, REF, DEV, limit=None, processes: int = 0):
    """
    Prepare per-spectrum datasets and emit progress for the UI, with optional
    sequential or parallel execution (multiprocessing.Pool).

    Overview
    --------
    - Resolve indices of required columns by headers (m/z and intensity).
    - Build contiguous segments for each spectrum id based on the spectra index.
    - Create tasks only for spectrum ids present in both raw and aligned inputs.
    - For each task: slice the subarrays, sort by m/z, verify alignment
      (``verify_datasets``), find a reference peak around ``REF`` within ``DEV``
      (``find_ref``), and store the result as a ``Dataset`` with a ``reference``.

    Modes
    -----
    - Sequential (``processes <= 0``): runs in the main thread, preserving
      existing variable names and logic.
    - Parallel (``processes > 0``): uses ``multiprocessing.Pool`` with an
      initializer (``pool_initializer``) and worker (``process_spectrum``).
      Tasks are processed in parallel; results may arrive unordered and are
      placed by ``spec_id``.

    Parameters
    ----------
    dataset_raw, dataset_aln : ndarray
        Raw and aligned datasets read from HDF5.
    attrs_raw, attrs_aln : list of str
        Column headers for the respective datasets.
    REF : float
        Reference m/z seed.
    DEV : float
        Acceptable deviation (±) around ``REF`` for reference search.
    limit : int or None, optional
        Optional limit on the number of spectra to process (debugging).
    processes : int, optional
        Number of processes for ``multiprocessing.Pool``. ``<= 0`` means
        sequential mode. Default is 0.

    Returns
    -------
    ndarray
        Array of shape ``(2, N)`` with ``dtype=Dataset``, where ``N`` is the
        number of processed spectra. ``dataset_list[0, spec_id]`` corresponds to
        the raw dataset; ``dataset_list[1, spec_id]`` to the aligned dataset.

    Notes
    -----
    - Only spectrum ids present in both raw and aligned datasets are processed.
    - The progress bar is initialized based on the number of tasks (common ids).
    - In parallel mode, result arrival order is not guaranteed.
    """

    row_raw = DatasetHeaders(attrs_raw)
    row_aln = DatasetHeaders(attrs_aln)

    int_type = None

    if "mz" in attrs_raw:
        mz_type = "mz"
    else:
        mz_type = "peak"
    if "Intensity" not in attrs_raw:
        for column in ["Area","SNR"]:
            if column in attrs_raw:
                int_type = column
                break
    else:
        int_type = "Intensity"

    if int_type is None:
        console.print('[bold red]Intensity type not stated in attrs_raw, check file input[/bold red]')
        raise typer.Exit(code=1)

    index_row_raw = dataset_raw[row_raw("spectra_ind")]
    index_row_aln = dataset_aln[row_aln("spectra_ind")]

    start_index, end_index = int(min(index_row_raw)), int(max(index_row_raw))
    if limit is not None:
        if start_index + limit <= end_index:
            end_index = start_index + limit

    set_num = end_index - start_index + 1

    dataset_list = np.empty((2, set_num), dtype=Dataset)

    # предрасчёт сегментов по каждому индексу спектра
    segments_raw = build_segments(index_row_raw)
    segments_aln = build_segments(index_row_aln)

    # список задач только для тех индексов, которые есть и в raw, и в aln
    tasks = []
    for spec_id in range(start_index, end_index + 1):
        if spec_id in segments_raw and spec_id in segments_aln:
            r0, r1 = segments_raw[spec_id]
            a0, a1 = segments_aln[spec_id]
            tasks.append((spec_id, r0, r1, a0, a1))

    task_amount = len(tasks)
    # последовательная ветка — сохранить существующую логику имен/переменных
    if processes <= 0:

        for spec_n, (spec_id, r0, r1, a0, a1) in enumerate(tqdm.tqdm(tasks)):
            index_raw, index_aln = np.where(index_row_raw == spec_id)[0], np.where(index_row_aln == spec_id)[0]
            data_raw_unsorted = dataset_raw[row_raw([mz_type,int_type]), index_raw[0]:index_raw[-1] + 1]
            data_aln_unsorted = dataset_aln[row_aln([mz_type,int_type]), index_aln[0]:index_aln[-1] + 1]

            data_raw = data_raw_unsorted[:, np.argsort(data_raw_unsorted, axis=1)[0]]
            data_aln = data_aln_unsorted[:, np.argsort(data_aln_unsorted, axis=1)[0]]

            data_raw_mz, data_aln_mz = data_raw[0], data_aln[0]
            data_raw_int, data_aln_int = data_raw[1], data_aln[1]

            data_raw_linked = Dataset(data_raw_mz, data_raw_int)
            data_aln_linked = Dataset(data_aln_mz, data_aln_int)

            checked_raw, checked_aln = verify_datasets(data_raw_linked, data_aln_linked, 1)

            _, ref_aln = find_ref(checked_aln, REF, DEV)
            _, ref_raw = find_ref(checked_raw, REF, DEV)

            checked_raw.reference = ref_raw
            checked_aln.reference = ref_aln

            dataset_list[0, spec_id] = np.array(checked_raw)
            dataset_list[1, spec_id] = np.array(checked_aln)

        return dataset_list

    # параллельная ветка — Pool с минимальными изменениями
    mz_idx_raw = row_raw(mz_type)
    intensity_idx_raw = row_raw(int_type)
    spectra_idx_raw = row_raw("spectra_ind")

    mz_idx_aln = row_aln(mz_type)
    intensity_idx_aln = row_aln(int_type)
    spectra_idx_aln = row_aln("spectra_ind")

    init_args = (
        dataset_raw,
        dataset_aln,
        (mz_idx_raw, intensity_idx_raw, spectra_idx_raw,
         mz_idx_aln, intensity_idx_aln, spectra_idx_aln),
        REF, DEV,
    )

    #multiprocessing.util.FINALIZE_MAX_DELAY = 10
    with Pool(processes=processes, initializer=pool_initializer, initargs=init_args) as pool:
        iterator_pool = pool.imap_unordered(process_spectrum, tasks)

        for spec_n, (spec_id, arr_raw, arr_aln) in enumerate(tqdm.tqdm(iterator_pool,total=task_amount)):
            dataset_list[0, spec_id] = arr_raw
            dataset_list[1, spec_id] = arr_aln


    return dataset_list


def build_segments(spectra_index_row: np.ndarray) -> dict[int, tuple[int, int]]:
    """
    Build contiguous [start, end] slices for each value of ``spectra_ind``.

    Parameters
    ----------
    spectra_index_row : ndarray
        1-D array of spectrum identifiers, typically the ``spectra_ind`` row
        from an HDF5 dataset.

    Returns
    -------
    dict[int, tuple[int, int]]
        Mapping from spectrum id to an inclusive ``(start, end)`` slice within
        ``spectra_index_row`` covering its contiguous block.
    """
    segments: dict[int, tuple[int, int]] = {}
    if spectra_index_row.size == 0:
        return segments
    change_pos = np.where(np.diff(spectra_index_row) != 0)[0]
    starts = np.concatenate(([0], change_pos + 1))
    ends = np.concatenate((change_pos, [spectra_index_row.size - 1]))
    ids = spectra_index_row[ends]
    for s, e, spec_id in zip(starts, ends, ids):
        segments[int(spec_id)] = (int(s), int(e))
    return segments


def prepare_array(distances):
    """
    Concatenate per-peak distances and build a 2-row sorted view with indices.

    Parameters
    ----------
    distances : ndarray or Sequence
        Pair or sequence of sequences to concatenate and index.

    Returns
    -------
    ndarray
        A 2 x K array with sorted values in row 0 and original indices in row 1.
    """
    concatenated = np.array([np.concatenate(sub) for sub in distances])
    indexes = np.repeat(np.arange(len(distances[0])), [len(sub_arr) for sub_arr in distances[0]])
    pre_sorted = np.vstack((concatenated, indexes))
    result = pre_sorted[:, pre_sorted[0].argsort()]
    return result

def simes(p_value, alpha = 0.05):
    """
    Calculate Simes method p-value for whole spectrum

    Parameters
    ----------
    p_value : ndarray
        p-value array
    alpha : float
        Confidence level. Default is 0.05

    Returns
    -------
    float
        simes value
    bool
        is test statistically significant
    """

    p_vals = np.sort(p_value)
    count = len(p_vals)

    simes_value = np.min(count*p_vals/np.arange(1, count+1))

    simes_significance = simes_value < alpha

    return simes_value, simes_significance

def concover(arr1:np.ndarray,arr2:np.ndarray):
    """
    Compare two distributions using a rank-based variance (Conover-like) test.

    Parameters
    ----------
    arr1, arr2 : ndarray
        Samples from two distributions.

    Returns
    -------
    float
        p-value for the test of equal scale/dispersion.
    """
    dev = lambda data: np.abs(data - np.median(data))

    dev1 = dev(arr1)
    dev2 = dev(arr2)

    all_devs = np.hstack((dev1, dev2))
    ranks = stats.rankdata(all_devs)

    rank1 = ranks[:len(dev1)]
    rank2 = ranks[len(dev1):]

    n = len(dev1)+len(dev2)
    mean_rank = (n+1)/2

    ss_between = (
        len(dev1)*(np.mean(rank1)-mean_rank)**2 +
        len(dev2)*(np.mean(rank2)-mean_rank)**2)

    ss_total = np.sum((ranks-mean_rank)**2)

    t = (n-1)*ss_between/ss_total
    p_value = 1 - stats.chi2.cdf(t, 2)
    return p_value


def stat_params_paired_single(peak_raw, peak_aln, alpha=0.05,return_p = True):
    """
    Compute paired statistics between raw and aligned peak positions.

    For each matched peak, compute mean difference, variances, normality check (to choose acceptable hypothesis tests) and JS-divergence

    Parameters
    ----------
    peak_raw, peak_aln : array_like
        Samples of raw and aligned values for a single peak.
    alpha : float, optional
        Significance level used in tests. Default is 0.05.
    return_p : bool, optional
        If True, function will return exact p-value, otherwise result of comparison with significance level. Default is True.

    Returns
    -------
    tuple
        ``(mean_diff, var_raw, var_aln, js_div, neq_mean, neq_var)``
        where boolean flags are returned as floats (0.0/1.0).
    """
    # вычислить среднее и дисперсии, проверить нормальность, проверить гипотезы о значимости различия средних и дисперсий, возможно посчитать форму распределения
    jsd = lambda p,q: 0.5*(sum(rel_entr(p,q))+sum(rel_entr(q,p)))
    kde_single_peak = lambda dots,n_eval: FFTKDE(bw='silverman',kernel='gaussian').fit(dots).evaluate(n_eval)[1]

    norm_var = lambda data: np.var(data-np.mean(data),ddof=1)
    mean_r, mean_a = np.mean(peak_raw), np.mean(peak_aln)
    var_r, var_a = norm_var(peak_raw), norm_var(peak_aln)

    check_normal_func = lambda data,p: stats.kstest(data,'norm',args=(np.mean(data),np.std(data)))[1]>p
    check_normal = check_normal_func(peak_raw, alpha) & check_normal_func(peak_aln, alpha)
    neq_var_p_val = stats.levene(peak_raw, peak_aln)[1]
    if check_normal:
        #neq_var_p_val = stats.levene(peak_raw, peak_aln)[1]
        neq_mean_p_val = stats.ttest_ind(peak_raw, peak_aln,nan_policy='omit')[1]
    else:
        neq_mean_p_val = stats.mannwhitneyu(peak_raw, peak_aln)[1]
        #neq_var_p_val = stats.fligner(peak_raw, peak_aln)[1]

    if return_p:
        neq_var = neq_var_p_val
        neq_mean = neq_mean_p_val
    else:
        neq_var = neq_var_p_val < alpha
        neq_mean = neq_mean_p_val < alpha
    return np.array([mean_r - mean_a, var_r, var_a,jsd(kde_single_peak(peak_raw,20),kde_single_peak(peak_aln,20)), float(neq_mean), float(neq_var)])


def stat_params_unpaired(ds):
    """
    Compute unpaired per-group statistics for a list of arrays.

    Parameters
    ----------
    ds : Sequence[array_like]
        Sequence of samples (e.g., peak positions per bin).

    Returns
    -------
    ndarray
        Array with columns: variance, dip statistic, dip p-value, skewness,
        kurtosis for each group.
    """
    res = np.array([[np.var(dot), *diptest(dot), stats.skew(dot), stats.kurtosis(dot)] for dot in ds])
    return res


def moving_average(a, n=2):
    """
    Compute the simple moving average over a 1D array.

    Parameters
    ----------
    a : ndarray
        Input array.
    n : int, optional
        Window size. Default is 2.

    Returns
    -------
    ndarray
        Averaged array of length ``len(a) - n + 1``.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def out_criteria(mz, intensity, int_threshold = 0.01, max_diff = 0.4, width_eps = 0.1):
    """
    .. warning::
        Current version of pipeline doesn't use this function

    Identify outlier peak intervals based on intensity and width heuristics.

    Parameters
    ----------
    mz : Dataset or LinkedList
        Peak centers with linked boundaries in `linked_array`.
    intensity : ndarray
        Intensities corresponding to `mz` centers.
    int_threshold : float, optional
        Fraction of the maximum intensity below which points are flagged.
    max_diff : float, optional
        Maximum relative change between consecutive intensities (as |a/b - 1|).
    width_eps : float, optional
        Threshold on normalized width ratio used for flagging.

    Returns
    -------
    ndarray
        Indices of points considered outliers.
    """
    min_int = np.max(intensity) * int_threshold
    first_or = intensity < min_int
    int_criteria = abs(intensity[:-1] / intensity[1:] - 1) < max_diff

    width_criteria = np.diff(mz) / moving_average(np.diff(mz.linked_array).flatten()) <= width_eps
    second_or = np.full(mz.shape, False)
    second_or[1:] = np.logical_and(int_criteria, width_criteria)

    return np.where(np.logical_or(first_or, second_or))[0]


def criteria_apply(arr, intensity):
    """
    .. warning::
        Current version of pipeline doesn't use this function

    Merge narrow neighboring intervals and drop flagged indices.

    Parameters
    ----------
    arr : LinkedList
        Peak centers with linked left/right boundaries.
    intensity : ndarray
        Intensities used to evaluate the criteria.

    Returns
    -------
    LinkedList
        Filtered peaks with adjusted boundaries.
    """
    arr_out = copy.deepcopy(arr)
    indexes = out_criteria(arr, intensity)
    for index in indexes:
        arr_out.linked_array[index-1] = sorted([arr.linked_array[index-1,0],arr.linked_array[index,1]])
    return arr_out.sync_delete(indexes)

def calculate(raw_path:str, ds_raw:str, aln_path:str, ds_aln:str, ref:int, dev:float=1.0, bandwidth:float=0.2, n_dots:int=20_000):
    try:
        features_raw, attrs_raw = File(raw_path).read(ds_raw)
        features_aln, attrs_aln = File(aln_path).read(ds_aln)

        processes = max(1, min((os.cpu_count() or 2) - 1, 3))

        distance_list = read_dataset(features_raw, attrs_raw, features_aln, attrs_aln, ref, dev,
                                     processes=processes, limit = 1000)

        distance_list_prepared = prepare_array(distance_list)
        raw_concat, aln_concat, id_concat = distance_list_prepared

        kde_x_raw, kde_y_raw = FFTKDE(bw=bandwidth, kernel='gaussian').fit(raw_concat).evaluate(n_dots)
        kde_x_aln, kde_y_aln = FFTKDE(bw=bandwidth, kernel='gaussian').fit(aln_concat).evaluate(n_dots)
        center_r, left_r, right_r = peak_picking(kde_x_raw, kde_y_raw)
        center_a, left_a, right_a = peak_picking(kde_x_aln, kde_y_aln)
        # восстановим высоту пиков
        max_center_r, max_center_a = np.interp(center_r, kde_x_raw, kde_y_raw), np.interp(center_a, kde_x_aln,
                                                                                          kde_y_aln)

        borders_r = np.stack((left_r, right_r), axis=1)
        borders_a = np.stack((left_a, right_a), axis=1)
        c_ds_raw = LinkedList(center_r, borders_r)  # .sync_delete(np.where(max_center_r <= epsilon)[0])
        c_ds_aln = LinkedList(center_a, borders_a)  # .sync_delete(np.where(max_center_a <= epsilon)[0])

        c_ds_raw_intensity, c_ds_aln_intensity = np.interp(c_ds_raw, kde_x_raw, kde_y_raw), np.interp(c_ds_aln,
                                                                                                      kde_x_aln,
                                                                                                      kde_y_aln)

        peak_lists_raw = sort_dots(raw_concat, c_ds_raw.linked_array[:, 0], c_ds_raw.linked_array[:, 1])
        peak_lists_aln = sort_dots(aln_concat, c_ds_aln.linked_array[:, 0], c_ds_aln.linked_array[:, 1])

        aln_peak_lists_raw, aln_peak_lists_aln, aln_kde_raw, aln_kde_aln = alignment.munkres(peak_lists_raw,
                                                                                             peak_lists_aln,
                                                                                             c_ds_raw,
                                                                                             c_ds_aln,
                                                                                             c_ds_raw_intensity,
                                                                                             c_ds_aln_intensity,
                                                                                             segmentation_threshold=400)

        s_p = np.array(pd.DataFrame(np.array(
            [stat_params_paired_single(x_el, y_el) for x_el, y_el in zip(aln_peak_lists_raw, aln_peak_lists_aln)],
            dtype='float')).dropna())

        result_text, result_type = construct_output(p_value=s_p[:, -1],
                                                    var_raw=s_p[:, 1],
                                                    var_aln=s_p[:, 2])

        console.print("[green]Processing completed...[/green]")

        should_save: bool = typer.confirm('Save intermediate results and all statistics to the disk?',
                                          default=True)

        flag_aborted_saving = False

        if should_save:
            while True:
                path_save = typer.prompt("Path to the directory:")

                if not os.path.exists(path_save):
                    create = typer.confirm('This directory does not exist. Create it?',
                                          default=True)
                    if create:
                        os.makedirs(path_save)
                        console.print("Directory created...")
                        console.print(f"Saving to the directory {path_save}...")
                        break
                    else:
                        console.print("Saving aborted.")
                        flag_aborted_saving = True
                        break
                else:
                    if os.path.isdir(path_save):
                        console.print(f"Saving to the directory {path_save}...")
                        break
                    else:
                        console.print('[yellow]You should provide path to the directory[/yellow]')
                        continue
            if not flag_aborted_saving:
                try:
                    kde_raw = pd.DataFrame({'mz':kde_x_raw,'int':kde_y_raw})
                    kde_aln = pd.DataFrame({'mz':kde_x_aln,'int':kde_y_aln})
                    peaks = pd.DataFrame({'mz_raw':aln_kde_raw,'mz_aln':aln_kde_aln})

                    unpaired_names = ['variance','dip statistic','dip p-value','skewness','kurtosis']

                    unpaired_raw,unpaired_aln = pd.DataFrame(stat_params_unpaired(peak_lists_raw),columns=unpaired_names),pd.DataFrame(stat_params_unpaired(peak_lists_aln),columns=unpaired_names)
                    unpaired_raw_a, unpaired_aln_a = pd.DataFrame(stat_params_unpaired(aln_peak_lists_raw),columns=unpaired_names), pd.DataFrame(stat_params_unpaired(aln_peak_lists_aln),columns=unpaired_names)


                    paired = pd.DataFrame(s_p,columns=['Distance','Var(raw)','Var(aln)','JSD','neq_mean?','neq_var?'])


                    excel_kde_filename = 'kde_data.xlsx'
                    kde_path = os.path.join(path_save,excel_kde_filename)
                    console.print('KDE data saving...')
                    with pd.ExcelWriter(kde_path,engine='xlsxwriter') as writer:
                        kde_raw.to_excel(writer,sheet_name='KDE(Raw)',index=True)
                        kde_aln.to_excel(writer,sheet_name='KDE(Aln)',index=True)
                        peaks.to_excel(writer,sheet_name='Peak Alignment',index=True)
                    console.print(f'[green]KDE data saved as {kde_path}[/green]')

                    excel_unpaired_filename = 'unpaired_data.xlsx'
                    unp_path = os.path.join(path_save, excel_unpaired_filename)
                    console.print('Statistics(Part 1) saving...')
                    with pd.ExcelWriter(unp_path, engine='xlsxwriter') as writer:
                        unpaired_raw.to_excel(writer,sheet_name='USt(Raw)')
                        unpaired_raw_a.to_excel(writer, sheet_name='USt(Raw,munkres)')
                        unpaired_aln.to_excel(writer, sheet_name='USt(Aln)')
                        unpaired_aln_a.to_excel(writer, sheet_name='USt(Aln,munkres)')

                    console.print(f'[green]Statistics(Part 1) saved as {unp_path}[/green]')


                    excel_paired_filename = 'paired_data.xlsx'
                    p_path = os.path.join(path_save,excel_paired_filename)
                    console.print('Statistics(Part 2) saving...')
                    with pd.ExcelWriter(p_path, engine='xlsxwriter') as writer:
                        paired.to_excel(writer,sheet_name='Stats')

                    console.print(f'[green]Statistics(Part 2) saved as {p_path}[/green]')
                except Exception as error:
                    console.print(f'[bold red]Saving error: {error}[/bold red]')

        console.print(result_text)
    except Exception as error:
        console.print(f'[bold red]Error: {error}[/bold red]')
        raise typer.Exit(code=1)

def get_arguments(config_path:Optional[Path]) -> Dict[str, Any]:

    settings = {}

    if config_path:
        console.print('Loading configuration from {}'.format(config_path))
        if not config_path.is_file():
            console.print(f'[bold red]Error: configuration file does not exist with path:{config_path} [/bold red]')
            raise typer.Exit(code=1)
        with open(config_path, 'r', encoding='utf8') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
                if not isinstance(settings, dict):
                    typer.echo('Error: YAML configuration file must be a dictionary.')
                settings['raw_path'] = yaml_config['FILE_NAMES'][0]
                settings['aln_path'] = yaml_config['FILE_NAMES'][1]
                settings['ref'] = yaml_config['REF']
                settings['dev'] = yaml_config['DEV']
                settings['ds_raw'] = yaml_config['DATASET_R']
                settings['ds_aln'] = yaml_config['DATASET_R']
                settings['bandwidth'] = yaml_config['BW']
                settings['n_dots'] = yaml_config['NDOTS']

            except yaml.YAMLError as e:
                console.print(f'[bold red]YAML parser error:{e} [/bold red]')
                raise typer.Exit(code=1)
    console.print('Configuration loaded.')
    return settings


@app.command()
def run(config: Optional[Path] = typer.Option(None,'-c','--config',help='Config file (if exists)'),
        raw_path: Optional[Path] = typer.Option(None,'-rp','--raw-path',help='Path to the HDF file with raw data'),
        aln_path: Optional[Path] = typer.Option(None,'-ap','--aln-path',help='Path to the HDF file with aligned data'),
        ref: Optional[float] = typer.Option(None,'-r','--ref',help='Reference peak position (now disused)'),
        dev: Optional[float] = typer.Option(None,'-d','--dev',help='The acceptable m/z deviation for peak positions'),
        ds_raw: Optional[str] = typer.Option(None,'-rd','--raw-dataset',help='Path to the dataset inside HDF for raw data'),
        ds_aln: Optional[str] = typer.Option(None,'-ad','--aln-dataset',help='Path to the dataset inside HDF for aligned data'),
        bandwidth: Optional[float] = typer.Option(None,'-bw','--bandwidth',help='Bandwidth parameter for KDE'),
        n_dots: Optional[int] = typer.Option(None,'-n','--ndots',help='The number of points for which the density estimation is calculated')):

    settings = {}
    if config:
        settings.update(get_arguments(config))

    settings.setdefault('bandwidth', 0.02)
    settings.setdefault('n_dots', 10_000)
    settings.setdefault('dev', 1.0)

    required_arguments = ['raw_path','aln_path','ref','ds_raw','ds_aln']

    if raw_path is not None:
        settings['raw_path'] = Path(raw_path)
    if aln_path is not None:
        settings['aln_path'] = Path(aln_path)
    if ref is not None:
        settings['ref'] = float(ref)
    if dev is not None:
        settings['dev'] = float(dev)
    if ds_raw is not None:
        settings['ds_raw'] = ds_raw
    if ds_aln is not None:
        settings['ds_aln'] = ds_aln
    if bandwidth is not None:
        settings['bandwidth'] = float(bandwidth)
    if n_dots is not None:
        settings['n_dots'] = int(n_dots)

    missing_keys = [key for key in required_arguments if settings.get(key) is None]

    if missing_keys:
        error_msg = f"[bold red]Required argument missed:[/bold red] [yellow]{', '.join(missing_keys)} [/yellow]"
        console.print(error_msg)
        raise typer.Exit(code=1)

    calculate(**settings)

if __name__ == "__main__":
    app()
