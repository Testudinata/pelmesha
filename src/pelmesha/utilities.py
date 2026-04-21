"""Various utilities used by the library"""
import time
import warnings
from typing import List, Union
import pandas as pd
from itertools import pairwise
import numpy as np
import scipy.interpolate as interpolate

FWHM_TO_SIGMA_FACTOR = 1 / np.sqrt(8 * np.log(2)) 

def format_time(value: float) -> str:
    """Convert time to nicer format"""
    if value <= 0.005:
        return f"{value * 1000000:.0f}us"
    elif value <= 0.1:
        return f"{value * 1000:.1f}ms"
    elif value > 86400:
        return f"{value / 86400:.2f}day"
    elif value > 1800:
        return f"{value / 3600:.2f}hr"
    elif value > 60:
        return f"{value / 60:.2f}min"
    return f"{value:.2f}s"


def time_loop(t_start: float, n_item: int, n_total: int, as_percentage: bool = True) -> str:
    """Calculate average, remaining and total times

    Parameters
    ----------
    t_start : float
        starting time of the for loop
    n_item : int
        index of the current item - assumes index starts at 0
    n_total : int
        total number of items in the for loop - assumes index starts at 0
    as_percentage : bool, optional
        if 'True', progress will be displayed as percentage rather than the raw value

    Returns
    -------
    timed : str
        loop timing information
    """
    t_tot = time.time() - t_start
    t_avg = t_tot / (n_item + 1)
    t_rem = t_avg * (n_total - n_item + 1)

    # calculate progress
    progress = f"{n_item}/{n_total + 1}"
    if as_percentage:
        progress = f"{(n_item / (n_total + 1)) * 100:.1f}%"

    return f"[Avg: {format_time(t_avg)} | Rem: {format_time(t_rem)} | Tot: {format_time(t_tot)} || {progress}]"


def shift(array, num, fill_value=0):
    """Shift 1d array to new position with 0 padding to prevent wraparound - this function is actually
    quicker than np.roll

    Parameters
    ----------
    array : np.ndarray
        array to be shifted
    num : int
        value by which the array should be shifted
    fill_value : Union[float, int]
        value to fill in the areas where wraparound would have happened
    """
    result = np.empty_like(array)
    if not isinstance(num, int):
        raise ValueError("`num` must be an integer")

    if num > 0:
        result[:num] = fill_value
        result[num:] = array[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = array[-num:]
    else:
        result[:] = array
    return result


def check_xy(x, array):
    """
    Check zvals input

    Parameters
    ----------
    x : np.ndarray
        1D array of separation units (N). The number of elements of xvals must equal the number of elements of
        zvals.shape[1]
    array : np.ndarray
        2D array of intensities that must have common separation units (M x N) where M is the number of vectors
        and N is number of points in the vector

    Returns
    -------
    zvals : np.ndarray
        2D array that should match the dimensions of xvals input
    """
    if x.shape[0] != array.shape[1]:
        if x.shape[0] != array.shape[0]:
            raise ValueError("Dimensions mismatch")
        array = array.T
        warnings.warn("The input array was rotated to match the x-axis input", UserWarning)

    return array


def generate_function(method, x, y):
    """
    Generate interpolation function

    Parameters
    ----------
    method : str
        name of the interpolator
    x : np.array
        1D array of separation units (N)
    y : np.ndarray
        1D array of intensity values (N)

    Returns
    -------
    fcn : scipy interpolator
        interpolation function
    """
    if method == "pchip":
        return interpolate.PchipInterpolator(x, y, extrapolate=False)
    return interpolate.interp1d(x, y, method, bounds_error=False, fill_value=(y[0],y[-1]))


def find_nearest_index(x: np.ndarray, value: Union[float, int]):
    """Find index of nearest value

    Parameters
    ----------
    x : np.array
        input array
    value : number (float, int)
        input value
    Returns
    -------
    index : int
        index of nearest value
    """
    x = np.asarray(x)
    return np.argmin(np.abs(x - value))


def convert_peak_values_to_index(x: np.ndarray, peaks) -> List:
    """Converts non-integer peak values to index value by finding
    the nearest value in the `xvals` array.

    Parameters
    ----------
    x : np.array
        input array
    peaks : list
        list of peaks

    Returns
    -------
    peaks_idx : list
        list of peaks as index
    """
    return [find_nearest_index(x, peak) for peak in peaks]

def split_array_by_num_distance(array, min_distance2split, by_column = 'mz'):
    """
    Split array by num distance
    args:
        array: array to split
        min_distance2split: minimum distance to split
        by_column: column to split by
    returns:
        batched_array: array split by num distance
    """
    if isinstance(array, pd.DataFrame):
        if by_column is None:
            raise ValueError("by_column is None when array is a DataFrame. Please specify by_column.")
        if by_column not in array.columns:
            raise ValueError("by_column is not in array.columns. Please specify a valid column.")
        nums_sequence = array.sort_values(by=by_column)[by_column].unique()
        array4mask = array[by_column].to_numpy()
    elif isinstance(array, np.ndarray):
        nums_sequence = np.unique(np.sort(array))
        array4mask = array

    nums_distance_bool = np.diff(nums_sequence) > min_distance2split
    left_nums = nums_sequence[:-1][nums_distance_bool]
    right_nums = nums_sequence[1:][nums_distance_bool] 
    mz_bins = np.concatenate(([nums_sequence[0] - min_distance2split], (left_nums + right_nums)/2, [nums_sequence[-1] + min_distance2split]))
    batched_array = [0]*(len(mz_bins) - 1)
    for n_batch, mz_bin in enumerate(pairwise(mz_bins)):
        mask = (array4mask > mz_bin[0]) & (array4mask < mz_bin[1])
        batched_array[n_batch] = array[mask]
    
    return batched_array

def chunking_datasets(data_objs, min_distance2split, index_names = None):
    """
    Chunking aln noaln data
    args:
        data_obj: data object
    returns:
        chunked_data_obj: chunked data object
    """
    if min_distance2split is None:
        raise ValueError("min_distance2split is None. Please specify min_distance2split.")
    elif isinstance(min_distance2split, list):
        min_distance2split = max(min_distance2split)
        Warning(f"min_distance2split is a list. Max value {min_distance2split} will be used.")
        
    if isinstance(data_objs, dict):
        index_names = index_names or data_objs.keys()
    merged_data_obj =  pd.concat(data_objs, index_names)
    return split_array_by_num_distance(merged_data_obj, min_distance2split = min_distance2split)

def split_pdtable_by_medfwhm(pd_table, split_peaks_min = 25, split_mz_min = 10): # TODO del if func lower is actual
    """
    Split pd.DataFrame table by median FWHM
    args:
        array: array to split
        min_distance2split: minimum distance to split
        by_column: column to split by
    returns:
        batched_array: array split by num distance
    """
    
    mz_sequence = pd_table.sort_values(by="mz")['mz'].unique()
    pd_table4mask = pd_table['mz'].to_numpy()

    min_distance2split = np.median(pd_table['FWHM']) * FWHM_TO_SIGMA_FACTOR * 12 # min_distance2split is 12 * median(sigma)

    # getting split borders
    mz_distance_bool = np.diff(mz_sequence) > min_distance2split
    left_mz = mz_sequence[:-1][mz_distance_bool]
    right_mz = mz_sequence[1:][mz_distance_bool] 
    mz_bins = np.concatenate(([mz_sequence[0] - min_distance2split], (left_mz + right_mz)/2, [mz_sequence[-1] + min_distance2split]))
    mz_bins_diffs = np.diff(mz_bins)
   
    # mz_bins correction by min split load 
    if split_mz_min is not None:
        # split_mz_min = min_distance2split * split_mz_min # split_mz_min is min_distance2split * split_mz_minl
        current_sum = 0
        mz_bins_corrected = [mz_bins[0]]
        for n, mz_bin_diff in enumerate(mz_bins_diffs,start=1):
            if (current_sum + mz_bin_diff) > split_mz_min:
                mz_bins_corrected.append(mz_bins[n])
                current_sum = 0
            else:
                current_sum += mz_bin_diff
        if mz_bins_corrected[-1] < mz_bins[-1]:
            mz_bins_corrected.append(mz_bins[-1])

    mz_bins = mz_bins_corrected

    # bathcing pd_table with correction by min number of peaks in batch 
    mask = np.zeros(len(pd_table4mask), dtype=bool)
    batched_pd_table = []
    if split_peaks_min is not None:
        for mz_bin in pairwise(mz_bins):
            mask = mask | (pd_table4mask > mz_bin[0]) & (pd_table4mask < mz_bin[1])
            if sum(mask) > split_peaks_min:
                batched_pd_table.append(pd_table[mask])
                mask.fill(0)
        if mask.any():
            batched_pd_table.append(pd_table[mask])
    else:
        batched_pd_table = [0]*(len(mz_bins) - 1)
        for n_batch, mz_bin in enumerate(pairwise(mz_bins)):
            mask = (pd_table4mask > mz_bin[0]) & (pd_table4mask < mz_bin[1])
            batched_pd_table[n_batch] = pd_table[mask]
    return batched_pd_table

def split_pdtable_by_peaks_gap(pd_table, split_peaks_min = 25, split_mz_min = 10):
    """
    Split pd.DataFrame table by median FWHM
    args:
        array: array to split
        min_distance2split: minimum distance to split
        by_column: column to split by
    returns:
        batched_array: array split by num distance
    """
    
    pd_table.sort_values(by = "mz", inplace = True, ignore_index = True)
    mz_distance_bool = (np.diff(pd_table['mz'].to_numpy()) - np.max( np.hstack(pd_table.loc[1:,'KD_bandwidth'].to_numpy(), pd_table.loc[:-1,'KD_bandwidth'].to_numpy()), axis = 0)*12) > 0

    mz_distance_bool[-1] = True
    borders_idx = np.where(np.concatenate(([True], mz_distance_bool)))[0]
    mz_bins_diffs = np.diff(pd_table.loc[borders_idx,'mz'])
    # borders_idx = pairwise(borders_idx)
    
    # !!!!!!!!!!!!!!! Тут остановился. Нужно решить как сплитить таблицу. Либо по новым индексам после сортировки (pairwise(левый и правый индекс сегмента)) и тут же найти размер сегмента по mz, либо вернуться к бинам
    # mz_bins = np.concatenate(([mz_sequence[0] - min_distance2split], (left_mz + right_mz)/2, [mz_sequence[-1] + min_distance2split]))
    # mz_bins_diffs = np.diff(mz_bins)
   
    ## mz_bins correction by min split load 
    # if split_mz_min is not None:
    #     # split_mz_min = min_distance2split * split_mz_min # split_mz_min is min_distance2split * split_mz_minl
    #     current_sum = 0
    #     mz_bins_corrected = [mz_bins[0]]
    #     for n, mz_bin_diff in enumerate(mz_bins_diffs,start=1):
    #         if (current_sum + mz_bin_diff) > split_mz_min:
    #             mz_bins_corrected.append(mz_bins[n])
    #             current_sum = 0
    #         else:
    #             current_sum += mz_bin_diff
    #     if mz_bins_corrected[-1] < mz_bins[-1]:
    #         mz_bins_corrected.append(mz_bins[-1])

    # mz_bins = mz_bins_corrected
    if split_mz_min is not None:
        current_sum = 0
        borders_idx_corrected = []
        left_idx = borders_idx[0]
        for n, mz_bin_diff in enumerate(mz_bins_diffs,start=1):
            if (current_sum + mz_bin_diff) > split_mz_min:
                borders_idx_corrected.append((left_idx, borders_idx[n]))
                current_sum = 0
                left_idx = borders_idx[n]
            else:
                current_sum += mz_bin_diff
        if borders_idx_corrected[-1][1] < borders_idx[-1]:
            borders_idx_corrected.append((left_idx, borders_idx[-1]))
    else:
        borders_idx = pairwise(borders_idx)
    borders_idx = borders_idx_corrected
    # mz_bins = borders_idx_corrected

    # bathcing pd_table with correction by min number of peaks in batch 
    # mask = np.zeros(len(pd_table4mask), dtype=bool)
    batched_pd_table = []
    if split_peaks_min is not None:
        left_idx = borders_idx[0][0]
        for border_idx in borders_idx:
            if border_idx[1] - left_idx + 1 >= split_peaks_min:
                batched_pd_table.append(pd_table.loc[slice(left_idx, border_idx[1]),:])
            else:
                left_idx = border_idx[0]
        if border_idx[1] - left_idx + 1 < split_peaks_min:
            batched_pd_table.append(pd_table.loc[slice(left_idx, border_idx[1]),:])
    else:
        batched_pd_table = [0]*(len(borders_idx))
        for n_batch, border_idx in enumerate(borders_idx):
            batched_pd_table[n_batch] = pd_table.loc[slice(*border_idx),:]
    return batched_pd_table