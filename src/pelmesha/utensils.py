"""Various utilities functions and constants used by the library"""
import time
import warnings
from typing import List, Union, Callable
import pandas as pd
from itertools import pairwise
import numpy as np
import scipy.interpolate as interpolate
from tqdm.auto import tqdm
import os
import math
from multiprocessing import Pool
from sklearn.preprocessing import normalize

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
    mz = pd_table['mz'].to_numpy()
    KD_bandwidth = pd_table['KD_bandwidth'].to_numpy()

    mz_distance_bool = np.diff(mz) > np.max( np.vstack((KD_bandwidth[1:], KD_bandwidth[:-1])), axis = 0)*12
    mz_distance_bool[-1] = True
    borders_idx = np.where(np.concatenate(([True], mz_distance_bool)))[0]
    mz_bins_diffs = np.diff(mz[borders_idx])
    borders_idx[-1] +=1

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
        borders_idx = borders_idx_corrected
    else:
        borders_idx = list(pairwise(borders_idx))
   
    # mz_bins = borders_idx_corrected

    # bathcing pd_table with correction by min number of peaks in batch 
    # mask = np.zeros(len(pd_table4mask), dtype=bool)
    
    if split_peaks_min is not None:
        batched_pd_table = []
        left_idx = borders_idx[0][0]
        for border_idx in borders_idx:
            if border_idx[1] - left_idx + 1 >= split_peaks_min:
                slc = slice(left_idx, border_idx[1])
                batched_pd_table.append((mz[slc], KD_bandwidth[slc]))
                left_idx = border_idx[1]
            # else:
            #     left_idx = border_idx[0]
        if border_idx[1] != left_idx:
            slc = slice(left_idx, border_idx[1])
            batched_pd_table.append((mz[slc], KD_bandwidth[slc]))
    else:
        batched_pd_table = [0]*(len(borders_idx))
        for n_batch, border_idx in enumerate(borders_idx):
            slc = slice(*border_idx)
            batched_pd_table[n_batch] = (mz[slc], KD_bandwidth[slc])
    return batched_pd_table

def _set_KDE_X_plot(plot_start, plot_end, min_dist):
    mz_range = plot_end - plot_start
    num_of_dots = int((mz_range)*5/min_dist)+1

    X_plot = np.linspace(np.float64(plot_start),np.float64(plot_end),num_of_dots)
    diffs = np.diff(X_plot)

    while not np.allclose(np.ones_like(diffs) * diffs[0], diffs):
        warnings.warn(f"X_plot is not uniform between {plot_start} and {plot_end} with num of dots: {num_of_dots} and distances between points {np.unique_values(diffs)}. Reducing number of dots for X_plot by 2 times")
        num_of_dots=int(num_of_dots/2)
        X_plot = np.linspace(plot_start,plot_end,num_of_dots)
        diffs = np.diff(X_plot)
        if num_of_dots<=1:
            raise AssertionError("Cannot get uniform data for KDE. See logs for info")
    return X_plot
#NEW 08072026

def del_hdf5(hdf5_path):
    """
    Delete an HDF5 file from disk if it exists.

    :param hdf5_path: Path to the HDF5 file to delete.
    :type hdf5_path: str
    """
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        print(f"Deleted file {os.path.basename(hdf5_path)} in directory {os.path.dirname(hdf5_path)}")


def mspeaks_KD(X, Y,oversegmentationfilter=None,peaklocation=1, return_pkY = False):
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
    
    # Remove oversegmented peaks
    if oversegmentationfilter:
        while True:
            peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
            pkX = np.empty(left_min.shape)
            
            for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
                mask = Y[lm:rm] >= th
                if not mask.any():
                    pkX[idx]=np.nan
                else:
                    pkX[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
            dpkX = np.concatenate(([np.inf], np.diff(pkX), [np.inf]))
            
            j = np.where((dpkX[1:-1] <= oversegmentationfilter) & (dpkX[1:-1] <= dpkX[:-2]) & (dpkX[1:-1] < dpkX[2:]))[0]
            if j.size == 0:
                break
            left_min = np.delete(left_min, j + 1)
            right_min = np.delete(right_min, j)
            
            val_max[j] = np.maximum(val_max[j], val_max[j + 1])
            val_max = np.delete(val_max, j + 1)
    else:
        peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
        pkX = np.empty(left_min.shape)
        
        for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
            mask = Y[lm:rm] >= th
            if not mask.any():
                pkX[idx]=np.nan
            else:
                pkX[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
    if return_pkY:
        return np.array((pkX, X[left_min], X[right_min], val_max))
    return np.array((pkX, X[left_min], X[right_min]))

def Peak_assignment(peakstable_batch,Xp_batch):
    """    
    Описание
    ----
    Вспомогательная функция к основной `Pgrouping_KD`. Определяет принадлежность значений mz к определённому значению пика  
    """
    if not peakstable_batch.empty:
        if len(Xp_batch) == 3:
            
            for peak, xl, xr in Xp_batch.T:
                bool_mask = (peakstable_batch['mz']>=xl) & (peakstable_batch['mz']<=xr)
                peakstable_batch.loc[bool_mask,"mz"] = peak
        else:

            for peak, xl, xr, density in Xp_batch.T:
                bool_mask = (peakstable_batch['mz']>=xl) & (peakstable_batch['mz']<=xr)
                peakstable_batch.loc[bool_mask,"mz"] = peak
                peakstable_batch.loc[bool_mask,"Density"] = density
    return peakstable_batch

def _align_kde_mz_grids(kde_mz_list: list[np.ndarray]) -> np.ndarray:
    """Построить общую mz-сетку для суммирования KDE от разных источников."""
    mz_min = min(kmz[0] for kmz in kde_mz_list)
    mz_max = max(kmz[-1] for kmz in kde_mz_list)
    # Использовать самую мелкую дискретизацию
    min_step = min(np.quantile(np.diff(kmz), q= 0.25) for kmz in kde_mz_list)
    num_points = int((mz_max - mz_min) / min_step) + 1
    return np.linspace(mz_min, mz_max, num_points)

def _summerize_kde_mz(kde_mz_list: list[np.ndarray],
                      kde_density_list: list[np.ndarray], 
                      normalize_bool: bool = True) -> tuple[np.ndarray, np.ndarray]:
    common_kde_mz = _align_kde_mz_grids(kde_mz_list)
    total_density = np.zeros_like(common_kde_mz)
    for kmz, kden in zip(kde_mz_list, kde_density_list):
        total_density += np.interp(common_kde_mz, kmz, kden, left=0, right=0)
    if normalize_bool:
        total_density = normalize( total_density.reshape(1, -1), norm='l1' ).squeeze()
    return common_kde_mz, total_density

def apply_kde_mzcorrection(peaklist: pd.DataFrame, 
                           kde_mz: np.ndarray, 
                           kde_density: np.ndarray,
                           cpu_num: int = 1) -> pd.DataFrame:
    """Применить коррекцию mz для пиков"""
    Xp_data = mspeaks_KD(kde_mz,kde_density)
    Xp = Xp_data[0]
    Xl = Xp_data[1]
    Xr = Xp_data[2]
    mz_sequence = np.sort(peaklist['mz'].unique())
    mz_num = len(mz_sequence)
    if mz_num < cpu_num*3:
        batches_num = mz_num
    else:
        batches_num = cpu_num*3
    idxmz_batches = list(pairwise(np.linspace(0,mz_sequence.shape[0],batches_num,dtype=int)))
    par_args=[None]*len(idxmz_batches)
    for batch_n,idx_batch in enumerate(idxmz_batches):
        mzb_min = mz_sequence[idx_batch[0]]
        mzb_max = mz_sequence[idx_batch[1]-1]
        idx_l = np.searchsorted(Xl, mzb_min, side='right') - 1
        idx_r = np.searchsorted(Xr, mzb_max, side='left')
        Xl_min = Xl[max(0, idx_l)]
        Xr_max = Xr[min(len(Xr) - 1, idx_r)]
        batch_indexes = (Xp>=Xl_min) & (Xp<=Xr_max)
        par_args[batch_n] = (peaklist.loc[(peaklist['mz'] >= mzb_min) & (peaklist['mz'] <= mzb_max)],
                             Xp_data[:,batch_indexes])
    with Pool(cpu_num) as p:
        grftable = p.starmap(Peak_assignment,par_args)
    grftable=pd.concat(grftable)
    return grftable

def _consesusing_peaks(peaklists: pd.DataFrame):
    """Удалить дублирующиеся пики после корректировки mz 
    с помощью плотности вероятности соследующими правилами:
    SNR - максимальное
    Intensity - максимальное
    Area - сумма
    PextL - минимальное
    PextR - максимальное
    FWHML - минимальное
    FWHMR - максимальное
    Остальные столбцы - первый встречающийся"""

    column_headers = peaklists.columns
    preset_rules = {
        "SNR": "max", "Intensity": "max", "Area": "sum",
        "PextL": "min", "PextR": "max", "FWHML": "min", "FWHMR": "max"
    }
    dict4drop = {col: agg for col, agg in preset_rules.items() if col in column_headers}
    excluded_cols = set(dict4drop.keys()) | {'spectra_ind', 'mz'}
    oth_cols = [col for col in column_headers if col not in excluded_cols]
    
    for col in oth_cols:
        dict4drop[col] = 'first'
    
    # Если у индекса есть имя (MultiIndex или именованный SingleIndex), сохраняем его
    base_index = [name for name in peaklists.index.names if name is not None]
    group_keys = base_index + ['spectra_ind', 'mz']

    result = peaklists.groupby(group_keys, as_index=False).agg(dict4drop)
    
    return result.set_index(base_index)

def _consensus_peaks_summary(feature_series: pd.Series) -> pd.DataFrame:
    """Summarise consensus peaks per (sample, roi, mz) multiplicity.

    Counts how many times each (index combo, mz) group occurs, keeps only
    groups that are actual has multiplicity (count > 1), then pivots to show the
    number of unique m/z values for each multiplicity, per (sample, roi)
    and globally across all samples.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Frame whose index is a (sample, roi, ...) MultiIndex and which has
        a corrected 'mz' column.

    Returns
    -------
    pd.DataFrame
        Pivot with index ('sample', 'roi'), columns = multiplicity ('count'),
        values = number of unique m/z, and column name
        'mz multiplicity'. Empty frame if there are no consensus peaks.
    """
    # Derive index names dynamically so nothing depends on a fixed level count.
    index_names = [name for name in feature_series.index.names if name is not None]
    peak_counts = (
        feature_series
        .groupby([*index_names], observed=True)
        .value_counts()
        .rename("count")
    )
    col_name = feature_series.name    
    # Proper boolean filter on the 'count' column (fixes the old [..., "mz"] mis-filter).
    consesused = peak_counts[peak_counts> 1]
    # Global row: collapse (sample, roi) into a single "all samples" bucket.
    n_consesused = len(consesused)
 
    consesused = consesused.reset_index()
    total_summary = consesused.copy()
    total_summary.loc[:,index_names[0]] = ["Total"] * n_consesused
    total_summary.loc[:,index_names[1]] = [''] * n_consesused 
     
    # return pd.concat([consesused, total_summary], ignore_index=True).pivot_table(
    #     index=["sample", "roi"],
    #     columns="count",
    #     values="mz",
    #     aggfunc="nunique",
    # ).add_suffix(" subs").fillna(0).astype('int')
    return (
    pd.concat([consesused, total_summary], ignore_index=True)
    .groupby(["sample", "roi",'count'])[col_name]
    .nunique()
    .unstack(level="count", fill_value=0)
    .add_suffix(" subs")
    )

def _nunique_summary(feature_series: pd.Series,
                     column_name: str | None = None) -> pd.Series:
    levels = feature_series.index.names
    
    multi_level_nunique = feature_series.groupby(level=levels).nunique()
    
    total_index = pd.MultiIndex.from_tuples([["Total"]+ [""] * (len(levels)-1)], names=levels)
    total_nunique = pd.Series([feature_series.nunique()], index=total_index)
    resulted = pd.concat([multi_level_nunique, total_nunique])
    if column_name:
        resulted.name = column_name
    return resulted
def _frequency_filtration(series: pd.Series,
                            countf: int | float | None = 10,
                            countf_rel: float | None = None) -> pd.Series:
    """Filter peaks that occur with a minimum multiplicity.

    Keeps only the rows whose ``mz`` value appears at least ``countf`` times in
    ``peaklist``. A relative threshold ``countf_rel`` (a fraction of the total
    number of rows) may be supplied instead of, or in addition to, the absolute
    one; when both are given the stricter (larger) threshold wins.

    Parameters
    ----------
    peaklist : pd.DataFrame
        Feature matrix that must contain an ``mz`` column.
    countf : int | float | None, default 10
        Minimum absolute occurrence count. Ignored when ``None``.
    countf_rel : float | None, default None
        Minimum occurrence count expressed as a fraction of ``len(peaklist)``.

    Returns
    -------
    pd.DataFrame
        A filtered copy of ``peaklist`` with only the original columns. The
        input frame is never mutated.

    Raises
    ------
    ValueError
        If ``countf`` is negative, if ``countf_rel`` is not within ``(0, 1]``,
        or if ``peaklist`` has no ``mz`` column.
    """
    if countf is not None and countf < 0:
        raise ValueError(f"countf must be >= 0, got {countf!r}")
    if countf_rel is not None and not 0 < countf_rel <= 1:
        raise ValueError(f"countf_rel must be in (0, 1], got {countf_rel!r}")

    if countf is None and countf_rel is None:
        return pd.Series(True, index=series.index)

    threshold = countf
    if countf_rel is not None:
        rel_threshold = countf_rel * series.size
        threshold = (
            rel_threshold
            if threshold is None
            else max(threshold, rel_threshold)
        )
    
    target_col = series.name if series.name is not None else "value"
    working_series = series if series.name is not None else series.rename(target_col)
    # Count occurrences of each m/z and keep rows meeting the threshold. When counting unqiue peaks - ignoring duplicates
    counts = (
        working_series.reset_index()
        .drop_duplicates()[target_col]
        .value_counts())
    
    peaklist_counts = series.map(counts)
    # return peaklist.loc[
    #     peaklist_with_counts["count"] >= threshold]
    return peaklist_counts >= threshold

def show_df(dataframe, title=""):
    try:
        from IPython.display import HTML
        # В Jupyter выводим красивый жирный заголовок HTML и саму таблицу
        if title:
            display(HTML(f"<h3>{title}</h3>"))
        display(dataframe)
    except (ImportError, NameError):
        # В обычном терминале выводим текст и dataframe
        if title:
            print(f"=== {title} ===")
        print(dataframe)