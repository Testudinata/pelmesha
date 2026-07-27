import numpy as np
from h5py import File
import math
import os
import warnings
from pybaselines import Baseline
from pelmesha.dough import DatasetHeaders, Indexator, SliceIndexator
from scipy.signal import savgol_filter
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from pelmesha import Configs, PipelineConfigurator, DataSource, PreparedDataSource

###########################################
#   Base pipeline functions               #
###########################################
def preprocess_configuration_base(
    datasource: "DataSource",
    roi,
    rmeta,
    configs: "Configs | PipelineConfigurator"):
    
    preprocess_configs = configs.get_step_configs("preprocess")
    mz_range = datasource.roi_metadata.loc[roi,'mz_range']
    
    internal_configs = {}
    internal_configs['process'] = {}
    resampled_mz = resample_mz_scale(*mz_range, **preprocess_configs['resample_mz_scale'])
    internal_configs['process']['resampled_mz'] = resampled_mz
    internal_configs['process']['mz_discrete_coeffs'] = datasource.roi_metadata.loc[roi,'discret_coeffs']

    baseline_algo = configs.configs.get('methods', None)
    if baseline_algo:
        baseline_algo = baseline_algo.get('Baseline', None)
        if baseline_algo:
            baseline_algo = next(iter(baseline_algo))
    if baseline_algo:
        if datasource.dcont:
            internal_configs['process']['Baseliner'] = getattr(Baseline(datasource.get_mz(rmeta['idxroi'].ravel()[0]), **configs['Baseline'], assume_sorted = True), baseline_algo)
        else:
            internal_configs['process']['Baseliner'] = baseline_algo

    internal_configs['peakpick'] = {}
    # Configure headers conditionally based on processing flags 
    if configs['peakpicker']["SNR_threshold"] and configs['peakpicker']["Calc_peak_area"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "Area", "SNR",  
                        "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"]
        
    elif configs['peakpicker']["SNR_threshold"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "SNR",  
                        "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"]
    elif configs['peakpicker']["Calc_peak_area"]:
        headers_list = ["spectra_ind", "mz", "Intensity", "Area",  
                        "PextL", "PextR", "FWHML", "FWHMR"]
    else: 
        headers_list = ["spectra_ind", "mz", "Intensity",  
                        "PextL", "PextR", "FWHML", "FWHMR"]
    configs.update({"headers": DatasetHeaders(headers_list)})
    return resampled_mz, headers_list, internal_configs


def process_spectra_base(
    mz: np.ndarray,
    intensity: np.ndarray,
    configs: "Configs | PipelineConfigurator",
    **internal_configs) -> np.ndarray:
    """Полный цикл предобработки масс-спектра:
    сглаживание → коррекция базовой линии → ресемпл →  выравнивание.
    mz - всегда вектор
    intensity - Если данные континуальны - матрица, если нет - вектор
    """
    resampled_mz = internal_configs.get('resampled_mz', None)
    baseline_algo = internal_configs.get('Baseliner', None)
    mz_discrete_coeffs = internal_configs.get('mz_discrete_coeffs', None)

    smooth_configs = configs.get('smoothing', {})
    msalign_configs = configs.get('msalign', {})
    modify_raw_spectrum_configs = configs.get('modify_raw_spectrum', {})

    #premodifying spectrum
    if modify_raw_spectrum_configs.get('add_zero_points_to_peaks') or modify_raw_spectrum_configs.get('mz_segments_to_zero'):
        mz, intensity = modify_raw_spectrum(mz, intensity, mz_discrete_coeffs, **modify_raw_spectrum_configs) 

    #Smoothing step
    if smooth_configs.get('smooth_algo') is not None: 
        intensity = smoothing(intensity, **smooth_configs)
    
    # BaselineCorrection step
    if baseline_algo is not None: 
        if isinstance(baseline_algo, str):
            if baseline_algo == 'asls': #Hack just for setting default params
                intensity = intensity - Baseline(mz, assume_sorted = True  **configs['Baseline']).asls(intensity, **configs['asls'])[0]
            else:
                intensity = intensity - getattr(Baseline(mz, assume_sorted = True,**configs['Baseline']), baseline_algo)(intensity, **configs[baseline_algo])[0]
        else:
            intensity = intensity - baseline_algo(intensity, **configs[baseline_algo.__name__])[0]
    
    # Resampling step
    if resampled_mz is not None:
        intensity = np.interp(resampled_mz, mz, intensity, left=intensity.ravel()[0], right=intensity.ravel()[-1])
        mz = resampled_mz
    
    # Aligning step
    if msalign_configs.get('align_peaks', None) is not None:
        intensity = msalign(mz, intensity, **msalign_configs)
    
    return mz, intensity

def peakpicking_base(
        mz: np.ndarray,
        intensity: np.ndarray,
        idx: int,
        configs: "Configs | PipelineConfigurator",
        **internal_configs) -> np.ndarray:

    mzsize = mz.size
    configs = configs['peakpicker'] #Получаем непосредственно словарь конфигов для функции peakpicker
    return peakpicker(mz, intensity, mzsize, idx, **configs)

def pgrouping_KDE(peaklist, 
                  split_mz_min,
                  split_peaks_min):
    split_params = {'split_mz_min': split_mz_min, 'split_peaks_min': split_peaks_min}
    pass
###########################################
#   Base processing functions             #
###########################################
### Code from __init__.py msalign (https://github.com/lukasz-migas/msalign)
"""Signal calibration and alignment by reference peaks - copy of MSALIGN function from MATLAB bioinformatics library."""
from .align import Aligner
from typing import List
__all__ = ["msalign", "Aligner"]

def msalign(
    x: np.ndarray,
    array: np.ndarray,
    align_peaks: List | None = None, # if None - return array
    align_method: str = "cubic",
    align_width: float = 10,
    align_ratio: float = 2.5,
    align_resolution: int = 100,
    align_iterations: int = 5,
    align_grid_steps: int = 20,
    align_shift_range: List = [-0.95,0.95],
    align_pweights: List = None,
    return_shifts: bool = False,
    align_by_index: bool = False,
    only_shift: bool = False,
):
    """Signal calibration and alignment by reference peaks

    A simplified version of the MSALIGN function found in MATLAB (see references for link)

    This version of the msalign function accepts most of the parameters that MATLAB's function accepts with the
    following exceptions: GroupValue, ShowPlotValue. A number of other parameters is allowed, although they have
    been renamed to comply with PEP8 conventions. The Python version is 8-60 times slower than the MATLAB
    implementation, which is mostly caused by a really slow instantiation of the
    `scipy.interpolate.PchipInterpolator` interpolator. In order to speed things up, I've also included several
    other interpolation methods which are significantly faster and give similar results.

    References
    ----------
    Monchamp, P., Andrade-Cetto, L., Zhang, J.Y., and Henson, R. (2007) Signal Processing Methods for Mass
    Spectrometry. In Systems Bioinformatics: An Engineering Case-Based Approach, G. Alterovitz and M.F. Ramoni, eds.
    Artech House Publishers).
    MSALIGN: https://nl.mathworks.com/help/bioinfo/ref/msalign.html

    Parameters
    ----------
    x : np.ndarray
        1D array of separation units (N). The number of elements of xvals must equal the number of elements of
        zvals.shape[1]
    array : np.ndarray
        2D array of intensities that must have common separation units (M x N) where M is the number of vectors
        and N is number of points in the vector
    align_peaks : list / None
        list of reference peaks that must be found in the xvals vector. Default: None. If None - return array as is
    align_method : str
        interpolation method. Default: 'cubic'. MATLAB version uses 'pchip' which is significantly slower in Python
    align_pweights: list (optional)
        list of weights associated with the list of peaks. Must be the same length as list of peaks
    align_width : float (optional)
        width of the gaussian peak in separation units. Default: 10
    align_ratio : float (optional)
        scaling value that determines the size of the window around every alignment peak. The synthetic signal is
        compared to the input signal within these regions. Default: 2.5
    align_resolution : int (optional)
        Default: 100
    align_iterations : int (optional)
        number of iterations. Increasing this value will (slightly) slow down the function but will improve
        performance. Default: 5
    align_grid_steps : int (optional)
        number of steps to be used in the grid search. Default: 20
    align_shift_range : list or numpy.ndarray, optional
    The maximum allowed shift values in the m/z axis. If `align_by_index` or `only_shift` is set to True, 
    these shifts are measured in data points (indices) instead. Default: [-0.95, 0.95].
    only_shift : bool
        determines if signal should be shifted (True) or rescaled (False). Default: True
    return_shifts : bool
        decide whether shift parameter `shift_opt` should also be returned. Default: False
    align_by_index : bool
        decide whether alignment should be done based on index rather than `xvals` array. Default: False
    """
    if align_peaks is None:
        return array
    
    aligner = Aligner(
        x,
        array,
        align_peaks,
        method=align_method,
        width=align_width,
        ratio=align_ratio,
        resolution=align_resolution,
        iterations=align_iterations,
        grid_steps=align_grid_steps,
        shift_range=align_shift_range,
        weights=align_pweights,
        return_shifts=return_shifts,
        align_by_index=align_by_index,
        only_shift=only_shift,
    )
    aligner.run()
    return aligner.apply()


msalign.__doc__ = Aligner.__doc__

def resample_mz_scale(mz_min: float, 
                      mz_max: float,
                      resample_mz_step: float = None,
                      resample_num_points: int = None):
    if resample_mz_step is not None and resample_num_points is not None:
        raise ValueError("Укажите что-то одно: либо resample_mz_step, либо resample_num_points.")
    if resample_mz_step is not None:
        resample_num_points = int(np.round((mz_max - mz_min)/resample_mz_step)) + 1
    if resample_num_points is not None:
        return np.linspace(mz_min, mz_max, resample_num_points)
    else:
        return
    
### Smoothing
def smoothing(y: np.ndarray, 
              smooth_algo: str = None, 
              smooth_window: int = 7, 
              smooth_cycles: int = 1) -> np.ndarray:
    if len(y) == 0:
        return np.array([])
    
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    if smooth_algo == 'MA':
        return _movaver(y, smooth_window, smooth_cycles)
    elif smooth_algo == 'GA':
        return _gaussian_filter(y, smooth_window, smooth_cycles)
    elif smooth_algo == 'SG':
        return _savgol(y, smooth_window, smooth_cycles)
    else:
        print("Smoothing algorithm not recognized")
        return y

def _movaver(y, window, cycles):
    if window < 3 or len(y) < window:
        return y.copy()
    
    pad_size = window // 2
    kernel = np.ones(window) / window
    
    for _ in range(cycles):
        padded = np.pad(y, (pad_size, pad_size), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        y = smoothed
    
    return y

def _gaussian_filter(y, window, cycles, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    
    if window < 3 or len(y) < window:
        return y.copy()
    
    for _ in range(cycles):
        y = gaussian_filter1d(y, sigma=sigma, mode='mirror')
    
    return y

def _savgol(y, window, cycles, order=3):
    if window <= order or len(y) < window:
        return y.copy()
    
    for _ in range(cycles):
        y = savgol_filter(y, window_length=window, polyorder=order, mode='mirror')
    
    return y

def peakpicker(mz,
              intens,
              xsize, 
              spectra_ind,
              fwhhfilter = None,
              oversegmentationfilter = None,
              heightfilter = None,
              rel_heightfilter = None,
              peaklocation = 1,
              noise_func = np.std,
              noise_est_iterations = 3, 
              SNR_threshold = 3, 
              Calc_peak_area = True, 
              headers = ["spectra_ind", "mz", "Intensity", "Area", "SNR", "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"]
              ) -> np.ndarray:
    """
    Общее описание
    ----
    Функция для получения характеристик пиков спектра. Если в каком-то параметре стоит `None`, то функция не будет производить фильтрацию или расчёты и не будет добавлять свойство пиков на выходе, что может экономить время и память.
    :param mz: mz
    :param intens: Intensity
    :param valley_dots: numpy array того какие точки спектра явлюятся наклонными. На входе подаётся как `np.where(np.diff(intens) != 0)[0]` или как строка этого результата, если intens матрица. Этот параметр нужен скорее для универсальности функции и оптимизации кол-ва расчётов/обращений. 
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `None`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `None`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `None`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `None`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `None`
    :param noise_func: функция оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `np.std`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    
    :type mz: `np.array`
    :type intens: `np.array`
    :type valley_dots: `np.array`
    :type oversegmentationfilter: `float`
    :type fwhhfilter: `float`
    :type heightfilter: `float`
    :type peaklocation: `float` and =<1
    :type rel_heightfilter: `float`
    :type SNR_threshold: `float`
    :type noise_func: function
    :type noise_est_iterations: `int`

    :return: peaklist with peak properties
    :rtype: `np.array`
    """
    #TODO: сделать гибридфильтра по SNR быстрого и медленного варианта: последний или послдние два цикла - шум определяется по окну вокруг пика, 
    # возможно окно - это std точек справа и слева, которые чисто шумовые (точки пика исключены), а кол-во точек и есть размер окна (не по m/z). Постараться сделать с numba
    props={}
    # Robust valley finding
    valley_dots = np.where(np.diff(intens) !=0)[0] 
    valley_dots = np.concatenate((valley_dots, [xsize-1]))    
    loc_min = np.diff(intens[valley_dots])
    loc_min = (np.array([True,*(loc_min < 0)])) & np.array(([*(loc_min > 0),True]))
    left_min = np.concatenate([[-1],valley_dots[:-1]])[loc_min][:-1] + 1
    right_min = valley_dots[loc_min][1:]

    # Compute max for every peak
    size = left_min.shape
    val_max = np.empty(size)
    pos_peak = np.empty(size,dtype=int)
    for idx, [lm, rm] in enumerate(zip(left_min, right_min)):
        val_max[idx] = np.max(intens[lm:rm])
        pos_peak[idx] = lm + np.argmax(intens[lm:rm])
    # Remove peaks below the height, relative height
    if heightfilter and rel_heightfilter:
        k = (val_max >= heightfilter) & (val_max/max(intens) >= rel_heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]
        pos_peak = pos_peak[k]
    elif heightfilter:
        k = (val_max >= heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]
        pos_peak = pos_peak[k]
    elif rel_heightfilter:
        k = (val_max/max(intens) >= rel_heightfilter)
        val_max = val_max[k]
        left_min = left_min[k]
        right_min = right_min[k]
        pos_peak = pos_peak[k]

    # Remove peaks below the SNR thresholds
    if SNR_threshold:
        noise_points = np.array([True]*xsize) # Zero iteration

        for it in range(noise_est_iterations):
            for idx in np.where(((val_max-np.mean(intens[noise_points]))/noise_func(intens[noise_points])>=SNR_threshold))[0]: #По сути тут расчёт z-score в чистом виде TODO: оценить скорость рассчётов моего варианта и scipy.stats.zscore
                sl = slice(left_min[idx],right_min[idx]+1)
                noise_points[sl] = False
        
        props["Noise"] = noise_func(intens[noise_points])
        props["Mean noise"]= np.mean(intens[noise_points])
        k = (val_max-props["Mean noise"])/props["Noise"]>=SNR_threshold
        
        val_max=val_max[k]
        left_min=left_min[k]
        right_min=right_min[k]
        pos_peak = pos_peak[k]
    # Compute FWHH for every peak
    size = left_min.shape
    props["FWHML"] = np.empty(size)
    props["FWHMR"]  = np.empty(size)
    for idx, [lm, rm, vm, pp] in enumerate(zip(left_min, right_min, val_max, pos_peak)):
        props["FWHML"][idx] = np.interp(vm/2,intens[lm:pp+1], mz[lm:pp+1])
        props["FWHMR"][idx] = np.interp(vm/2,intens[pp:rm+1][::-1], mz[pp:rm+1][::-1])
    # Remove peaks with FWHH thresholds
    if fwhhfilter:
        if isinstance(fwhhfilter,tuple):
            k = ((props["FWHMR"] - props["FWHML"]) >= fwhhfilter[0]) & ((props["FWHMR"] - props["FWHML"]) <= fwhhfilter[1])
        else:
            k = (props["FWHMR"] - props["FWHML"]) >= fwhhfilter
        val_max = val_max[k]
        props["FWHML"] = props["FWHML"][k]
        props["FWHMR"] = props["FWHMR"][k]
        left_min = left_min[k]
        right_min = right_min[k]
        pos_peak = pos_peak[k]
    # Remove oversegmented peaks
    if oversegmentationfilter:
        if isinstance(oversegmentationfilter,str):
            oversegmentationfilter = np.median(props["FWHMR"]-props["FWHML"])
        while True:
            peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
            pkmz = np.empty(left_min.shape)
            
            for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
                mask = intens[lm:rm] >= th
                if not mask.any():
                    pkmz[idx]=np.nan
                else:
                    pkmz[idx] = np.sum(intens[lm:rm][mask] * mz[lm:rm][mask]) / np.sum(intens[lm:rm][mask])
            dpkmz = np.concatenate(([np.inf], np.diff(pkmz), [np.inf]))
            
            j = np.where((dpkmz[1:-1] <= oversegmentationfilter) & (dpkmz[1:-1] <= dpkmz[:-2]) & (dpkmz[1:-1] < dpkmz[2:]))[0]
            if j.size == 0:
                break
            left_min = np.delete(left_min, j + 1)
            right_min = np.delete(right_min, j)
            props["FWHML"] = np.delete(props["FWHML"], j + 1)
            props["FWHMR"] = np.delete(props["FWHMR"], j)
            ## New TODO: Test
            stack_j = np.vstack((j,j+1))
            pos_peak_oversegmentation = pos_peak[stack_j]
            val_max_oversegmentation = val_max[stack_j]
            max_idx = np.argmax(val_max_oversegmentation, axis = 0)
            range_j = np.arange(len(j))
            val_max[j] = val_max_oversegmentation[max_idx, range_j]
            pos_peak[j] = pos_peak_oversegmentation[max_idx, range_j]
            val_max = np.delete(val_max, j + 1)
            pos_peak = np.delete(pos_peak, j + 1)

            ## New
            ### OLD
            # val_max[j] = np.maximum(val_max[j], val_max[j + 1])
            # val_max = np.delete(val_max, j + 1)
            ### OLD
    else:
        peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
        pkmz = np.empty(left_min.shape)
        
        for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
            mask = intens[lm:rm] >= th
            if not mask.any():
                pkmz[idx] = np.nan
            else:
                pkmz[idx] = np.sum(intens[lm:rm][mask] * mz[lm:rm][mask]) / np.sum(intens[lm:rm][mask])

    signal_num = len(val_max)
    ## Area calculation
    if Calc_peak_area:
        props["Area"] = np.empty((signal_num,))
        for idx in range(signal_num):
            sl = slice(left_min[idx],right_min[idx]+1,1)
            if min(intens[sl])<0:
                props["Area"][idx] = np.trapezoid(intens[sl] - min(intens[sl]),mz[sl])
            else:
                props["Area"][idx] = np.trapezoid(intens[sl],mz[sl])
    if SNR_threshold:
        props["SNR"] = (val_max - props["Mean noise"])/props["Noise"]
        props["Noise"] = [props["Noise"]]*signal_num
        props["Mean noise"]= [props["Mean noise"]]*signal_num

    left_bool = np.array(pos_peak - left_min) > 2
    right_bool = np.array(right_min - pos_peak) > 2
    if left_bool.any():
        left_min[left_bool] += 1
    if right_bool.any():
        right_min[right_bool] -= 1
    props["PextL"] = mz[left_min] 
    props["PextR"] = mz[right_min]
    return np.column_stack(([spectra_ind]*signal_num,pkmz, val_max, *(props[key] for key in headers[3:])))


def modify_raw_spectrum(mz, 
                        ints, 
                        mz_discret_coeffs,
                        add_zero_points_to_peaks = False,
                        mz_segments_to_zero = None):
    """
    Modify raw spectrum data. Reduce to zero segemnts and add zero points to peaks if their peaks are cropped (especially if peaks consists from 1-2 points)
    """
    if mz_segments_to_zero and mz_discret_coeffs is not None:
        mz, ints = mz_segments_to_zero(mz, ints, mz_discret_coeffs)
    if add_zero_points_to_peaks:
        mz, ints = add_zero_points_to_peaks(mz, ints, mz_discret_coeffs)
    return mz, ints

def add_zero_points_to_peaks(mz, ints, mz_discret_coeffs):
    """
    Add zero points to peaks
    """
    mz_discretion_model = np.poly1d(mz_discret_coeffs)

    diff_mz = np.diff(mz)
    mz_discr = mz_discretion_model(mz[:-1])
    big_gap_bool = diff_mz > 3.5*mz_discr
    small_gap_bool = (diff_mz > 1.75*mz_discr) ^ big_gap_bool
    new_val = {}
    if np.any(big_gap_bool):
        new_val['left'] = mz[np.append(big_gap_bool, [False])] + mz_discr[big_gap_bool]
        new_val['right'] = mz[np.append([False], big_gap_bool)] - mz_discr[big_gap_bool]

    if np.any(small_gap_bool):
        new_val['small'] = mz[np.append(small_gap_bool, [False])] + mz_discr[small_gap_bool]

    if new_val:
        new_val['borders'] = [mz[0]-mz_discr[0], mz[-1]+mz_discr[-1]]
        new_val = np.concatenate(list(new_val.values()), axis=None)
        idx = np.searchsorted(mz, new_val)
        mz = np.insert(mz, idx, new_val)
        ints = np.insert(ints, idx, 0)
        
    # sorting by mz
    idx_sort = np.argsort(mz)
    new_loc_mz = mz[idx_sort]
    new_loc_ints = ints[idx_sort]

    return new_loc_mz, new_loc_ints

def reduce_signal_to_zero(mz, ints, mz_segments_to_zero):
    """
    Reduce signal to zero
    """
    artefacts_bool = np.zeros(len(mz), dtype=bool)
    for distortion in mz_segments_to_zero:
        artefacts_bool = artefacts_bool | ((mz > distortion[0]) & (mz < distortion[1]))
        ints[artefacts_bool] = 0
    return mz, ints

class Drawer():
    def __init__(self, datasource: "str | DataSource | PreparedDataSource"):
        if isinstance(datasource, (DataSource, str)):
            if isinstance(datasource, str):
                datasource = DataSource(datasource)
            self.processed_spectra_path = datasource.processed_spectra_path
            self.peaklist_path = datasource.peaklist_path
            self.datasource = datasource
            if datasource.configs_path is not None:
                self.prepdata = PreparedDataSource(datasource, datasource.configs_path)
            else:
                self.prepdata = None
                if self.processed_spectra_path is None and self.peaklist_path is None:
                    raise ValueError("No processed spectra, peaklist or configs found. Only raw datasource")
        elif isinstance(datasource, PreparedDataSource):
            self.prepdata = datasource
            self.datasource = datasource._datasource
            self.processed_spectra_path = self.datasource.processed_spectra_path
            self.peaklist_path = self.datasource.peaklist_path
    def _draw(self,
              mz: np.ndarray,
              intens: np.ndarray,
              peaklist: np.ndarray | None = None,
              headers: list[str] | None = None,
              roi: str | None = None,
              mz_range: tuple[float, float] | None = None,
              spectrum_idx: int | None = None):
        
        plt.figure().set_figwidth(25)
        plt.gcf().set_figheight(5)
        datasource = self.datasource
        diapcalc = lambda mz, plot_mz_range: (np.array(mz>plot_mz_range[0]) & np.array(mz<plot_mz_range[1])) if plot_mz_range is not None else range(len(mz))

        # Draw raw
        Label = ["Raw mass spectrum"]
        mz_raw, intens_raw = datasource.get_spectrum(spectrum_idx)

        if mz is None:
            mz = mz_raw
        if mz_range is None:
            mz_range = (mz[0], mz[-1])

        diap_raw = diapcalc(mz_raw, mz_range)
        plt.plot(mz_raw[diap_raw], intens_raw[diap_raw],alpha=0.75)

        # Draw processed

        Label.append("Processed mass spectrum")
        diap = diapcalc(mz, mz_range)
        plt.plot(mz[diap], intens[diap],alpha=0.75)
        
        # Draw peaklist
        if peaklist is not None:
            if isinstance(peaklist, np.ndarray):
                peaklist = pd.DataFrame(peaklist, columns = headers)
            peaklist = peaklist.astype({"spectra_ind": int})
            peaklist.query("mz>@mz_range[0] and mz<@mz_range[1] and spectra_ind == @spectrum_idx").plot(x="mz",y="Intensity",ax = plt.gca(),style = "x", color = "k")
            left_intens=[]
            for left_base in peaklist.query("PextL>@mz_range[0] and PextL<@mz_range[1] and spectra_ind == @spectrum_idx")['PextL']:
                left_intens.append(intens[mz>=left_base][0])
            right_intens = []
            for right_base in peaklist.query("PextR>@mz_range[0] and PextR<@mz_range[1] and spectra_ind == @spectrum_idx")['PextR']:
                right_intens.append(intens[mz<=right_base][-1])
            plt.plot(peaklist.query("PextL>@mz_range[0] and PextL<@mz_range[1] and spectra_ind == @spectrum_idx")['PextL'],
            left_intens,'v')
            plt.plot(peaklist.query("PextR>@mz_range[0] and PextR<@mz_range[1] and spectra_ind == @spectrum_idx")['PextR'],
            right_intens,'^')
            Label=Label+[f'Peaks', 'Left peak base','Right peak base']
        plt.grid(visible=True,which="both")
        plt.xlim(mz_range)
        plt.legend([*Label])
        plt.minorticks_on()
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.title(f"Sample: {datasource.sample_name}, roi: {roi}, spectrum idx: {spectrum_idx}")
        plt.show()

    def audit_processing(self,
                         roi: str | list | None = None,
                         draw_mz_range: tuple[float, float] | None = None,
                         draw_spectrum_idx: int | None = None,
                         dtypeconv: np.dtype | None = None):
        if roi is None:
            roi = list(self.datasource.roi_metadata.index)
        elif isinstance(roi, str):
            roi = [roi]
        elif isinstance(roi, list):
            pass
        else:
            raise ValueError("Invalid roi")

        pipeline = Pipeline(self.prepdata)
        datasource = self.datasource
        roi_metadata = datasource.roi_metadata
        for r in roi:
            if draw_spectrum_idx is None:
                rmeta = roi_metadata.loc[r]
                idxs = Indexator(rmeta["idxroi"].to_numpy(int))
                spectrum_idx = list(idxs)[np.random.randint(0,idxs.count)]
            else:
                spectrum_idx = draw_spectrum_idx

            processing_stream = pipeline._multistream_pipeline(Pipeline._procfunc_wrapper, r, idxs = spectrum_idx, dtypeconv=dtypeconv)
            mz, headers = next(processing_stream)
            loc_idx, proc_intensity = next(processing_stream)
            roi_configs = self.prepdata.roi_configs[r]
            peakpick_function = roi_configs._peakpick_function
            if peakpick_function:
                peaklist_configs = roi_configs.get_step_configs('peakpick')
                peaklist = peakpick_function(mz,
                                             proc_intensity.squeeze(),
                                             [spectrum_idx],
                                             peaklist_configs,
                                             )
                # headers = peaklist_configs['headers']
            else:
                peaklist = None

            self._draw(mz, proc_intensity.squeeze(), peaklist, headers, r, draw_mz_range, spectrum_idx)

    def draw_processed_data(self,
                            roi: str | list | None = None,
                            draw_mz_range: tuple[float, float] | None = None,
                            draw_spectrum_idx: int | None = None):
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
            
            if self.processed_spectra_path is None:
                if self.prepdata is None:
                    raise ValueError("No processed spectra path or configs to get processed spectrum")
                else:
                    pipeline = Pipeline(self.prepdata)
                    stream = pipeline._multistream_pipeline(Pipeline._procfunc_wrapper,r, cpu_num=1, idxs = spectrum_idx)
                    mz, headers = next(stream)
                    _, data_int = next(stream)
                    data_int = data_int[0]
            else:
                with File(self.processed_spectra_path, "r") as hdf5:
                    data_int = hdf5[r]["int"][datasource._get_local_roi_idx(spectrum_idx), :]
                    mz = hdf5[r]["mz"][:]
            if self.peaklist_path is not None:
                with File(self.peaklist_path, "r") as hdf5:
                    headers = hdf5[r]['peaklists'].attrs["Column headers"]
                    peaklist = pd.DataFrame(hdf5[r]["peaklists"][:], columns = headers).astype({"spectra_ind": int}).query('spectra_ind == @spectrum_idx')
            else:
                peaklist = None

            self._draw(mz, data_int, peaklist, headers, r, draw_mz_range, spectrum_idx)
class Pipeline:
    def __init__(self,
                 prepdata: "PreparedDataSource"):
        '''WIP
        Unified interface for running MSI data processing.

        Pipeline is a thin orchestrator that accepts a PreparedDataSource 
        and runs processing methods using the pipeline functions stored in the configs object.

        Parameters
        ----------
        configs : PreparedDataSource
            Configuration object with datasource and pipeline functions.
        '''
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
        '''
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
        '''
        datasource = self._datasource
        
        hdf5_save_path = os.path.join(os.path.split(datasource.file_path)[0],'processed_pelmesha',datasource.sample_name + '_processed_spectra.hdf5')
        if os.path.exists(hdf5_save_path):
            os.remove(hdf5_save_path)
        if not os.path.exists(os.path.split(hdf5_save_path)[0]):
            os.makedirs(os.path.split(hdf5_save_path)[0])

        cpu_num = cpu_count()-free_cpus
        roi_metadata = datasource.roi_metadata

        if dtypeconv is None:
            dtypeconv = datasource.metadata.iloc[0]["dtype_raw"]
        dtypeconv = np.dtype(dtypeconv)
        bytes_flsize = dtypeconv.itemsize
        chunk_size_by_elements = int(max(1,np.ceil(h5chunk_size_MB*(1024**2)/bytes_flsize)))
        for roi in roi_metadata.index:
            print(f'Processing ROI {roi}')
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
             
        datasource = self._datasource
        hdf5_save_path = os.path.join(os.path.split(datasource.file_path)[0],'processed_pelmesha',datasource.sample_name + '_peaklists.hdf5')
        if os.path.exists(hdf5_save_path):
            os.remove(hdf5_save_path)
        if not os.path.exists(os.path.split(hdf5_save_path)[0]):
            os.makedirs(os.path.split(hdf5_save_path)[0])

        if dtypeconv is None:
            dtypeconv = datasource.metadata.iloc[0]["dtype_raw"]
        dtypeconv = np.dtype(dtypeconv)
        cpu_num = cpu_count()-free_cpus
        bytes_flsize = dtypeconv.itemsize
        chunk_size_by_elements = int(max(1,np.ceil(h5chunk_size_MB*(1024**2)/bytes_flsize)))
        
        roi_metadata = datasource.roi_metadata
        for roi in roi_metadata.index:
            print(f'Processing ROI {roi}')
            peakpicking_stream = self._multistream_pipeline(self._peakpick_wrapper,
                                                           roi = roi,
                                                           cpu_num = cpu_num,
                                                           Ram_GB_limit = Ram_GB_limit,
                                                           dtypeconv = dtypeconv)
            gen_mz, headers_list = next(peakpicking_stream)
            
            with File(hdf5_save_path,"a") as hdf5:
                n_heads = len(headers_list)
                hdf5.create_dataset(roi + "/peaklists",(0, n_heads), maxshape = (None, n_heads), chunks=(chunk_size_by_elements/n_heads, n_heads), dtype=dtypeconv)
                hdf5[roi]["peaklists"].attrs["Column headers"] = headers_list
                for peaklists in peakpicking_stream:
                    list_size = len(peaklists)
                    hdf5[roi]["peaklists"].resize((hdf5[roi]["peaklists"].shape[0] + list_size, n_heads))
                    hdf5[roi]["peaklists"][-list_size:,:] = peaklists

            if draw:
                Drawer(self.prepdata).draw_processed_data(roi, draw_mz_range, draw_spectrum_idx)

        self.prepdata.save()
        
    def _multistream_pipeline(self,
                              process_wrapper: Callable,
                              roi: str = None,
                              cpu_num: int = 1, 
                              Ram_GB_limit: int = 2,
                              dtypeconv:  np.dtype | str | None = None,
                              idxs: Indexator | SliceIndexator | int | None = None):
        """Основная функция генератор результатов мультипроцессинга"""
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
        elif process_wrapper.__name__ == "_peakpick_wrapper":
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
        
        if isinstance(idxs, int):
            row_idxs, data = partial_worker(np.asarray((idxs,idxs+1), dtype=np.int64))
            yield row_idxs, data
        else:
            if mz is not None:
                size_per_spec = len(mz)
            idxs_batches = datasource.split_idxs(idxs = idxs,cpu_count=cpu_num, Ramcap_GB = Ram_GB_limit, size_per_spec = size_per_spec)

            with Pool(cpu_num) as p:
                for data in tqdm(p.imap_unordered(partial_worker, idxs_batches), total=len(idxs_batches), unit = 'batch'):
                    yield data
        

    @staticmethod
    def _procfunc_wrapper(idxs: Indexator | SliceIndexator | tuple| np.ndarray,
                          datasource: "DataSource",
                          configs: "dict | Configs | PipelineConfigurator",
                          dtypeconv: np.dtype | None = None,
                          **internal_configs
                          ):
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
