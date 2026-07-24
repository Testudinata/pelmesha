import numpy as np

import math 
from pybaselines import Baseline
from pelmesha.dough import AdaptiveParameter, DatasetHeaders, Indexator
from scipy.signal import savgol_filter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pelmesha import Configs, PipelineConfigurator, DataSource

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
    
    
    baseline_algo = configs.configs.get('method', None)
    if baseline_algo:
        baseline_algo = baseline_algo.get('Baseline', None)
        if baseline_algo:
            baseline_algo = next(iter(baseline_algo))
    if baseline_algo:
        if datasource.dcont or resampled_mz:
            if resampled_mz:
                internal_configs['process']['Baseliner'] = getattr(Baseline(resampled_mz, **configs['Baseline']), baseline_algo)
            else: 
                internal_configs['process']['Baseliner'] = getattr(Baseline(datasource.get_mz(rmeta['idxroi'].flatten()[0]), **configs['Baseline']), baseline_algo)
        else:
            internal_configs['process']['Baseliner'] = baseline_algo
    internal_configs['peakpick'] = {}

    # Configure headers conditionally based on processing flags 
    if configs['peakpicker']["SNR_threshold"] and configs['peakpicker']["Calc_peak_area"]:
        configs.update({"headers": DatasetHeaders([  
                                        "spectra_ind", "mz", "Intensity", "Area", "SNR",  
                                        "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"  
                                    ])})
    elif configs['peakpicker']["SNR_threshold"]:
        configs.update({"headers": DatasetHeaders([  
                                        "spectra_ind", "mz", "Intensity", "SNR",  
                                        "PextL", "PextR", "FWHML", "FWHMR", "Noise", "Mean noise"  
                                    ])})
    elif configs['peakpicker']["Calc_peak_area"]:
        configs.update({"headers": DatasetHeaders([  
                                        "spectra_ind", "mz", "Intensity", "Area",  
                                        "PextL", "PextR", "FWHML", "FWHMR"
                                    ])})
    else: 
        configs.update({"headers": DatasetHeaders([  
                                        "spectra_ind", "mz", "Intensity",  
                                        "PextL", "PextR", "FWHML", "FWHMR"
                                    ])})

    return resampled_mz, internal_configs


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
    # TODO внести add_zero_points
    resampled_mz = internal_configs.get('resampled_mz', None)
    baseline_algo = internal_configs.get('baseline_algo', None)


    smooth_configs = configs['smoothing']
    msalign_configs = configs['msalign']
    if len(intensity.shape)==1:
        intensity = intensity[np.newaxis, :]
    if smooth_configs['smooth_algo'] is not None: #Smoothing step
        intensity = tuple(smoothing(spectrum, **smooth_configs) for spectrum in intensity)
    if baseline_algo is not None: # BaselineCorrection step
        if isinstance(baseline_algo,str):
            if baseline_algo == 'asls': #Just for setting default params
                intensity = Baseline(mz, **configs['Baseline']).asls(intensity, **configs['asls'])
            else:
                intensity = getattr(Baseline(mz, **configs['Baseline']), baseline_algo)(intensity, **configs[baseline_algo])
        # if isinstance(Baseliner, AdaptiveParameter):
        #     intensity = Baseliner(mz, configs['Baseline'])(intensity, **configs[Baseliner.__name__])
        else:
            intensity = tuple(baseline_algo(spectrum, **configs[baseline_algo.__name__]) for spectrum in intensity)
    if resampled_mz is not None: # Resampling step
        intensity = tuple(
        np.interp(resampled_mz, mz, spectrum, left=intensity[0], right=intensity[-1])
        for spectrum in intensity)
        mz = resampled_mz
        
    if isinstance(intensity, tuple):
        intensity = np.vstack(intensity)
    if msalign_configs['align_peaks'] is not None:
        intensity = msalign(mz, intensity, **msalign_configs)
    
    return mz, intensity

def peakpicking_base(
        mz: np.ndarray,
        intensity: np.ndarray,
        idxs: Indexator | list,
        configs: "Configs | PipelineConfigurator",
        **internal_configs) -> np.ndarray:

    mzsize = mz.size
    int_shape = intensity.shape
    if len(int_shape)==1:
        
        intensity = intensity[np.newaxis, :] 
    if int_shape[0] <2:
        return peakpicker(mz,
                         intensity,
                         mzsize,
                         idxs[0],
                         **configs)
    else:
        peaklists = {}
        for n, idx in enumerate(idxs):
            peaklists[n] = peakpicker(mz,
                                      intensity[n,:],
                                      mzsize,
                                      idx,
                                      **configs)
        return np.vstack(tuple(peaklists.values()))

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
    align_peaks: List,
    method: str = "cubic",
    width: float = 10,
    ratio: float = 2.5,
    resolution: int = 100,
    iterations: int = 5,
    grid_steps: int = 20,
    shift_range: List = None,
    align_pweights: List = None,
    return_shifts: bool = False,
    align_by_index: bool = False,
    only_shift: bool = False,
):
    aligner = Aligner(
        x,
        array,
        align_peaks,
        method=method,
        width=width,
        ratio=ratio,
        resolution=resolution,
        iterations=iterations,
        grid_steps=grid_steps,
        shift_range=shift_range,
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
              smooth_algo: str = 'GA', 
              smooth_window: int = 7, 
              smooth_cycles: int = 2) -> np.ndarray:
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
              SNR_threshold = None, 
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

def _baseliner_prep(baseline_algo,mz_scale, baseliner_init_configs):
    if baseline_algo:
        return getattr(Baseline(mz_scale, **baseliner_init_configs),baseline_algo)
    else:
        return None
    
def _shift_range_to_dots(window_shift_mz: tuple | list | None,
                         dots_distance: float) -> tuple | None:
    """Convert m/z shift range to dot-based shift range."""
    if window_shift_mz:
        return tuple(int(shift_mz / dots_distance) for shift_mz in window_shift_mz)
    return None
