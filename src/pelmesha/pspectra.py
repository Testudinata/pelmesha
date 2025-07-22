import pandas as pd
import numpy as np
from itertools import product, pairwise, zip_longest
from torch.multiprocessing import Pool, cpu_count, Manager
from threading import Thread
from pybaselines import Baseline
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
from pelmesha.loaders import find_paths, logger
from pyimzml.ImzMLParser import ImzMLParser
from h5py import File
import gc
import math
import os
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from math import sqrt

### Code from __init__.py msalign (https://github.com/lukasz-migas/msalign)
"""Signal calibration and alignment by reference peaks - copy of MSALIGN function from MATLAB bioinformatics library."""
from .align import Aligner
from typing import List
__all__ = ["msalign", "Aligner"]


def msalign(
    x: np.ndarray,
    array: np.ndarray,
    peaks: List,
    method: str = "cubic",
    width: float = 10,
    ratio: float = 2.5,
    resolution: int = 100,
    iterations: int = 5,
    grid_steps: int = 20,
    shift_range: List = None,
    weights: List = None,
    return_shifts: bool = False,
    align_by_index: bool = False,
    only_shift: bool = False,
):
    aligner = Aligner(
        x,
        array,
        peaks,
        method=method,
        width=width,
        ratio=ratio,
        resolution=resolution,
        iterations=iterations,
        grid_steps=grid_steps,
        shift_range=shift_range,
        weights=weights,
        return_shifts=return_shifts,
        align_by_index=align_by_index,
        only_shift=only_shift,
    )
    aligner.run()
    return aligner.apply()


msalign.__doc__ = Aligner.__doc__
### END of code from __init__.py msalign (https://github.com/lukasz-migas/msalign)


### Base functions

def imzml2hdf5(path_list, dtypeconv='single', chunk_rowsize = "Auto", chunk_bsize = 10000000, reconv = False):
    """
    Description
    ----
    Conversion of raw data from imzML to hdf5 according to a list of file paths/root folders from path_list.

    If a folder path rather than a file path is provided, a search for imzML files in subfolders is performed.

    The conversion of imzML file data is written to the folder above the root folder for the imzML file. 
    If the upper folder is common for several files, their data is written to a single hdf5 file but into different hdf5 datasets.
    The dataset name is taken from the name of the folder containing the imzML file.

    :param path_list: list of str or paths to folder or `imzML` file
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param chunk_rowsize: chunking hdf5 datasets for partial and efficient loading data to RAM. The default is `"Auto"`

        `"Auto"` - автоматический подбор кол-ва строк записи матрицы в hdf5 на основе размера chunk_bsize,

        `"Full"` - датасет сохраняется и выгружается целиком (занимает много RAM), 

        `int` - датасет дробится по заданному числу строк, где каждая строка это данные спектра.
    :param chunk_bsize: the chunk size for hdf5 datasets in bytes. Optional, works only with chunk_rowsize =`"Auto"`. The default is `10000000`
    
    :type path_list: list or str
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type chunk_rowsize: {`"Auto"`,`"Full"`, `int`}
    :type chunk_bsize: `int`, optional

    :return: None
    """

    if isinstance(path_list, str):
        path_list=[path_list]
    sample_imzmlpath_list=[] # Словарь списка sample для каждого слайда (отдельный путь в path_list)
    
    imzmlpath_list = find_paths(path_list) ## Поиск файлов imzml и создание списка корневых папок с файлами ".imzml"
    sample_tot_num = len(imzmlpath_list)  # счётчик общего количества sample, используется для создания количества процессов не более этого значения (не критично, но оптимально вдруг, чтобы не создавать пул нерабочих процессов, что возможно ест ресурс компа)
    if sample_tot_num ==0:
        warnings.warn("Sample total num is - 0. Couldn't find imzML files")
        return
    ##
    ## Создание списков наименований слайдов, roi и рассчёт общего количества roi
    for path in imzmlpath_list:

        Slides_path=os.path.dirname(os.path.dirname(path)) #Определяем путь в root директорию. Это директория, где будут храниться данные обработки в hdf5 файле
        Slide_name=os.path.basename(Slides_path) #Определяем имя слайда
         
        if all('_rawdata.hdf5' not in item for item in os.listdir(Slides_path)): # условие для выгрузки из imzml и конвертации данных в hdf5     
            sample_imzmlpath_list.append([Slides_path,path])
        else:
            if reconv:
                os.remove(os.path.join(Slides_path,Slide_name)+"_rawdata.hdf5")
                sample_imzmlpath_list.append([Slides_path,path])
            else:
                print(f"Data on the path {path} has hdf5 file for raw data. Change argument 'reconv' to True, if needed to reconvert")

    ##    
    
    ## Определение количества пула процессов
    cpu_num = cpu_count()-1
    if cpu_num > sample_tot_num:
        cpu_num = sample_tot_num
    

    ## Выгрузка данных с помощью ImzMLParser'a и их конвертация в hdf5 (в дальнейшем работаем с hdf5)
    manager = Manager()
    print_queue = Manager().Queue()
    queue = manager.Queue()
    queue.put(True)
    t = Thread(target=printer,args=[print_queue])
    t.start()
    print_queue.put(len(sample_imzmlpath_list))
    with Pool(cpu_num) as p:
        p.starmap(hdf5_writer,product(sample_imzmlpath_list,[queue],[print_queue],[dtypeconv],[chunk_rowsize],[chunk_bsize]))
    p.join()

    print_queue.put(Sentinel()) # Остановка работы функции printer
    t.join() # Wait for all printing to complete
    ##
    return None

def Raw2proc(data_obj_path, baseliner_algo = 'asls', params2baseliner_algo={}, #penalized_poly - самый быстрый вариант. asls - меньше "отрицательных" точек по сравнению с penalized_poly, что лучше работает с пикпикингом с фильтрацией порогом по интенсивности, но в ~2 раза дольше считает
              align_peaks = None, weights_list=None, max_shift_mz=0.95, only_shift = True,params2align={},
              resample_to_dots = None, 
              smooth_algo = None, smooth_window=0.075, smooth_cycles=1,
              draw = True, mz_diap4draw = None, rewrite = False, Ram_GB = 1, h5chunk_size_MB = 10,dtypeconv='single',
              free_cores=1):
    """
    Общее описание
    ----
    Функция обработки сырых спектров из `imzML` только до обработанных спектров и после их записи в файл hdf5 под названием "Slidename_specdata.hdf5" в датасет "int". Конечным результатом является обработанный спектр. Требуется много места на жёстком диске.

    :param data_obj_path: list of paths to root folders where to search imzml files in subfolders 
    :param baseliner: Baseline class for baseline correction
    :param baseliner_algo: Algorithm of baseline correction. Default: `"asls"`

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    :param params2baseliner_algo: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html). Default: `{}`

        .. Example: {"lam" : 500000, "diff_order" : 1}
    :param align_peaks: list of reference peaks for align. Default: `None`
    :param weights_list: list of weights for reference peaks in aligning. Default: `None`
    :param max_shift_mz: max spectrum shift at aligning in mz. Default: `0.95`
    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`). Default: `{}`

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param smooth_algo: spectrum smoothing algorithm. Default is `"None"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param smooth_window: window size in mz for smooth. Default:`0.075`
    :param smooth_cycles: Number of iterations for spectrum smooth. Default: `1`
    :param draw: Draw example graphs of raw and proccessed random spectrum of image. Default: `True`
    :param mz_diap4draw: Range for graphs draw. Default: `None`
    :param rewrite: delete old hdf5 before writing new spectra data. Default: `False`
    :param Ram_GB: Determine max sizes in GB of the data proccesing on CPU cores at moment.
    :param h5chunk_size_MB: the chunk size for hdf5 datasets writing in MB
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param free_cores: Number of CPU cores don't used in multiproccessing

    :type data_obj_path: `list`
    :type max_shift_mz: `float`
    :type resample_to_dots: `int`
    :type baseliner: `Baseline` class
    :type baseliner_algo: `str`
    :type params2baseliner_algo: `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`,`None`}
    :type smooth_window: `float`
    :type smooth_cycles: `int`
    :type draw: `bool`
    :type mz_diap4draw: `list` or `None`
    :type rewrite: `bool`
    :type Ram_GB: `float`
    :type h5chunk_size_MB: `float`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type free_cores: `1`

    :return: `None`
    :rtype: `NoneType`
    """
    
    # Process args
    #defaults parametres for align
    pars = list(set(["width","iterations"])-set(params2align.keys()))
    if pars:
        params2align_default = {"iterations":3, "width":0.3}
        for par in pars:
            params2align[par]=params2align_default[par]
    params2align["only_shift"]=only_shift
    #Create thread for printing text in multiprocessing
    manager = Manager()
    print_queue = Manager().Queue()
    queue = manager.Queue()
    queue.put(True)
    t = Thread(target=printer,args=[print_queue])  
    t.start()
    if isinstance(data_obj_path,str):
        data_obj_path=[data_obj_path]
    # Определение количества пула процессов
    cpu_num = cpu_count()-free_cores
    Ram_GB = Ram_GB*1e+9
    batch_bsize = Ram_GB/cpu_num
    if dtypeconv =='single':
        bytes_flsize = 4
    elif dtypeconv == 'double':
        bytes_flsize = 8
    elif dtypeconv == 'half':
        bytes_flsize = 2
    ##
    
    ###I. Finding slide directory for rawdata of samples (imzml)
    path_dict=find_imzml_roots(data_obj_path)
    ###I Finding slide directory with rawdata of samples (path_list) - DONE
    
    # Working with slides
    for file_path in path_dict.keys():
        slide = os.path.basename(file_path)  
        print(f"The {slide} raw spectra data is on progress.")
        
        data_obj_coord={}
        sample_list = path_dict[file_path]
        # sample_list =[]
        # # Searching direct path to imzml files (samples) 
        # for root, dirs, files in os.walk(file_path):
        #     for file in files: 
        #         if file.endswith('.imzML'):
        #             sample_list.append(os.path.join(root,file))

        ###II. Extracting spectra coordinates, roi indexes, other metadata for proccessing slide samples from poslog and _info text files. Input arguments organization for "intprocc_parbatched_imzml" function (spectra processing batched and parallelized)
        ###II. Вытаскивание данных координат точек и индексов для областей(roi), если их несколько, из poslog и _info. Организация входных параметров для функции "intprocc_parbatched_imzml", включая batch'ей для параллельной выгрузки
        print(f"Slide's {slide} spectra coordinates and metadata extraction for preparation parallel proccessing")
        with Pool(cpu_num) as p:
            data_obj_temp = p.starmap(poslog_parbatched,list(product(sample_list,[batch_bsize],[dtypeconv],[print_queue],[cpu_num],[resample_to_dots])))
        p.join()    
        args_batches=[]
        for data in data_obj_temp:
            data_obj_coord=data_obj_coord|data[0]
            args_batches = args_batches + data[1]
        del data_obj_temp
        gc.collect()
        #### Подготовка аргументов для процессинга данных

        args2procc = Manager().dict({"baseliner_algo": baseliner_algo, "params2baseliner_algo": params2baseliner_algo,"params2align":params2align, "align_peaks":align_peaks,"weights_list":weights_list, "dots_shift":max_shift_mz,"smooth_algo":smooth_algo,"smooth_window":smooth_window, "smooth_cycles":smooth_cycles})
            ## Определение среза по строкам (chunk) в hdf5
        if resample_to_dots is not None:
            chunk_size = np.ceil(h5chunk_size_MB*1e+6/(bytes_flsize*resample_to_dots))
            args_batches=list(args_batch+(resample_to_dots,args2procc,queue, chunk_size) for args_batch in args_batches)
        else:
            chunk_size={}
            temp_args_batches=[]
            for args_batch in args_batches:
                chunk_size[args_batch[1]] = np.ceil(h5chunk_size_MB*1e+6/(bytes_flsize*args_batch[4]))
                temp_args_batches.append(args_batch+(resample_to_dots,args2procc,queue,chunk_size[args_batch[1]]))
            args_batches=temp_args_batches
        ##II. Coordinates, metadata and organization of input arguments for parallelized and batched proccessing of spectra - DONE
        ## Deleting old hdf5 file if it has
        
        if rewrite: # if TRUE - delete hdf5
            try:
                os.remove(os.path.join(file_path,slide)+"_specdata.hdf5")
            except:
                pass
        ## 
        if os.path.isfile(os.path.join(file_path,slide)+"_specdata.hdf5"): # if hdf5 exist
            with File(os.path.join(file_path,slide)+"_specdata.hdf5","r") as data_obj_procc:

                for sample in data_obj_procc.keys():
                    for roi in data_obj_procc[sample].keys():    
                        try:
                            data_obj_procc[sample][roi]['int']
                            data_obj_procc.close()
                            os.remove(os.path.join(file_path,slide)+"_specdata.hdf5")
                            print("Old hdf5 file deleted")
                            
                            break
                            #print('tryed',sample,roi)
                        except:
                            pass
                        
                    else:
                        continue
                    break


        print(f"Slide's {slide} spectra parallel proccessing")

        hdf5_coords(file_path,slide,data_obj_coord,chunk_size)

        print_queue.put(len(args_batches)) #Создаём tqdm для отслеживания процесса обработки
        with Pool(cpu_num) as p:
            p.starmap(int2procc_parbatched,args_batches)
        p.join()
        ##III. Proccessing, peakpicking and writing to hdf5 - Done
        ##IV. Drawing example result
        
        if draw and (resample_to_dots is not None or data_obj_coord[sample][roi]["continuous"]):
            data_obj_procc = File(os.path.join(file_path,slide)+"_specdata.hdf5","r")
            print(f"Slide's {slide} spectra drawing results")
            for sample in data_obj_coord.keys():
                for roi in data_obj_coord[sample].keys():
                 #Секция для отрисовки полученных результатов
                    #num_spec=np.random.randint(0,data_obj_procc[sample][roi]['int'][:].shape[0])
                    
                    plt.figure().set_figwidth(25)
                    plt.gcf().set_figheight(5)
                    idx_start, numspec = data_obj_coord[sample][roi]["idxroi"]
                    raw = ImzMLParser(data_obj_coord[sample][roi]["source"])
                    #num_spec = np.random.randint(0,len(raw.mzLengths))
                    idx_spec = np.random.randint(idx_start,idx_start+numspec)
                    data_mz_old, data_int_old = raw.getspectrum(idx_spec)
                    print(f'Spectrum number: {idx_spec}')
                 
                    if mz_diap4draw is not None:
                        diapold=(np.array(data_mz_old > mz_diap4draw[0]) & np.array(data_mz_old < mz_diap4draw[1]))
                        diapnew=(np.array(data_obj_procc[sample][roi]['mz'][:]>mz_diap4draw[0]) & np.array(data_obj_procc[sample][roi]['mz'][:]<mz_diap4draw[1])) 
                    else:
                        diapold=range(len(data_mz_old))
                        diapnew=range(len(data_obj_procc[sample][roi]["mz"]))
                    plt.plot(data_mz_old[diapold], data_int_old[diapold],'--')
                    plt.plot(data_obj_procc[sample][roi]["mz"][diapnew], data_obj_procc[sample][roi]["int"][idx_spec-idx_start,:][diapnew],alpha=0.75)
                    plt.grid(visible=True,which="both")
                    plt.legend(['Raw spectra', 'Proccessed spectra'])
                    plt.xlabel("m/z")
                    plt.ylabel("Intensity")
                    plt.minorticks_on()
                    plt.xlim((mz_diap4draw[0],mz_diap4draw[1]))
                    plt.show()
                    
                    del diapold, diapnew
                
                gc.collect()
            data_obj_procc.close()
        with open(os.path.join(file_path,"Processing_settings.txt"), "w") as file:
            file.write("###Raw spectra proccessing\n##Spectra aligning(msalign function):\n")
            with pd.option_context('display.max_colwidth', None):
                with pd.option_context('display.max_rows', None):
                    if isinstance(weights_list,np.ndarray):
                        weights_list = np.array2string(weights_list,separator=',')
                    file.write(f"Aligning peaks list:{align_peaks}\nPeaks weights: {weights_list}\nMax shift in mz scale: {max_shift_mz}\nOther align parametres:")
            for key in params2align.keys():
                file.write(f"{key}: {params2align[key]}\n")
            file.write(f"##Resampling data (interp1d function):\nResampling to dots:{resample_to_dots}\n")
            file.write(f"Previous data in dots:\n")
            for sample in data_obj_coord.keys():
                for roi in data_obj_coord[sample].keys():
                    file.write(f'   for {sample} roi {roi}: ')
                    sourced=ImzMLParser(data_obj_coord[sample][roi]['source'])
                    if data_obj_coord[sample][roi]['continuous']:
                        data_mz_old=sourced.getspectrum(0)[0]
                        file.write(f'continious data with {len(data_mz_old)} number of dots between {data_mz_old[0]} and {data_mz_old[-1]} m/z')
                    else:
                        file.write(f'proccessed data with median {np.median(sourced.mzLengths)} number of dots')

            file.write(f"##Baseline correction (pybaseline package, class Baseline)):\nAlgorithm:{baseliner_algo}\n")
            for key in params2baseliner_algo.keys():
                file.write(f"{key}: {params2baseliner_algo[key]}\n")
            file.write(f"##Smoothing (smoothing function):\nSmooth algorithm: {smooth_algo}\nWindow {smooth_window}\nCycles {smooth_cycles}\n")    
            file.write(f"##Other parametres:\nBatch size: {batch_bsize/1000} Mb\nData type convertion to: {dtypeconv}\n")  
        print_queue.put(0)

    # Закрытие hdf5 объёктов
    # Closing threads and hdf5 object
    
    print_queue.put(Sentinel())
    t.join()
    
       
    return

def Raw2peaklist(data_obj_path, baseliner_algo = 'asls', params2baseliner_algo={}, #penalized_poly - самый быстрый вариант. asls - меньше "отрицательных" точек по сравнению с penalized_poly, что лучше работает с пикпикингом с фильтрацией порогом по интенсивности, но в ~2 раза дольше считает
              align_peaks = None, weights_list=None, max_shift_mz=0.95,only_shift = True,params2align={},
              resample_to_dots = None, 
              smooth_algo = None, smooth_window=0.075, smooth_cycles=1,
              oversegmentationfilter = 0, fwhhfilter = 0, heightfilter=0, peaklocation=1,rel_heightfilter=0,
              SNR_threshold = 3.5, noise_est = "std",noise_est_iterations = 3,
              draw = True, mz_diap4draw = None, rewrite = False, Ram_GB = 1, h5chunk_size_MB = 10,dtypeconv='single',
              free_cores=1):
    """
    Общее описание
    ----
    Функция обработки сырых спектров из `imzML` до пиклиста и после их записи в файл hdf5 под названием "Slidename_specdata.hdf5" в датасет "peaklists". Конечным результатом является полный пиклист-матрица имаджа.

    :param data_obj_path: list of paths to root folders where to search imzml files in subfolders 
    :param baseliner: Baseline class for baseline correction
    :param baseliner_algo: Algorithm of baseline correction. Default: `"asls"`

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    :param params2baseliner_algo: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html). Default: `{}`

        .. Example: {"lam" : 500000, "diff_order" : 1}
    :param align_peaks: list of reference peaks for align. Default: `None`
    :param weights_list: list of weights for reference peaks in aligning. Default: `None`
    :param max_shift_mz: max spectrum shift at aligning in mz. Default: `0.95`
    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`). Default: `{}`

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param smooth_algo: spectrum smoothing algorithm. Default is `"None"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `0`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `0`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `0`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `0`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `3.5`
    :param noise_est: алгоритм оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `"std"`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    :param smooth_window: window size in mz for smooth. Default:`0.075`
    :param smooth_cycles: Number of iterations for spectrum smooth. Default: `1`
    :param draw: Draw example graphs of raw and proccessed random spectrum of image. Default: `True`
    :param mz_diap4draw: Range for graphs draw. Default: `None`
    :param rewrite: delete old hdf5 before writing new spectra data. Default: `False`
    :param Ram_GB: Determine max sizes in GB of the data proccesing on CPU cores at moment.
    :param h5chunk_size_MB: the chunk size for hdf5 datasets writing in MB
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param free_cores: Number of CPU cores don't used in multiproccessing

    :type data_obj_path: `list`
    :type max_shift_mz: `float`
    :type resample_to_dots: `int`
    :type baseliner: `Baseline` class
    :type baseliner_algo: `str`
    :type params2baseliner_algo: `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`,`None`}
    :type oversegmentationfilter: `float`
    :type fwhhfilter: `float`
    :type heightfilter: `float`
    :type peaklocation: `float` and =<1
    :type rel_heightfilter: `float`
    :type SNR_threshold: `float`
    :type noise_est: {`"std"`,`"mad"`}
    :type noise_est_iterations: `int`
    :type smooth_window: `float`
    :type smooth_cycles: `int`
    :type draw: `bool`
    :type mz_diap4draw: `list` or `None`
    :type rewrite: `bool`
    :type Ram_GB: `float`
    :type h5chunk_size_MB: `float`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type free_cores: `1`

    :return: `None`
    :rtype: `NoneType`
    """
    
    # Process args
    #defaults parametres for align
    pars = list(set(["width","iterations"])-set(params2align.keys()))
    if pars:
        params2align_default = {"iterations":3, "width":0.3}
        for par in pars:
            params2align[par]=params2align_default[par]
    params2align["only_shift"]=only_shift
    
    if not isinstance(peaklocation, (int, float)) or not np.isscalar(peaklocation) or peaklocation < 0 or peaklocation > 1:
        raise ValueError("mspeaks: Invalid peak location")
    if not isinstance(fwhhfilter, (int, float)) or not np.isscalar(fwhhfilter) or fwhhfilter < 0:
        raise ValueError("mspeaks: Invalid FWHH filter")
    if not isinstance(oversegmentationfilter, (int, float)) or not np.isscalar(oversegmentationfilter):
        if isinstance(oversegmentationfilter, str):
            oversegmentationfilter=oversegmentationfilter.lower()
        else:
            raise ValueError("mspeaks: Invalid oversegmentation filter")
    elif oversegmentationfilter < 0:
        raise ValueError("mspeaks: Invalid oversegmentation filter")
    if not isinstance(heightfilter, (int, float)) or not np.isscalar(heightfilter) or heightfilter < 0:
        raise ValueError("mspeaks: Invalid height filter")
    if not isinstance(rel_heightfilter, (int, float)) or not np.isscalar(rel_heightfilter) or rel_heightfilter < 0 or rel_heightfilter > 100:
        raise ValueError("mspeaks: Invalid relative height filter")


    #Create thread for printing text in multiprocessing
    manager = Manager()
    print_queue = Manager().Queue()
    queue = manager.Queue()
    queue.put(True)
    t = Thread(target=printer,args=[print_queue])  
    t.start()
    if isinstance(data_obj_path,str):
        data_obj_path=[data_obj_path]
    # Определение количества пула процессов
    cpu_num = cpu_count()-free_cores
    Ram_GB = Ram_GB*1e+9
    batch_bsize = Ram_GB/cpu_num
    if dtypeconv =='single':
        bytes_flsize = 4
    elif dtypeconv == 'double':
        bytes_flsize = 8
    elif dtypeconv == 'half':
        bytes_flsize = 2
    ##
    chunk_size = np.ceil(h5chunk_size_MB*1e+6/(bytes_flsize*11))

    ###I. Finding slide directory for rawdata of samples (imzml)
    path_list=find_imzml_roots(data_obj_path)

    ###I Finding slide directory with rawdata of samples (path_list) - DONE
    if noise_est == "MAD":
        noise_func= MAD
    elif noise_est == "std":
        noise_func=np.std
    
    # Processing
    for file_path in path_list.keys():
        slide = os.path.basename(file_path)  
        print(f"The {slide} raw spectra data is on progress.")
        
        data_obj_coord={}
        sample_list = path_list[file_path]

        ###II. Extracting spectra coordinates, roi indexes, other metadata for proccessing slide samples from poslog and _info text files. Input arguments organization for "intprocc_parbatched_imzml" function (spectra processing batched and parallelized)
        ###II. Вытаскивание данных координат точек и индексов для областей(roi), если их несколько, из poslog и _info. Организация входных параметров для функции "intprocc_parbatched_imzml", включая batch'ей для параллельной выгрузки
        print(f"Slide's {slide} spectra coordinates and metadata extraction for preparation parallel proccessing")
        with Pool(cpu_num) as p:
            data_obj_temp = p.starmap(poslog_parbatched,list(product(sample_list,[batch_bsize],[dtypeconv],[print_queue],[cpu_num],[resample_to_dots])))
        p.join()    
        args_batches=[]
        for data in data_obj_temp:
            data_obj_coord=data_obj_coord|data[0]
            args_batches = args_batches + data[1]
        del data_obj_temp
        gc.collect()
        #### Подготовка аргументов для процессинга данных
        args2procc = Manager().dict({"baseliner_algo": baseliner_algo, "params2baseliner_algo": params2baseliner_algo,"params2align":params2align, "align_peaks":align_peaks,"weights_list":weights_list, "dots_shift":max_shift_mz,"smooth_algo":smooth_algo,"smooth_window":smooth_window, "smooth_cycles":smooth_cycles})
        args2peakpicking = Manager().dict({"oversegmentationfilter": oversegmentationfilter, "fwhhfilter": fwhhfilter, "heightfilter": heightfilter,"rel_heightfilter":rel_heightfilter, "peaklocation": peaklocation,
                        "SNR_threshold": SNR_threshold,"noise_func": noise_func, "noise_est_iterations": noise_est_iterations, "print_queue":print_queue})
            ## Определение байтового размера одной точки
        args_batches=list(args_batch+(resample_to_dots,args2procc,args2peakpicking,queue, chunk_size) for args_batch in args_batches)
        ##II. Coordinates, metadata and organization of input arguments for parallelized and batched proccessing of spectra - DONE
        ## Deleting old hdf5 file if it has dataset with rawpeaklist
        base_path = os.path.join(file_path,slide)
        if rewrite: # if TRUE - delete hdf5
            try:
                os.remove(base_path+"_specdata.hdf5")
            except:
                pass
        ## 
        if os.path.isfile(base_path+"_specdata.hdf5"): # if hdf5 exist
            data_obj_feat = File(base_path+"_specdata.hdf5","r")

            #try:
            flag_feat = False
            flag_mzint = True
            
            for sample in data_obj_feat.keys():
                for roi in data_obj_feat[sample].keys():
                    
                    try:
                        data_obj_feat[sample][roi]['int']
                        #print('tryed',sample,roi)
                    except:
                        data_obj_feat.close()
                        flag_mzint = False
                        os.remove(base_path+"_specdata.hdf5")
                        #print("deleted")
                        break
                    try:
                        data_obj_feat[sample][roi]['peaklists']
                        flag_feat = True
                    except:
                        pass
                    
                else:
                    continue
                
                break
            if flag_mzint:
                data_obj_feat.close()
                #print("closed")
            
            if flag_feat and flag_mzint:
                
                if os.path.exists(base_path+"new.hdf5"):
                    os.remove(base_path+"new.hdf5")
                data_obj_feat_new = File(base_path+"new.hdf5","a")
                data_obj_feat = File(base_path+"_specdata.hdf5","r")
                for sample in data_obj_feat.keys():
                    for roi in data_obj_feat[sample].keys():    
                        data_obj_feat_new.create_dataset(sample+"/" + roi + "/" + "mz",data=data_obj_feat[sample][roi]["mz"][:], chunks = data_obj_feat[sample][roi]["mz"].chunks) 
                        data_obj_feat_new.create_dataset(sample+"/" + roi + "/" + "int",data=data_obj_feat[sample][roi]["int"][:], chunks = data_obj_feat[sample][roi]["int"].chunks)
                data_obj_feat_new.close()
                data_obj_feat.close()
                #print("repacked")
                os.remove(base_path+"_specdata.hdf5")
                os.rename(base_path+"new.hdf5",base_path+"_specdata.hdf5")
                
            data_obj_feat = File(base_path+"_specdata.hdf5","a")

        else: # if hdf5 doesn't exist
            data_obj_feat = File(base_path+"_specdata.hdf5","a")
        ## Deleting old hdf5 file - Done
        data_obj_feat.close()

        ##III. Writing coordinates
        print(f"Slide's {slide} spectra coordinates writing")
        hdf5_coords(file_path,slide,data_obj_coord,chunk_size)
        ##IV. Proccessing, peakpicking and writing to hdf5
        print(f"Slide's {slide} spectra parallel proccessing")

        print_queue.put(len(args_batches))
        with Pool(cpu_num) as p:
            p.starmap(int2proc2peaklist_parbatched,args_batches)
        p.join()
        ##IV. Proccessing, peakpicking and writing to hdf5 - Done
        data_obj_feat = File(base_path+"_specdata.hdf5","r")
        if draw: #Секция для отрисовки полученных результатов
            args_batches=pd.DataFrame(args_batches)
            for sample in data_obj_coord.keys():
                for roi in data_obj_coord[sample].keys():
                    
                    dataf = pd.DataFrame(data_obj_feat[sample][roi]["peaklists"][:].T, data_obj_feat[sample][roi]["peaklists"].attrs["Column headers"]).T
                    dataf = dataf.astype({"spectra_ind": int})
                    
                    plt.figure().set_figwidth(25)
                    plt.gcf().set_figheight(5)

                    idx_start, numspec = data_obj_coord[sample][roi]["idxroi"]
                    raw = ImzMLParser(data_obj_coord[sample][roi]["source"])
                    idx_spec = np.random.randint(idx_start,idx_start+numspec)
                    print(f'Spectrum number: {idx_spec}')
                    data_mz_old, data_int_old = raw.getspectrum(idx_spec)
                    roi_idx_spec = idx_spec-idx_start
                    loc_args2procc={"baseliner_algo": baseliner_algo, "params2baseliner_algo": params2baseliner_algo,"params2align":params2align, "align_peaks":align_peaks,"weights_list":weights_list,"smooth_algo":smooth_algo, "smooth_cycles":smooth_cycles}
                    if resample_to_dots:
                        min_mz, max_mz = args_batches.loc[(args_batches.loc[:,1]==sample) & (args_batches.loc[:,2]==roi)].iloc[0,8]
                        data_mz = np.array(list(np.linspace(min_mz,max_mz,resample_to_dots)))
                        if loc_args2procc["params2align"]["only_shift"]:
                            dots_shift = int(max_shift_mz/(np.median(np.diff(data_mz))))
                        else:
                            dots_shift = max_shift_mz
                        loc_args2procc['dots_shift']=dots_shift
                        loc_args2procc["smooth_window"]=int(smooth_window/(np.median(np.diff(data_mz))))
                        data_int = DataProc_resample1d(data_int_old,data_mz_old,data_mz,Baseline(data_mz),**loc_args2procc)
                    else:
                        data_mz = data_mz_old
                        if loc_args2procc["params2align"]["only_shift"]:
                            dots_shift = int(max_shift_mz/(np.median(np.diff(data_mz))))
                        else:
                            dots_shift = max_shift_mz
                        loc_args2procc['dots_shift']=dots_shift
                        loc_args2procc["smooth_window"]=int(smooth_window/(np.median(np.diff(data_mz))))
                        data_int = DataProc_base1d(data_int_old,data_mz,Baseline(data_mz),**loc_args2procc)
                
                    if mz_diap4draw:

                        diapold=(np.array(data_mz_old>mz_diap4draw[0]) & np.array(data_mz_old<mz_diap4draw[1]))
                        diap = (np.array(data_mz>mz_diap4draw[0]) & np.array(data_mz<mz_diap4draw[1]))
                        dataf.query("mz>@mz_diap4draw[0] and mz<@mz_diap4draw[1] and spectra_ind==@roi_idx_spec").plot(x="mz",y="Intensity",ax = plt.gca(), style = "x")
                        startg = mz_diap4draw[0]
                        endg = mz_diap4draw[1]
                    else:
                        diapold=range(len(data_mz_old))
                        diap = range(len(data_mz))
                        startg = min(data_mz)
                        endg = max(data_mz)
                        dataf.query("spectra_ind==@roi_idx_spec").plot(x="mz",y="Intensity",ax = plt.gca(), style = "x")
                    

                    plt.plot(dataf.query("PextL>@startg and PextL<@endg and spectra_ind==@roi_idx_spec")['PextL'],
                            [0]*len(dataf.query("PextL>@startg and PextL<@endg and spectra_ind==@roi_idx_spec")['PextL']),'v')
                    plt.plot(dataf.query("PextR>@startg and PextR<@endg and spectra_ind==@roi_idx_spec")['PextR'],
                            [0]*len(dataf.query("PextR>@startg and PextR<@endg and spectra_ind==@roi_idx_spec")['PextR']),'^')
                    plt.plot(data_mz_old[diapold], data_int_old[diapold])
                    plt.plot(data_mz[diap], data_int[diap])
                    plt.grid(visible = True, which="both")
                    plt.title(f'Sample: {sample}, roi: {roi}')
                    plt.legend(['Peaks', 'Peak`s left base', 'Peak`s right base', 'Original spectrum','Processed spectrum'])
                    plt.minorticks_on()
                    plt.xlim((startg,endg))
                    plt.show()
                    del diapold
                
       
        print_queue.put(0) #closing old tqdm bar
        data_obj_feat.close()            
        with open(os.path.join(file_path,"Processing_settings.txt"), "w") as file:
            file.write("###Raw spectra proccessing\n##Spectra aligning(msalign function):\n")
            file.write(f"Aligning peaks list:{align_peaks}\nPeaks weights: {weights_list}\nMax shift in mz scale: {max_shift_mz}\nOther align parametres:")
            for key in params2align.keys():
                file.write(f"{key}: {params2align[key]}\n")
            file.write(f"##Resampling data (interp1d function):\nResampling to dots:{resample_to_dots}.\n")
            file.write(f"Previous data in dots:\n")
            for sample in data_obj_coord.keys():
                for roi in data_obj_coord[sample].keys():
                    file.write(f'   for {sample} roi {roi}: ')
                    sourced=ImzMLParser(data_obj_coord[sample][roi]['source'])
                    if data_obj_coord[sample][roi]['continuous']:
                        data_mz_old=sourced.getspectrum(0)[0]
                        file.write(f'continious data with {len(data_mz_old)} number of dots between {data_mz_old[0]} and {data_mz_old[-1]} m/z')
                    else:
                        file.write(f'proccessed data with median {np.median(sourced.mzLengths)} number of dots')

            file.write(f"##Baseline correction (pybaseline package, class Baseline)):\nAlgorithm:{baseliner_algo}\n")
            for key in params2baseliner_algo.keys():
                file.write(f"{key}: {params2baseliner_algo[key]}\n")
            file.write(f"##Smoothing (smoothing function):\nSmooth algorithm: {smooth_algo}\nWindow {smooth_window}\nCycles {smooth_cycles}\n")    
            file.write(f"##Other parametres:\nBatch size: {batch_bsize/1000000} Mb\nData type convertion to: {dtypeconv}\n")  
            file.write("\n\n##Peak picking\n")
            if isinstance(oversegmentationfilter,str):
                file.write(f"Oversegmentation filter is local median FWHH\n")
            else:
                file.write(f"Oversegmentation filter: {oversegmentationfilter}\n")
            file.write(f"Full width at half height (FWHH) filter: {fwhhfilter} mz\n")    
            file.write(f"Absolute height threshold: {heightfilter}\nRelative height threshold: {rel_heightfilter}\n")
            file.write(f"\n##Noise estimation by std or MAD\nIterations of noise estimation: {noise_est_iterations}\nMAD or std: {noise_est}\nPeaks SNR threshold filtration: {SNR_threshold}\n")
            file.write(f"\n##Other parametres:\nBatch size: {batch_bsize/1000000} Mb\nData type convertion to: {dtypeconv}\n")    

    # Закрытие hdf5 объёктов
    # Closing threads and hdf5 object

    print_queue.put(Sentinel())

    t.join()

    gc.collect()
    return

def proc2peaklist(data_obj_path, oversegmentationfilter = 0, fwhhfilter = 0, heightfilter=0.5, peaklocation=1,rel_heightfilter=0,
              SNR_threshold = 3.5, noise_est = "std",noise_est_iterations = 3,
              draw = True, mz_diap4draw = None, Ram_GB = 1, h5chunk_size_MB = 10,dtypeconv='single',
              free_cores=1):
    """
    Общее описание
    ----
    Функция для получения пиклистов из обработанных спектров и после их записи в файл hdf5 под названием "Slidename_specdata.hdf5" в датасет "peaklists". Конечным результатом является полный пиклист-матрица имаджа.

    :param data_obj_path: list of paths to root folders where to search imzml files in subfolders 

    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `0`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `0`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `0`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `0`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `3.5`
    :param noise_est: алгоритм оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `"std"`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    :param draw: Draw example graphs of raw and proccessed random spectrum of image. Default: `True`
    :param mz_diap4draw: Range for graphs draw. Default: `None`
    :param rewrite: delete old hdf5 before writing new spectra data. Default: `False`
    :param Ram_GB: Determine max sizes in GB of the data proccesing on CPU cores at moment.
    :param h5chunk_size_MB: the chunk size for hdf5 datasets writing in MB
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param free_cores: Number of CPU cores don't used in multiproccessing

    :type data_obj_path: `list`
    :type oversegmentationfilter: `float`
    :type fwhhfilter: `float`
    :type heightfilter: `float`
    :type peaklocation: `float` and =<1
    :type rel_heightfilter: `float`
    :type SNR_threshold: `float`
    :type noise_est: {`"std"`,`"mad"`}
    :type noise_est_iterations: `int`
    :type draw: `bool`
    :type mz_diap4draw: `list` or `None`
    :type rewrite: `bool`
    :type Ram_GB: `float`
    :type h5chunk_size_MB: `float`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type free_cores: `1`

    :return: `None`
    :rtype: `NoneType`
    """

    manager = Manager()
    print_queue = Manager().Queue()
    queue = manager.Queue()
    queue.put(True)
    t = Thread(target=printer,args=[print_queue])
    t.start()
    if isinstance(data_obj_path,str):
        data_obj_path=[data_obj_path]
    # Определение количества пула процессов
    cpu_num = cpu_count()-free_cores
    Ram_GB = Ram_GB*1e+9
    h5chunk_size_MB=h5chunk_size_MB*1e+6
    batch_bsize = Ram_GB/cpu_num
    if dtypeconv =='single':
        bytes_flsize = 4
    elif dtypeconv == 'double':
        bytes_flsize = 8
    elif dtypeconv == 'half':
        bytes_flsize = 2
    ##
    chunk_size = np.ceil(h5chunk_size_MB/(bytes_flsize*11))

    if noise_est == "MAD":
        noise_func= MAD
    elif noise_est == "std":
        noise_func=np.std
    args2peakpicking = Manager().dict({"oversegmentationfilter": oversegmentationfilter, "fwhhfilter": fwhhfilter, "heightfilter": heightfilter,"rel_heightfilter":rel_heightfilter, "peaklocation": peaklocation,
                        "SNR_threshold": SNR_threshold,"noise_func": noise_func, "noise_est_iterations": noise_est_iterations, "print_queue":print_queue})

    
    path_list=find_paths(data_obj_path,file_end = '_specdata.hdf5')
    # path_list=[]
    # for path in data_obj_path:
    #     for root, dirs, files in os.walk(path):
    #         for file in files: 
    #             if file.endswith('_specdata.hdf5'):
    #                 path_list.append(os.path.join(root,file))
                #if file.endswith('.imzML'):
                #    imzml_list.append(os.path.join(root,file))
    
    for file_path in path_list:        
        ##II. Coordinates, metadata and organization of input arguments for parallelized and batched proccessing of spectra - DONE

        ## Deleting old hdf5 file if it has dataset with peaklists
        directory_path = os.path.dirname(file_path)
        slide = os.path.basename(file_path).replace('_specdata.hdf5','')
        print(f"The {slide} processed spectra data is loaded from the hdf5 file.")
        data_obj_feat = File(file_path,"r")
        
        flag_feat = False
        flag_mzint = True
        for sample in data_obj_feat.keys():
            for roi in data_obj_feat[sample].keys():
                
                try:
                    data_obj_feat[sample][roi]['int']
                    #print('tryed',sample,roi)
                except:
                    data_obj_feat.close()
                    print(f"Spectra of at least one of the imaging is not recorded in hdf5. Processing aborted on slide {slide}, sample {sample}, roi {roi}.")
                    return
                try:
                    data_obj_feat[sample][roi]['peaklists']
                    flag_feat = True
                except:
                    pass

        if flag_mzint:
            data_obj_feat.close()
            #print("closed")
        if flag_feat and flag_mzint:
            if os.path.exists(os.path.join(file_path,slide)+"new.hdf5"):
                os.remove(os.path.join(file_path,slide)+"new.hdf5")
            data_obj_feat_new = File(os.path.join(directory_path,slide)+"new.hdf5","a")
            data_obj_feat = File(file_path,"r")
            for sample in data_obj_feat.keys():
                for roi in data_obj_feat[sample].keys():    
                    data_obj_feat_new.create_dataset(sample+"/" + roi + "/" + "mz",data=data_obj_feat[sample][roi]["mz"][:], chunks = data_obj_feat[sample][roi]["mz"].chunks) 
                    data_obj_feat_new.create_dataset(sample+"/" + roi + "/" + "int",data=data_obj_feat[sample][roi]["int"][:], chunks = data_obj_feat[sample][roi]["int"].chunks)
                    data_obj_feat_new.create_dataset(sample+"/" + roi + "/" + "xy",data=data_obj_feat[sample][roi]["xy"][:], chunks = data_obj_feat[sample][roi]["xy"].chunks)
                    data_obj_feat_new[sample][roi].attrs['source']=data_obj_feat[sample][roi].attrs['source']
                    data_obj_feat_new[sample][roi].attrs['continuous']=data_obj_feat[sample][roi].attrs['continuous']
                    data_obj_feat_new[sample][roi].attrs['idxroi']=data_obj_feat[sample][roi].attrs['idxroi']
                    try:
                        data_obj_feat_new.create_dataset(sample+"/" + roi + "/" + "z",data=data_obj_feat[sample][roi]["z"][:], chunks = data_obj_feat[sample][roi]["z"].chunks)
                    except:
                        pass
            data_obj_feat_new.close()
            data_obj_feat.close()
            #print("repacked")
            os.remove(file_path)
            os.rename(os.path.join(file_path,slide)+"new.hdf5",file_path)
        
        data_obj_feat = File(file_path,"r")
        
        ##III. Peakpicking and writing to hdf5
        args_batches=[]
        print(f"Slide's {slide} spectra parallel peak picking")
        ## Ram managment
        num_of_processes_works=0
        for sample in data_obj_feat.keys():
            for roi in data_obj_feat[sample].keys():
                try:
                    
                    roi_chunks = len(list(data_obj_feat[sample][roi]["int"].iter_chunks()))
                    chunk_size_MB = data_obj_feat[sample][roi]["int"].nbytes/roi_chunks
                    num_batch4chunks = int(np.ceil(data_obj_feat[sample][roi]["int"].nbytes/Ram_GB))
                    iter_chunks=[None]*roi_chunks
                    for n,chunk in enumerate(data_obj_feat[sample][roi]["int"].iter_chunks()):
                        iter_chunks[n]=chunk[0]
                except:
                    roi_chunks = int(np.ceil(data_obj_feat[sample][roi]["int"].nbytes/h5chunk_size_MB))+1
                    chunk_size_MB=h5chunk_size_MB
                    if roi_chunks <cpu_num*2:
                        chunk_size_MB=h5chunk_size_MB*roi_chunks/cpu_num*2
                        roi_chunks=cpu_num*2+1
                    if np.ceil(Ram_GB/chunk_size_MB)<cpu_num or np.ceil(data_obj_feat[sample][roi]["int"].nbytes/chunk_size_MB)<cpu_num:
                        chunk_size_MB=[Ram_GB/cpu_num]
                        chunk_size_MB.append(data_obj_feat[sample][roi]["int"].nbytes/cpu_num)
                        roi_chunks = int(np.ceil(data_obj_feat[sample][roi]["int"].nbytes/min(chunk_size_MB)))+1

                    num_batch4chunks = np.ceil(data_obj_feat[sample][roi]["int"].nbytes/Ram_GB)             
                    iter_chunks = list(slice(*x) for x in list(pairwise(np.linspace(0,data_obj_feat[sample][roi]["int"].shape[0],roi_chunks,dtype=int))))
                num_of_processes_works =num_of_processes_works+len(iter_chunks)
                batches = (np.array_split(np.array(iter_chunks),num_batch4chunks))
                args_batches += list(product(batches,[sample],[roi]))
        data_obj_feat.close()
        print_queue.put(num_of_processes_works)
        with Pool(cpu_num) as p:
        
            for args_batch in args_batches:

                sample=args_batch[1]
                roi = args_batch[2]
                args_batch = list(product(args_batch[0],[sample],[roi],[file_path],[args2peakpicking],[dtypeconv],[print_queue]))
                results = np.vstack(p.starmap(proc2peaklist_parbatched,args_batch))
                with File(file_path,'a') as data_obj_feat:
                    
                    try:
                        
                        start_row = data_obj_feat[sample][roi]["peaklists"].shape[0]
                        npeaks = results.shape[0]
                        data_obj_feat[sample][roi]["peaklists"].resize(start_row + npeaks ,0)
                        data_obj_feat[sample][roi]["peaklists"][start_row:(start_row+ npeaks),:] = results
                    except:
                        data_obj_feat.create_dataset(sample + "/" + roi + "/peaklists",results.shape, maxshape = (None, 11), chunks=(chunk_size, 11))

                        data_obj_feat[sample][roi]["peaklists"][:] = results
                        data_obj_feat[sample][roi]["peaklists"].attrs["Column headers"] = ["spectra_ind","mz","Intensity","Area","SNR","PextL","PextR","FWHML","FWHMR","Noise","Mean noise"]

                    data_obj_feat.close()
                
        p.join()
        ##III. Peakpicking and writing to hdf5 - Done
        ##IV. Writing results
        print(f"Slide's {slide} spectra writing feature results")
        data_obj_feat = File(file_path,"a")
        for sample in data_obj_feat.keys():
            for roi in data_obj_feat[sample].keys():

                #print(f"sample{sample} roi {roi} num spec{max(data_obj_feat[sample][roi]["peaklists"][:,0])} COORDS NUM {data_obj_coord[sample][roi]["xy"].shape}")
                if draw: #Секция для отрисовки полученных результатов
                    
                    dataf = pd.DataFrame(data_obj_feat[sample][roi]["peaklists"][:].T, data_obj_feat[sample][roi]["peaklists"].attrs["Column headers"]).T
                    dataf = dataf.astype({"spectra_ind": int})

                    
                    plt.figure().set_figwidth(25)
                    plt.gcf().set_figheight(5)
                    idx_start = data_obj_feat[sample][roi].attrs['idxroi'][0]
                    nspec= data_obj_feat[sample][roi].attrs['idxroi'][1]
                    path2imzml = data_obj_feat[sample][roi].attrs['source']

                    num_spec = np.random.randint(idx_start,idx_start+nspec)
                    print(f'Spectrum number: {num_spec}')
                    roi_num_spec = num_spec-idx_start
                    data_mz_old, data_int_old = ImzMLParser(path2imzml).getspectrum(num_spec)
                    data_int_old.shape = (data_int_old.shape[0],1)

                    
                    data_mz = data_obj_feat[sample][roi]['mz'][:]
                    data_int = data_obj_feat[sample][roi]['int'][roi_num_spec,:]
                
                    if mz_diap4draw is not None:

                        diapold=(np.array(data_mz_old>mz_diap4draw[0]) & np.array(data_mz_old<mz_diap4draw[1]))
                        diap = (np.array(data_mz>mz_diap4draw[0]) & np.array(data_mz<mz_diap4draw[1]))
                        dataf.query("mz>@mz_diap4draw[0] and mz<@mz_diap4draw[1] and spectra_ind==@roi_num_spec").plot(x="mz",y="Intensity",ax = plt.gca(), style = "x")
                        startg = mz_diap4draw[0]
                        endg = mz_diap4draw[1]
                    else:
                        diapold=range(len(data_mz_old))
                        diap = range(len(data_mz))
                        startg = min(data_mz)
                        endg = max(data_mz)
                        dataf.query("spectra_ind==@roi_num_spec").plot(x="mz",y="Intensity",ax = plt.gca(), style = "x")
                    

                    plt.plot(dataf.query("PextL>@startg and PextL<@endg and spectra_ind==@roi_num_spec")['PextL'],
                            [0]*len(dataf.query("PextL>@startg and PextL<@endg and spectra_ind==@roi_num_spec")['PextL']),'v')
                    plt.plot(dataf.query("PextR>@startg and PextR<@endg and spectra_ind==@roi_num_spec")['PextR'],
                            [0]*len(dataf.query("PextR>@startg and PextR<@endg and spectra_ind==@roi_num_spec")['PextR']),'^')
                    plt.plot(data_mz_old[diapold], data_int_old[diapold])
                    plt.plot(data_mz[diap], data_int[diap])
                    plt.grid(visible = True, which="both")
                    plt.title(f'Sample: {sample}, roi: {roi}')
                    plt.legend(['Peaks', 'Peak`s left base', 'Peak`s right base', 'Original spectrum','Processed spectrum'])
                    plt.minorticks_on()
                    plt.xlim(mz_diap4draw)
                    plt.show()
                    del diapold
        try:
            with open(os.path.join(directory_path,"Processing_settings.txt"), "r") as file:
                text = file.read()
        except:
            text = ''
            pass
        with open(os.path.join(directory_path,"Processing_settings.txt"), "w") as file:
            file.write(text[:text.find("\n\n##Peak picking")])
            file.write("\n\n##Peak picking\n")
            if isinstance(oversegmentationfilter,str):
                file.write(f"Oversegmentation filter is local median FWHH")
            else:
                file.write(f"Oversegmentation filter: {oversegmentationfilter}")
            file.write(f"Full width at half height (FWHH) filter: {fwhhfilter} mz")      
            file.write(f"Absolute height threshold: {heightfilter}\nRelative height threshold: {rel_heightfilter}\n")
            file.write(f"\n##Noise estimation by std or MAD\nIterations of noise estimation: {noise_est_iterations}\nMAD or std: {noise_est}\nPeaks SNR threshold filtration: {SNR_threshold}\n")
            file.write(f"\n##Other parametres:\nBatch size: {batch_bsize/1000000} Mb\nData type convertion to: {dtypeconv}\n")
        print_queue.put(0) # закрываем tqdm
    gc.collect()
    # Закрытие hdf5 объёктов
    # Closing threads and hdf5 object
    data_obj_feat.close()
    print_queue.put(Sentinel())
    t.join()
    
    return

### Utility functions for multiprocessing
class Sentinel: 
    """Заглушка для прекращения цикла while в функции printer"""
    pass

def printer(print_queue):
    '''
    Вспомогательная функция используется для отображения сообщений в дочерних процессах, включая отображение прогресса с помощью tqdm.
    Еcли с помощью put на входе str, то печатает сообщение, если на входе число первый раз, то создаёт tqdm объект и отображает прогресс, если повторно число - то удаляет tqdm объект, если True, то отображает продвижение процесса.
    :param print_queue: is a multiproccesing.Manager.Queue() proxy object
    :type print_queue: multiproccesing.Manager.Queue() proxy object
    '''
    while True:
        msg = print_queue.get()
        if isinstance(msg, Sentinel):
            try:
                pbar.close()
            except:
                pass
            break
        elif isinstance(msg,str):
            print(msg, flush=True)
        elif msg is True:
            pbar.update(1)
        else:
            try:
                pbar.close()
                del pbar
            except:
                pbar = tqdm(total = msg,desc="Batches progress",smoothing = 0.005)       

def hdf5_writer(foldersample_path, queue,print_queue, dtypeconv,chunk_rowsize,chunk_bsize):
    """
    Общее описание
    ----
    Вспомогательная функция для imzml2hdf5
    
    :param foldersample_path: list of str with paths `imzML` file
    :param queue: Менеджер для задержки работы процесса перед записью данных датасета. Необходимо для однопоточной записи данных, так как запись данных в hdf5 невозможна в мультипоточном режиме
    :param print_queue: Менеджер для печати сообщений на экран с процесса.
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    :param chunk_rowsize: chunking hdf5 datasets for partial and efficient loading data to RAM. The default is `"Auto"`

        `"Auto"` - автоматический подбор кол-ва строк записи матрицы в hdf5 на основе размера chunk_bsize,

        `"Full"` - датасет сохраняется и выгружается целиком (занимает много RAM), 

        `int` - датасет дробится по заданному числу строк, где каждая строка это данные спектра.
    :param chunk_bsize: the chunk size for hdf5 datasets in bytes. Optional, works only with chunk_rowsize =`"Auto"`. The default is `10000000`
    
    :type foldersample_path: list
    :type queue: Manager.Queue()
    :type print_queue: Manager.Queue()
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}
    :type chunk_rowsize: {`"Auto"`,`"Full"`, `int`}
    :type chunk_bsize: `int`, optional

    :return: None
    :rtype: Nonetype
    """
    Slide_folder_path = foldersample_path[0]
    Slide_name = os.path.basename(Slide_folder_path)
    sample_path2imzml = foldersample_path[1]
    folder_path2imzml = os.path.dirname(sample_path2imzml)
    sample_name = os.path.splitext(os.path.basename(sample_path2imzml))[0]

    ## Извлечение из poslog физических координат
    sample_data={}
    count=0
    idx_first=0
    roi_idx = {}
    
    try:
        sample_imzml=ImzMLParser(sample_path2imzml)
    except FileNotFoundError: #Если нет imzML файла в папке - пропуск
        print_queue.put(f'No {sample_path2imzml} file in directory {Slide_folder_path}')
        return
    
    try:
        dcont = sample_imzml.metadata.pretty()["file_description"]["continuous"]
    except KeyError:
        dcont = not sample_imzml.metadata.pretty()["file_description"]["processed"]

    poslog_specdata = [None]*len(sample_imzml.coordinates) #Данные строк в poslog с записью roi и координат снятого спектра.
    roi_list = []
    dots_num={}
    try:
        with open(os.path.join(folder_path2imzml,sample_name)+"_poslog.txt") as f:
            data = f.readlines()
            
            ##первая итерация записи координат начиная с третьей строки
            coords =  data[2].split(' ')
            roi_num = re.search('R(.+?)X', data[2]).group(1)
            roi_list.append(roi_num)
            poslog_specdata[count]=(roi_num,float(coords[-3]), float(coords[-2]))
            
            sample_data[roi_num]={}
            roi_idx[roi_num] = idx_first
            
            sample_data[roi_num]["mz"] = np.array(sample_imzml.getspectrum(roi_idx[roi_num])[0],dtype=dtypeconv)
            sample_data[roi_num]["z"] = np.array(float(coords[-1]),dtype=dtypeconv)
            
            count+=1
            ## продолжение итераций    
            for i in range(2,len(data)-1):
                coords =  data[i+1].split(' ')
                
                if(coords[-4]!='__'):
                    roi_num = re.search('R(.+?)X', data[i+1]).group(1)
                    poslog_specdata[count]=(roi_num,float(coords[-3]), float(coords[-2]))
                    
                    if roi_num not in roi_list[-1]:
                        roi_list.append(roi_num)
                        sample_data[roi_num]={}
                        roi_idx[roi_num] = []
                        roi_idx[roi_list[-2]] = (idx_first, count-idx_first)
                        
                        idx_first=count
                        sample_data[roi_num]["mz"] = np.array(sample_imzml.getspectrum(roi_idx[roi_num])[0],dtype=dtypeconv)
                        sample_data[roi_num]["z"] = float(coords[-1])
                    
                    count +=1
            roi_idx[roi_num] = (idx_first, count-idx_first)
            
            ## Preallocating координаты и int
            for roi in roi_list:
                dots_num[roi] = len(sample_data[roi]["mz"])
                sample_data[roi]["xy"] = np.empty((roi_idx[roi][1],2))
                sample_data[roi]["int"] = np.empty((roi_idx[roi][1],dots_num[roi]), dtype=dtypeconv)
                ### Определим Chunksize при автоматическом определениии размера
                if chunk_rowsize == "Auto":
                    if dtypeconv == 'single':
                        bytes_flsize = 4
                    elif dtypeconv == 'double':
                        bytes_flsize = 8
                    elif dtypeconv == 'half':
                        bytes_flsize = 2
                    chunk_rowsize = np.ceil(chunk_bsize/(bytes_flsize*dots_num[roi]))
                ###
            
            ##
            ## Заполнение пустых матриц координат и int 
            for idx, (roi,x,y)  in enumerate(poslog_specdata):
                sample_data[roi]["xy"][idx,:] = [x, y]
                sample_data[roi]["int"][idx-roi_idx[roi][0],:] = sample_imzml.getspectrum(idx)[1]
        del coords, idx_first
    except FileNotFoundError:
        roi_num = "00"
        roi_list.append(roi_num)
        sample_data[roi_num]={}
        
        numspectra = len(sample_imzml.coordinates)

        sample_data[roi_num]["xy"] = np.empty((numspectra,2))
        
        if dcont:
            sample_data[roi_num]["mz"] = np.array(sample_imzml.getspectrum(0)[0],dtype=dtypeconv)
            dots_num[roi_num] = sample_data[roi_num]["mz"].shape[1]
            sample_data[roi_num]["int"] = np.empty((numspectra,dots_num[roi_num]), dtype=dtypeconv)
            for idx in range(numspectra):
                sample_data[roi_num]["xy"][idx,:] = sample_imzml.get_physical_coordinates(idx)
        else:
            print_queue.put(f"Sample: {sample_path2imzml}\nThe data in the imzml file is not continuous. It will not be recorded in HDF5 format.")
            print_queue.put(True)
            return # Заглушка. Нет идей как грамотно впихнуть данные в hdf5, где надо пихать матрицы, а не листы с произвольным размером 
            sample_data[roi_num]["mz"] = [0]*numspectra
            sample_data[roi_num]["int"] = [0]*numspectra
            for idx in range(numspectra):
                sample_data[roi_num]["mz"][idx],sample_data[roi_num]["int"][idx] = sample_imzml.getspectrum(idx)
                sample_data[roi_num]["xy"][idx,:] = sample_imzml.get_physical_coordinates(idx)

            sample_data[roi_num]["mz"]=np.array(list(zip_longest(*sample_data[roi_num]["mz"], fillvalue=0))).astype(dtypeconv).T
            sample_data[roi_num]["int"]=np.array(list(zip_longest(*sample_data[roi_num]["int"], fillvalue=0))).astype(dtypeconv).T
            dots_num[roi_num] = sample_data[roi_num]["mz"].shape[1]
        sample_data[roi_num]["z"] = sample_imzml.coordinates[0][-1]

    if chunk_rowsize == "Auto":
        if dtypeconv =='single':
            bytes_flsize = 4
        elif dtypeconv == 'double':
            bytes_flsize = 8
        elif dtypeconv == 'half':
            bytes_flsize = 2
        chunk_rowsize = chunk_bsize/(bytes_flsize*dots_num[roi_num])
    del roi_num, dots_num
    ##
    
    
    ## Автоматическое определение имени датасета
    folder_name=os.path.basename(folder_path2imzml)
    if sample_name == folder_name:
        ds_name = folder_name
    else:
        ds_name=folder_name+"_"+sample_name
    ##
    ## Запись в hdf5

    temp = queue.get()
    print_queue.put(f"Slide {Slide_name} with sample {sample_name} is waiting queue")

    hdf5_raw=File(os.path.join(Slide_folder_path,Slide_name)+"_rawdata.hdf5","a")

    if chunk_rowsize == "Full":
        for roi in roi_list:
            print_queue.put(f"Slide {Slide_name} with sample {sample_name}"+" roi "+roi+" data writing is in progress")
            #for type in ['/xy','/z']:
                #hdf5.create_dataset(ds_name+'/'+roi+type, data=sample_data[roi][type.replace("/","")])

            for type in  ['/mz','/int']:
                hdf5_raw.create_dataset(ds_name+'/'+roi+type, data=sample_data[roi][type.replace("/","")])
            #hdf5[ds_name][roi].attrs['continues'] =dcont #Data points type    
            hdf5_raw[ds_name][roi].attrs['continues'] =dcont #Data points type       
        #hdf5.close()
        hdf5_raw.close()
    else:
        for roi in roi_list:
            print_queue.put(ds_name+" roi "+roi+" data writing is in progress")
            
                ### chunked version
            
            hdf5_raw.create_dataset(ds_name+'/'+roi+'/int', data=sample_data[roi]['int'],chunks=(chunk_rowsize,sample_data[roi]['int'].shape[1]))
            hdf5_raw.create_dataset(ds_name+'/'+roi+'/xy', data=sample_data[roi]['xy'],chunks=(chunk_rowsize,sample_data[roi]['xy'].shape[1]))
            #hdf5.create_dataset(ds_name+'/'+roi+'/xy', data=sample_data[roi]['xy'],chunks=(chunk_rowsize,sample_data[roi]['xy'].shape[1]))
                ###
            hdf5_raw.create_dataset(ds_name+'/'+roi+'/mz', data=sample_data[roi]['mz'])
            #hdf5.create_dataset(ds_name+'/'+roi+'/z', data=sample_data[roi]['z'])
            hdf5_raw.create_dataset(ds_name+'/'+roi+'/z', data=sample_data[roi]['z'])
            #hdf5[ds_name][roi].attrs['continues'] = dcont #Data points type    
            hdf5_raw[ds_name][roi].attrs['continues'] =dcont #Data points type       
        #hdf5.close()
        hdf5_raw.close()
    print_queue.put(f"{sample_path2imzml} data writing is finished")
    print_queue.put(True)
    queue.put(True)

def int2procc_parbatched(sample_file_path, sample,roi,interval,dots_num,dtypeconv,dcont,print_queue, discon_resample_range = (None,None),resample_to_dots = None,
                        args2procc={},queue = None, chunk_size = 100):
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой обработки спектров. Используется в функции Raw2proc.

    :param sample_file_path: path to spectra source `imzML`
    :param sample: Sample name
    :param roi: Roi name
    :param interval: spectrum idx range used for proccesing
    :param dots_num: number of dots in spectrum
    :param dcont: spectra type of the data continues or discontinues
    :param print_queue: Менеджер для отображения сообщений на экран с процесса.
    :param discon_resample_range: mz range for data resample if imzML data is discontinuous
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param args2procc: params packaged in dictionary for proccessing
    :param queue: Number of iterations for spectrum smooth. Default: `1`
    :param chunk_size: number of rows per chunk in hdf5
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    
    :type data_obj_path: `str`
    :type sample: `str`
    :type roi: `str`
    :type interval: `list`
    :type dots_num: `int`
    :type dcont: `bool`
    :type print_queue: `Manager.Queue()`
    :type discon_resample_range: `tuple`
    :type resample_to_dots: `int`
    :type args2procc: `dict`
    :type queue: `Manager.Queue()`
    :type chunk_size: `int`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `None`
    :rtype: `NoneType`
    """
        
    ## Пояснение, что какого-то файла не удалось найти/открыть при массовой обработке данных
    idx_range = range(*interval)
    sample_imzml=ImzMLParser(sample_file_path)
    loc_args2procc = args2procc.copy()

    if dcont:
        ## Preallocating int
        data_int = np.empty((interval[-1]-interval[0],dots_num), dtype=dtypeconv)
        ##
        ## Заполнение пустых матриц int 
        for n,idx in enumerate(idx_range):
            data_int[n,:] = sample_imzml.getspectrum(idx)[1].astype(dtypeconv)
        
        if resample_to_dots is not None:
            data_mz_old = sample_imzml.getspectrum(interval[0])[0].astype(dtypeconv)
            data_mz = np.array(list(np.linspace(min(data_mz_old),max(data_mz_old),resample_to_dots)))
            loc_args2procc["smooth_window"] = int(args2procc["smooth_window"]/(np.median(np.diff(data_mz))))
            if loc_args2procc["params2align"]["only_shift"]:
                loc_args2procc["dots_shift"]= int(args2procc["dots_shift"]/(np.median(np.diff(data_mz))))
            data_int = DataProc_resample(data_int,data_mz_old,data_mz,Baseline(data_mz),**loc_args2procc)
            dots_num = resample_to_dots
        else:
            
            data_mz = sample_imzml.getspectrum(interval[0])[0].astype(dtypeconv)
            loc_args2procc["smooth_window"] = int(args2procc["smooth_window"]/(np.median(np.diff(data_mz))))

            if loc_args2procc["params2align"]["only_shift"]:
                loc_args2procc["dots_shift"] = int(args2procc["dots_shift"]/(np.median(np.diff(data_mz))))
            data_int = DataProc_base(data_int,data_mz,Baseline(data_mz),**loc_args2procc)
    else:
        
        if resample_to_dots:
            data_int={}
            dots_num = resample_to_dots
            data_mz = np.array(list(np.linspace(*discon_resample_range,resample_to_dots)))
            loc_args2procc["smooth_window"] = int(args2procc["smooth_window"]/(np.median(np.diff(data_mz))))
            if loc_args2procc["params2align"]["only_shift"]:
                loc_args2procc["dots_shift"] = int(args2procc["dots_shift"]/(np.median(np.diff(data_mz))))
            baseliner = Baseline(data_mz)
            for n, idx in enumerate(idx_range):
                data_mz_old,data_int_old=sample_imzml.getspectrum(idx)
                data_int[n] = DataProc_resample1d(data_int_old,data_mz_old,data_mz,baseliner,**loc_args2procc)
            
            data_int= np.vstack(tuple(data_int.values()))
        else:
            print_queue.put("The proccesed data is not continuous. The code for this is being written.")
            return
            ## Получение пиков
            data_mz = np.empty((interval[-1]-interval[0],dots_num), dtype=dtypeconv)
            for n, idx in enumerate(idx_range):
                data_mz[n,:] = sample_imzml.getspectrum(idx)[0].astype(dtypeconv)
                loc_args2procc["smooth_window"] = int(args2procc["smooth_window"]/(np.median(np.diff(data_mz))))
                if loc_args2procc["params2align"]["only_shift"]: 
                    loc_args2procc["dots_shift"] = int(args2procc["dots_shift"]/(np.median(np.diff(data_mz))))
                data_int[n,:] = DataProc_base(sample_imzml.getspectrum(idx)[1].astype(dtypeconv),data_mz,[Baseline(data_mz)],**loc_args2procc)

    ### Запись пиков в hdf5 очередью
    temp = queue.get()
    Slide_folder = os.path.dirname(os.path.dirname(sample_file_path))
    hdf5 = File(os.path.join(Slide_folder,os.path.basename(Slide_folder)) +"_specdata.hdf5","a")
    idx_start,numspec = hdf5[sample][roi].attrs['idxroi']
    sl = range(interval[0]-idx_start,interval[1]-idx_start)

    try:
        hdf5[sample][roi]["int"][sl,:] = data_int
    except:
        hdf5.create_dataset(sample + "/" + roi + "/int", (numspec, dots_num), chunks=(chunk_size, dots_num))
        hdf5.create_dataset(sample + "/" + roi + "/mz", data = data_mz)
        hdf5[sample][roi]["int"][sl,:] = data_int

    hdf5.close()
    print_queue.put(True)
    queue.put(True)
    
    return

def int2proc2peaklist_parbatched(sample_file_path, sample,roi,interval,dots_num,dtypeconv,dcont,print_queue, discon_resample_range = (None,None),resample_to_dots = None,
                        args2procc={},args2peakpicking={}, queue = None, chunk_size = 10000):
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой обработки сырыах спектров до пиклистов, без сохранения промежуточных результатов. Используется в функции Raw2peaklist.

    :param sample_file_path: path to spectra source `imzML`
    :param sample: Sample name
    :param roi: Roi name
    :param interval: spectrum idx range used for proccesing
    :param dots_num: number of dots in spectrum
    :param dcont: spectra type of the data continues or discontinues
    :param print_queue: Менеджер для отображения сообщений на экран с процесса.
    :param discon_resample_range: mz range for data resample if imzML data is discontinuous
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param args2procc: params packaged in dictionary for spectra proccessing
    :param args2peakpicking: params packaged in dictionary for peakpicking
    :param queue: Number of iterations for spectrum smooth.
    :param chunk_size: number of rows per chunk in hdf5
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    
    :type data_obj_path: `str`
    :type sample: `str`
    :type roi: `str`
    :type interval: `list`
    :type dots_num: `int`
    :type dcont: `bool`
    :type print_queue: `Manager.Queue()`
    :type discon_resample_range: `tuple`
    :type resample_to_dots: `int`
    :type args2procc: `dict`
    :type args2peakpicking: `dict`
    :type queue: `Manager.Queue()`
    :type chunk_size: `int`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `None`
    :rtype: `NoneType`
    """
    ## Пояснение, что какого-то файла не удалось найти/открыть при массовой обработке данных
    idx_range = range(interval[0],interval[1])
    sample_imzml=ImzMLParser(sample_file_path)
    temp = queue.get()
    Slide_folder = os.path.dirname(os.path.dirname(sample_file_path))
    with File(os.path.join(Slide_folder,os.path.basename(Slide_folder)) +"_specdata.hdf5",'r', libver='latest') as hdf5:
        idx_start = hdf5[sample][roi].attrs['idxroi'][0]
    queue.put(True)
    nspec_range=range(interval[0]-idx_start,interval[1]-idx_start)
    loc_args2procc = args2procc.copy()

    if dcont:
        
        ## Preallocating int
        data_int = np.empty((interval[-1]-interval[0],dots_num), dtype=dtypeconv)
  
        ##
        ## Заполнение пустых матриц int 
        for n,idx in enumerate(idx_range):
            data_int[n,:] = sample_imzml.getspectrum(idx)[1].astype(dtypeconv)
        
        if resample_to_dots:
            data_mz_old = sample_imzml.getspectrum(0)[0].astype(dtypeconv)
            data_mz = np.array(list(np.linspace(min(data_mz_old),max(data_mz_old),resample_to_dots)))
            loc_args2procc["smooth_window"] = int(args2procc["smooth_window"]/(np.median(np.diff(data_mz))))
            if loc_args2procc["params2align"]["only_shift"]:
                loc_args2procc["dots_shift"]= int(args2procc["dots_shift"]/(np.median(np.diff(data_mz))))
            data_int = DataProc_resample(data_int,data_mz_old,data_mz,Baseline(data_mz),**loc_args2procc)

        else:
            data_mz = sample_imzml.getspectrum(0)[0].astype(dtypeconv)
            loc_args2procc["smooth_window"] = int(args2procc["smooth_window"]/(np.median(np.diff(data_mz))))
            if loc_args2procc["params2align"]["only_shift"]: 
                loc_args2procc["dots_shift"] = int(args2procc["dots_shift"]/(np.median(np.diff(data_mz))))
            data_int = DataProc_base(data_int,data_mz,Baseline(data_mz),**loc_args2procc)
        ## Получение пиков
        
        peaklists = mspeaks_arrayopt(data_mz,data_int,nspec_range,**args2peakpicking)

    else:

        peaklists={}
        if resample_to_dots:
            data_mz = np.array(list(np.linspace(*discon_resample_range,resample_to_dots)))
            loc_args2procc["smooth_window"] = int(args2procc["smooth_window"]/(np.median(np.diff(data_mz))))
            if loc_args2procc["params2align"]["only_shift"]:
                loc_args2procc["dots_shift"] = int(args2procc["dots_shift"]/(np.median(np.diff(data_mz))))
            baseliner = Baseline(data_mz)
            for n, idx in enumerate(idx_range):
                data_mz_old,data_int=sample_imzml.getspectrum(idx)
                peaklists[n] = mspeaks_opt(data_mz, DataProc_resample1d(data_int,data_mz_old,data_mz,baseliner,**loc_args2procc), 
                                           nspec_range[n], **args2peakpicking)
         
        else:
            ## Получение пиков

            for n, idx in enumerate(idx_range):
                data_mz = sample_imzml.getspectrum(idx)[0]
                loc_args2procc["smooth_window"] = int(args2procc["smooth_window"]/(np.median(np.diff(data_mz))))
                if loc_args2procc["params2align"]["only_shift"]:
                    loc_args2procc["dots_shift"] = int(args2procc["dots_shift"]/(np.median(np.diff(data_mz))))
               
                peaklists[n] = mspeaks_opt(data_mz, DataProc_base1d(sample_imzml.getspectrum(idx)[1],data_mz,Baseline(data_mz),**loc_args2procc), 
                                           nspec_range[n], **args2peakpicking)
               
        peaklists = np.vstack(tuple(peaklists.values()))
    ### Запись пиков в hdf5 очередью
    temp = queue.get()

    with File(os.path.join(Slide_folder,os.path.basename(Slide_folder)) +"_specdata.hdf5","a", libver='latest') as hdf5:

        try:
   
            start_row = hdf5[sample][roi]["peaklists"].shape[0]
            hdf5[sample][roi]["peaklists"].resize(start_row + peaklists.shape[0],0)
            hdf5[sample][roi]["peaklists"][start_row:(start_row+peaklists.shape[0]),:] = peaklists

        except:

            hdf5.create_dataset(sample + "/" + roi + "/peaklists",peaklists.shape, maxshape = (None, 11), chunks=(chunk_size, 11))

            hdf5[sample][roi]["peaklists"][:] = peaklists

            hdf5[sample][roi]["peaklists"].attrs["Column headers"] = ["spectra_ind","mz","Intensity","Area","SNR","PextL","PextR","FWHML","FWHMR","Noise","Mean noise"]
   #print_queue.put(f"{sample + " " + roi} batch {interval} data writing is finished")

    print_queue.put(True)
    queue.put(True)
    
    return
            
def proc2peaklist_parbatched(sl, sample ,roi ,sample_file_path,args2peakpicking={},dtypeconv='double',print_queue=None):
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессингового пикпикинга из обработанных спектров и сохранением полученных пиклистов в hdf5 под названием "[Slidename]_specdata.hdf5" в датасет "peaklists". Используется в функции proc2peaklist.

    :param sample_file_path: path to spectra source `imzML`
    :param sample: Sample name
    :param roi: Roi name
    :param interval: spectrum idx range used for proccesing
    :param dots_num: number of dots in spectrum
    :param dcont: spectra type of the data continues or discontinues
    :param print_queue: Менеджер для отображения сообщений на экран с процесса.
    :param discon_resample_range: mz range for data resample if imzML data is discontinuous
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param args2procc: params packaged in dictionary for spectra proccessing
    :param args2peakpicking: params packaged in dictionary for peakpicking
    :param queue: Number of iterations for spectrum smooth.
    :param chunk_size: number of rows per chunk in hdf5
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    
    :type data_obj_path: `str`
    :type sample: `str`
    :type roi: `str`
    :type interval: `list`
    :type dots_num: `int`
    :type dcont: `bool`
    :type print_queue: `Manager.Queue()`
    :type discon_resample_range: `tuple`
    :type resample_to_dots: `int`
    :type args2procc: `dict`
    :type args2peakpicking: `dict`
    :type queue: `Manager.Queue()`
    :type chunk_size: `int`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `None`
    :rtype: `NoneType`
    """
    ## Пояснение, что какого-то файла не удалось найти/открыть при массовой обработке данных
    
    with File(sample_file_path,'r', libver='latest', swmr=True) as data_obj:
        idx_range = range(sl.start,sl.stop)
        peaklists = mspeaks_arrayopt(data_obj[sample][roi]['mz'][:].astype(dtypeconv), data_obj[sample][roi]['int'][idx_range,:].astype(dtypeconv),idx_range,**args2peakpicking)
    print_queue.put(True)
    return peaklists

def poslog_parbatched(sample_file, batch_bsize, dtypeconv, print_queue,cpu_num,resample_to_dots): 
    """
    Общее описание
    ----
    Функия обработки файла poslog: 
    1. Разделение данных по регионам ("R00" к примеру, а выходных данных он же будет просто "00"), а также индексы начала и конца этих регионов (словарь roi_idx). 
    roi_idx используются в другой функции для разделения спектров в imzml файле по регионам
    2. Вытаскивание информации по координатам. Переменная, содержащая эти данные на выходе - data_obj (словарь) 
    3. Составление индексов для разбивки по батчам для параллелизации вычислений, в том числе подготовка параметров обработки.

    :param sample_file: path to spectra source `imzML`
    :param batch_bsize: размер одного батча в байтах. Определяется автоматически на основе рассчётов по кол-ву ядер и заданного параметра в других функциях по максимальному занимаемому RAM (Like `Ram_GB`)
    :param print_queue: Менеджер для отображения сообщений на экран с процесса.
    :param cpu_num: предполагаемое кол-во используемых процессов CPU
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`
    
    :type sample_file: `str`
    :type batch_bsize: `float`
    :type cpu_num: `int`
    :type print_queue: `Manager.Queue()`
    :type resample_to_dots: `int`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `tuple` with contains dict with coordinates and packaged params for multiproccessing
    :rtype: `tuple`
    """ 

    ### 1. Поиск пути к файлу imzml и составления их списка
    ### 1. Searching and opening imzml file
    
    folder_path2imzml = os.path.dirname(sample_file)
    sample_name = os.path.splitext(os.path.basename(sample_file))[0]
    folder_name = os.path.basename(folder_path2imzml)

    if folder_name == sample_name:
        sample=sample_name
        poslog_err = sample_name
    else:
        sample = folder_name+"_"+sample_name        
    base_path = os.path.join(folder_path2imzml,sample_name)
    ## Определение байтового размера одной точки
    if dtypeconv =='single':
        bytes_flsize = 4
    elif dtypeconv == 'double':
        bytes_flsize = 8
    elif dtypeconv == 'half':
        bytes_flsize = 2
    ##
    try:
        sample_imzml=ImzMLParser(sample_file)
        data_obj={} 
        data_obj[sample]={}
    except FileNotFoundError: #Если нет imzML файла в папке - пропуск
        print_queue.put(f'No {sample+".imzML"} file in directory {folder_path2imzml}')
        return
    ### 1. Файл найден и открыт в sample_imzml файле - DONE
    ### 1. File found and opened in sample_imzml file - DONE
    ### Data extraction
    try: ### b. Extraction from _poslog and _info text files
        count=0
        idx_first=0
        roi_idx = {} 
        roi_idx[sample]={} # Информация sample по индексам спектров roi=(индекс первого спектра, кол-во спектров roi)

        roi_list = []
        try:
            with open(base_path+"_info.txt") as f:
                data_info = f.readlines()
                raw_data_points = int(data_info[12].split(' ')[1]) # Информация по кол-ву точек спектра
                spectra_num = int(data_info[2].split(' ')[-1]) # Информация по кол-ву спектров в sample
        except:   
            raw_data_points = int(np.quantile(sample_imzml.mzLengths, 0.95))
            spectra_num = len(sample_imzml.mzLengths)
        try:
            dcont = sample_imzml.metadata.pretty()["file_description"]["continuous"]

        except KeyError:
            dcont = not sample_imzml.metadata.pretty()["file_description"]["processed"]

        ###
        ### Так как файлы imzml с poslog исключительно континуальные (одна шкала mz для всех спектров), то здесь работаем исключительно в таком варианте
        ### Выгрузка данных с poslog
        ### 4.b. and 5.b. Extraction data from "poslog" coordinates and roi references of spectra
        with open(base_path+"_poslog.txt") as f:
            data = f.readlines()
            
            poslog_specdata = [None]*spectra_num #Данные строк в poslog с записью roi и координат снятого спектра.
            ##первая итерация записи координат начиная с третьей строки
            coords =  data[2].split(' ') 
            roi_num = re.search('R(.+?)X', data[2]).group(1)
            roi_list.append(roi_num)
            poslog_specdata[count]=(roi_num,float(coords[-3]), float(coords[-2]))
            
            roi_idx[sample][roi_num] = idx_first
            data_obj[sample][roi_num] = {"z":np.array(float(coords[-1]),dtype=dtypeconv)}
            
            count+=1
            ## продолжение итераций    
            for i in range(2,len(data)-1):
                coords =  data[i+1].split(' ')
                
                if(coords[-4]!='__'):
                    roi_num = re.search('R(.+?)X', data[i+1]).group(1)
                    poslog_specdata[count]=(roi_num,float(coords[-3]), float(coords[-2]))
                    
                    ### Строгое положение из-за roi_list[-2] и условия в if
                    if roi_num not in roi_list[-1]:
                        roi_list.append(roi_num)
                        roi_idx[sample][roi_num] = []
                        roi_idx[sample][roi_list[-2]] = (idx_first, count-idx_first)
                        idx_first=count
                        data_obj[sample][roi_num] = {"z":np.array(float(coords[-1]),dtype=dtypeconv)}
                    ###
                    count +=1
            roi_idx[sample][roi_num] = (idx_first, count-idx_first) ### 2.b. Num of spectra of roi/sample
        ### 4.b. and 5.b. Extraction data from "poslog" coordinates and roi references of spectra - DONE
        ### Определение типа данных: континуальные(одна шкала mz для всех спектров) или нет. (возможно стоит доработать, поставленные условия определения типа данных выставлены на основе имеющихся) 
        intraw_par_args = []
        ### 3.b. Number of batches and its sizes determination

        if resample_to_dots is not None:
            spectra_rowsize = batch_bsize/(bytes_flsize*resample_to_dots) ### Возможно слабое место из-за raw_data_points, так как возможно могут существовать ситуации, когда в разных roi разное кол-во точек (надо это как-то уточнить)
            #для создания актуальных батчей после ресемпла
        else:
            spectra_rowsize = batch_bsize/(bytes_flsize*raw_data_points) ### Возможно слабое место из-за raw_data_points, так как возможно могут существовать ситуации, когда в разных roi разное кол-во точек (надо это как-то уточнить)
        roi_count = len(roi_list)
        n_int = int(spectra_num/(roi_count*spectra_rowsize)+1)
        if n_int*roi_count<cpu_num*2:
            n_int = int(cpu_num*2/roi_count)+1
            
        for roi in roi_list:
            data_obj[sample][roi]["xy"] = np.empty((roi_idx[sample][roi][1],2))
            #data_obj[sample][roi]["mz_raw"] = sample_imzml.getspectrum(roi_idx[sample][roi_num][0])[0]
            data_obj[sample][roi]["continuous"] = dcont
            data_obj[sample][roi]["idxroi"] = roi_idx[sample][roi]
            data_obj[sample][roi]["source"] = sample_file
            if dcont:
                min_mz = min(sample_imzml.getspectrum(roi_idx[sample][roi][0])[0].astype(dtypeconv))
                max_mz = max(sample_imzml.getspectrum(roi_idx[sample][roi][0])[0].astype(dtypeconv))
            else:    
                min_mz = min(sample_imzml.getspectrum(roi_idx[sample][roi][0])[0].astype(dtypeconv))
                max_mz = max(sample_imzml.getspectrum(roi_idx[sample][roi][0])[0].astype(dtypeconv))
                for idx in range(roi_idx[sample][roi][0]+1,roi_idx[sample][roi][0]+roi_idx[sample][roi][1]):
                    min_mz = min([min_mz,min(sample_imzml.getspectrum(idx)[0].astype(dtypeconv))])
                    max_mz = max([max_mz,max(sample_imzml.getspectrum(idx)[0].astype(dtypeconv))])
           

        ### 3.b. Number of batches and its sizes determination - Done

            ### 6. Preparing index parameters associated with sample, ROI and organizing them into argument list for parallel processing of spectra
            intraw_par_args = [*intraw_par_args, *list(product([sample_file],[sample],[roi], pairwise( np.linspace(roi_idx[sample][roi][0], roi_idx[sample][roi][0]+roi_idx[sample][roi][1] , n_int, dtype=int)), [raw_data_points], [dtypeconv], [dcont], [print_queue], [(min_mz,max_mz)] )  )]
        
        for idx, (roi,x,y) in enumerate(poslog_specdata):            
            data_obj[sample][roi]["xy"][idx-roi_idx[sample][roi][0],:] = [x, y]
        ### 6. Preparing index parameters associated with sample, ROI and organizing them into argument list for parallel processing of spectra - Done
        #print_queue.put(data_obj)
        del roi_num, coords, idx_first
        #print_queue.put("Done")
        ###
        ### Если нет poslog файла в папке, берём координаты из imzml
        ### a. If there is no poslog file in the folder, take coordinates from imzml
    except FileNotFoundError: 
        #print_queue.put("Done 2")
        print_queue.put(f'The {poslog_err+"_poslog.txt"} file is not in directory {folder_path2imzml}, the coordinate data is taken from the imzML file')
        #print_queue.put("Done 2")
        roi = "00" # roi только один, так как там вроде нельзя настраивать и определять без poslog
        roi_list = []
        roi_list.append(roi_list)
        data_obj[sample][roi]={}
        ### Определение типа данных: континуальные(одна шкала mz для всех спектров) или нет. (возможно стоит доработать, поставленные условия определения типа данных выставлены на основе имеющихся) 
        try:
            dcont = sample_imzml.metadata.pretty()["file_description"]["continuous"]
        except KeyError:
            dcont = not sample_imzml.metadata.pretty()["file_description"]["processed"]

        ###
        dpoints = sample_imzml.mzLengths
        specnum = len(dpoints)
        
        
        data_obj[sample][roi]["xy"] = np.empty((specnum,2))
        data_obj[sample][roi]["continuous"] = dcont
        data_obj[sample][roi]["idxroi"] = (0,specnum)
        data_obj[sample][roi]["source"] = sample_file
        ## Batching and organization spectra data for parallel proccesing
        if dcont:
            min_mz = min(sample_imzml.getspectrum(0)[0]).astype(dtypeconv)
            max_mz = max(sample_imzml.getspectrum(0)[0]).astype(dtypeconv)
            # Finding number of spectrum dots of continous data
            raw_data_points = dpoints[0] 
            try:
                for idx in range(specnum):
                    data_obj[sample][roi]["xy"][idx,:] = sample_imzml.get_physical_coordinates(idx)
            except:
                data_obj[sample][roi]["xy"] = np.array(sample_imzml.coordinates)[:,[0,1]]
        else:
            min_mz=min(sample_imzml.getspectrum(0)[0].astype(dtypeconv))
            max_mz=max(sample_imzml.getspectrum(0)[0].astype(dtypeconv))
            data_obj[sample][roi]["xy"][0,:] = sample_imzml.get_physical_coordinates(0)
            try:
                for idx in range(1,specnum,1):
                    data_obj[sample][roi]["xy"][idx,:] = sample_imzml.get_physical_coordinates(idx)
                    min_mz = min([min_mz,min(sample_imzml.getspectrum(idx)[0].astype(dtypeconv))])
                    max_mz = max([max_mz,max(sample_imzml.getspectrum(idx)[0].astype(dtypeconv))])
            except:
                data_obj[sample][roi]["xy"] = np.array(sample_imzml.coordinates)[:,[0,1]]

                for idx in range(1,specnum,1):
                    min_mz = min([min_mz,min(sample_imzml.getspectrum(idx)[0].astype(dtypeconv))])
                    max_mz = max([max_mz,max(sample_imzml.getspectrum(idx)[0].astype(dtypeconv))])
            #data_obj[sample][roi]["mz_raw"] = [0]*specnum
            raw_data_points = np.quantile(dpoints,0.95)

        if resample_to_dots is not None:
            spectra_rowsize = batch_bsize/(bytes_flsize*resample_to_dots)
        else:
            spectra_rowsize = batch_bsize/(bytes_flsize*raw_data_points)
        n_int = int(specnum/spectra_rowsize+1)
        if n_int<cpu_num*2:
            n_int = cpu_num*2
            #raw_data_points = 0
        intraw_par_args = list(product([sample_file],[sample],[roi], pairwise( np.linspace(0, specnum , n_int, dtype=int)),[raw_data_points], [dtypeconv], [dcont], [print_queue], [(min_mz,max_mz)] )  )

        del dpoints 
        
        data_obj[sample][roi]["z"] = 0 #Заглушка, нигде z не узнать
    return (data_obj, intraw_par_args)
### Utility functions for processing

def hdf5_coords(file_path,slide,data_obj_coord,chunk_size):
    """
    Общее описание
    ----
    Вспомогательная функция для создания двух hdf5 файлов "[Slidename]_specdata.hdf5" и "[Slidename]_features.hdf5" и записи в них координат и часть данных таких как путь к первоисточнику, континуальность данных и записью принадлежности индекса спектра к определённому roi в sample.

    :param file_path: path to folder for writing `hdf5`.
    :param slide: параметр задающий Slidename в названии файла `hdf5`
    :param data_obj_coord: словарь схожий по структуре записи с будущими hdf5 и непосредственно из которого берутся все данные для записи.
    :param chunk_size: количество строк, на которые разделяется матрица в hdf5 файле
    
    :type file_path: `str`
    :type slide: `str`
    :type data_obj_coord: `dict`
    :type chunk_size: `int`

    :return: `None`
    :rtype: `NoneType`
    """ 

 
    for file_end in ["_specdata","_features"]:
        data_obj = File(os.path.join(file_path,slide)+f"{file_end}.hdf5","a")
        for sample in data_obj_coord.keys():
            for roi in data_obj_coord[sample].keys():
                try:
                    data_obj[sample][roi]["xy"][:]
                except:        
                    try:
                        if isinstance(chunk_size, dict):
                            data_obj.create_dataset(sample+"/" + roi + "/" + "xy",data=data_obj_coord[sample][roi]["xy"], chunks = (chunk_size[sample],2))
                        else:
                            data_obj.create_dataset(sample+"/" + roi + "/" + "xy",data=data_obj_coord[sample][roi]["xy"], chunks = (chunk_size,2))
                        data_obj.create_dataset(sample+"/" + roi + "/" + "z",data=data_obj_coord[sample][roi]["z"])
                        
                    except ValueError:
                        data_obj.create_dataset(sample+"/" + roi + "/" + "xy",data=data_obj_coord[sample][roi]["xy"])
                        data_obj.create_dataset(sample+"/" + roi + "/" + "z",data=data_obj_coord[sample][roi]["z"])
                finally:

                    data_obj[sample][roi].attrs['source'] = data_obj_coord[sample][roi]['source']
                    data_obj[sample][roi].attrs['continuous'] = data_obj_coord[sample][roi]['continuous']
                    data_obj[sample][roi].attrs['idxroi'] = data_obj_coord[sample][roi]['idxroi']
        data_obj.close()
    return

def DataProc_base(y,x,baseliner,baseliner_algo,params2baseliner_algo,
                   params2align,align_peaks, weights_list, dots_shift,
                   smooth_algo,smooth_window,smooth_cycles): 
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой предобработки спектров без ресемплинга. Function works only with continual data or with one dimensional array/list.
    
    :param y: array of spectra intensities with shape (n,d), where each row (n) corresponds to intensities of spectrum and column (d) corresponds to dots of spectra   
    :param x: array of spectra mz with shape (1,d)
    :param baseliner: Baseline class for baseline correction
    :param baseliner_algo: Algorithm of baseline correction.

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    

    :param params2baseliner_algo: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html)
        
        .. Example: {"lam" : 500000, "diff_order" : 1}

    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`).

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param align_peaks: list of reference peaks for align
    :param weights_list: list of weights for reference peaks in aligning
    :param dots_shift: max spectrum shift in dots
    :param smooth_algo: spectrum smoothing algorithm. Default is `"GA"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param smooth_window: window size for smooth
    :param smooth_cycles: Number of iterations for spectrum smooth

    :type y: `array`
    :type x: `array` or `list`
    :type baseliner: `Baseline` class
    :type baseliner_algo: `str`
    :type params2baseliner_algo:  `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`}
    :type smooth_window: `float`
    :type smooth_cycles: `int`

    :return: array of proccessed spectra
    :rtype: `np.array`

    .. todo:: code refactoring with class
    """
    
    ## Spectra preproccessing
    rows = y.shape[0] #определяем кол-во спектров(строк)
    
    if align_peaks: # Этап выравнивания спектров. Выравнивание линейное на основе выбранных референсных пиков с выставлением их веса и максимального сдвига по mz (в коде переводится сдвиг по кол-во точек)
        y = msalign(x,y,align_peaks, weights=weights_list,**params2align,shift_range=[-dots_shift,dots_shift])
    
    if baseliner_algo:
        if smooth_algo: # Этап сглаживания спектров с одновременным удалением базовой линии
            for i in range(rows):
                y[i,:] = smoothing(y[i,:] - getattr(baseliner,baseliner_algo)(y[i,:],**params2baseliner_algo)[0],smooth=smooth_algo,window=smooth_window,cycles=smooth_cycles)
        else:
            for i in range(rows): # Вариант с просто удалением базовой линии
                y[i,:] -= getattr(baseliner,baseliner_algo)(y[i,:],**params2baseliner_algo)[0]
    else:
        if smooth_algo: # Этап сглаживания спектров
            for i in range(rows):
                y[i,:] = smoothing(y[i,:],smooth=smooth_algo,window=smooth_window,cycles=smooth_cycles)
    
    return np.array(y)# Перевод в numpy матрицу

def DataProc_base1d(y,x,baseliner, baseliner_algo,params2baseliner_algo,
                     params2align,align_peaks, weights_list, dots_shift,
                     smooth_algo,smooth_window,smooth_cycles): # Data preprocessing without resampling one dimensional
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой предобработки спектров без ресемплинга. Function works only with continual data or with one dimensional array/list.
    
    :param y: array of spectrum intensities with shape (1,d) or list   
    :param x: array of mz with shape (1,d)
    :param baseliner: Baseline class for baseline correction
    :param baseliner_algo: Algorithm of baseline correction.

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    

    :param params2baseliner_algo: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html)
        
        .. Example: {"lam" : 500000, "diff_order" : 1}

    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`).

        .. Example: {"iterations" : 2, "only_shift" : False}

    :param align_peaks: list of reference peaks for align
    :param weights_list: list of weights for reference peaks in aligning
    :param dots_shift: max spectrum shift in dots
    :param smooth_algo: spectrum smoothing algorithm. Default is `"GA"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param smooth_window: window size for smooth
    :param smooth_cycles: Number of iterations for spectrum smooth
    
    :type y: `array` or `list`
    :type x: `array` or `list`
    :type baseliner: `Baseline` class
    :type baseliner_algo: `str`
    :type params2baseliner_algo:  `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`}
    :type smooth_window: `float`
    :type smooth_cycles: `int`

    :return: array of proccessed spectra
    :rtype: `np.array`

    .. todo:: code refactoring with class
    """
    ## Spectra preproccessing
    if align_peaks: # Этап выравнивания спектров. Выравнивание линейное на основе выбранных референсных пиков с выставлением их веса и максимального сдвига по mz (в коде переводится сдвиг по кол-во точек)
        y = msalign(x,y,align_peaks, weights=weights_list,**params2align,shift_range=[-dots_shift,dots_shift]).flatten()

    if baseliner_algo:
        if smooth_algo: # Этап сглаживания спектров с одновременным удалением базовой линии
            y = smoothing(y - getattr(baseliner,baseliner_algo)(y,**params2baseliner_algo)[0],smooth=smooth_algo,window=smooth_window,cycles=smooth_cycles)
        else:
            # Вариант с просто удалением базовой линии
            y = y- getattr(baseliner,baseliner_algo)(y,**params2baseliner_algo)[0]
    else:
        if smooth_algo: # Этап сглаживания спектров
            y = smoothing(y,smooth=smooth_algo,window=smooth_window,cycles=smooth_cycles)        

    return np.array(y)# Перевод в numpy матрицу

def DataProc_resample(y,x,xnew,baseliner,baseliner_algo, params2baseliner_algo,
                       params2align, align_peaks, weights_list, dots_shift, 
                       smooth_algo, smooth_window, smooth_cycles):              # Data preprocessing with uniform dots resampling
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой предобработки спектров с ресемплингом. Function works only with continual data or with one dimensional array/list.
    
    :param y: array of spectra intensities with shape (n,d), where each row (n) corresponds to intensities of spectrum and column (d) corresponds to dots of spectra   
    :param x: array of spectra mz with shape (1,d)
    :param baseliner: Baseline class for baseline correction
    :param baseliner_algo: Algorithm of baseline correction.

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    

    :param params2baseliner_algo: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html)

                
        .. Example: {"lam" : 500000, "diff_order" : 1}

    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`).

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param align_peaks: list of reference peaks for align
    :param weights_list: list of weights for reference peaks in aligning
    :param dots_shift: max spectrum shift in dots
    :param smooth_algo: spectrum smoothing algorithm. Default is `"GA"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param smooth_window: window size for smooth
    :param smooth_cycles: Number of iterations for spectrum smooth
    
    :type y: `array`
    :type x: `array` or `list`
    :type baseliner: `Baseline` class
    :type baseliner_algo: `str`
    :type params2baseliner_algo:  `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`}
    :type smooth_window: `float`
    :type smooth_cycles: `int`

    :return: array of proccessed spectra
    :rtype: `np.array`

    .. todo:: code refactoring with class
    """
    ## Spectra preproccessing
    rows = y.shape[0]
    
    y = interp1d(x,y,fill_value=(y[:,0],y[:,-1]),bounds_error = False )(xnew)
    #y = np.interp(xnew,x,y)
    if align_peaks: #Aligning
        y = msalign(xnew,y,align_peaks, weights=weights_list,**params2align,shift_range=[-dots_shift,dots_shift]) #shift range работает по точкам, а не m/z. На вход DataProc идёт по m/z, на выходе -  
    if baseliner_algo:
        if smooth_algo:
            for i in range(rows): #smooth and baseline correction
                y[i,:] = smoothing(y[i,:] - getattr(baseliner,baseliner_algo)(y[i,:],**params2baseliner_algo)[0],smooth=smooth_algo,window=smooth_window,cycles=smooth_cycles)
            
        else:
            for i in range(rows): #baseline correction
                y[i,:] = y[i,:] - getattr(baseliner,baseliner_algo)(y[i,:],**params2baseliner_algo)[0]            

    else:
        if smooth_algo:
            for i in range(rows): #smooth
                y[i,:] = smoothing(y[i,:],smooth=smooth_algo,window=smooth_window,cycles=smooth_cycles)

    y=np.array(y) # Перевод в numpy матрицу
    
    return y
def DataProc_resample1d(y,x,xnew,baseliner,baseliner_algo, params2baseliner_algo,
                       params2align, align_peaks, weights_list, dots_shift, 
                       smooth_algo, smooth_window, smooth_cycles):              # Data preprocessing with uniform dots resampling
    """
    Общее описание
    ----
    Вспомогательная функция для мультипроцессинговой предобработки одного спектра c ресемплингом. Function works only with continual data or with one dimensional array/list.
    
    :param y: array of spectra intensities with shape (n,d), where each row (n) corresponds to intensities of spectrum and column (d) corresponds to dots of spectra   
    :param x: array of spectra mz with shape (1,d)
    :param baseliner: Baseline class for baseline correction
    :param baseliner_algo: Algorithm of baseline correction.

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    

    :param params2baseliner_algo: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html)

                
        .. Example: {"lam" : 500000, "diff_order" : 1}

    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`).

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param align_peaks: list of reference peaks for align
    :param weights_list: list of weights for reference peaks in aligning
    :param dots_shift: max spectrum shift in dots
    :param smooth_algo: spectrum smoothing algorithm. Default is `"GA"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param smooth_window: window size for smooth
    :param smooth_cycles: Number of iterations for spectrum smooth
    
    :type y: `array`
    :type x: `array` or `list`
    :type baseliner: `Baseline` class
    :type baseliner_algo: `str`
    :type params2baseliner_algo:  `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`}
    :type smooth_window: `float`
    :type smooth_cycles: `int`

    :return: array of proccessed spectra
    :rtype: `np.array`

    .. todo:: code refactoring with class
    """
    ## Spectra preproccessing
    
    y = interp1d(x,y,fill_value=(y[0],y[-1]),bounds_error = False )(xnew)

    #y = np.interp(xnew,x,y)
    if align_peaks: #Aligning
        y = msalign(xnew,y,align_peaks, weights=weights_list,**params2align,shift_range=[-dots_shift,dots_shift]).flatten() #shift range работает по точкам, а не m/z. На вход DataProc идёт по m/z, на выходе -  

    if baseliner_algo:
        if smooth_algo:
             #smooth and baseline correction
            y = smoothing(y - getattr(baseliner,baseliner_algo)(y,**params2baseliner_algo)[0],smooth=smooth_algo,window=smooth_window,cycles=smooth_cycles)

        else:
            #baseline correction
            y = y - getattr(baseliner,baseliner_algo)(y,**params2baseliner_algo)[0]            

    else:
        if smooth_algo:
            #smooth
            y = smoothing(y,smooth=smooth_algo,window=smooth_window,cycles=smooth_cycles)
    
    return np.array(y) # Перевод в numpy матрицу

def find_imzml_roots(paths):
    path_dict={}
    for path in paths:
        Sample_folder =os.path.dirname(path)
        Slide_folder = os.path.dirname(Sample_folder)
        if path.lower().endswith('.imzml'):
            path_dict.setdefault(Slide_folder,[])
            path_dict[Slide_folder].append(path)
        else:
            for root, dirs, files in os.walk(path):
                for file in files: 
                    if file.lower().endswith('.imzml'):
                        path_dict.setdefault(Sample_folder,[])
                        path_dict[Sample_folder].append(os.path.join(root,file))
            del root, dirs, files
    for key in path_dict.keys():
        path_dict[key]=list(set(path_dict[key]))
    return path_dict

#Вырезка кода для сглаживания от великого любителя кодить Martin Strohalm, который написал mMass - адекватную, качественную и бесплатную прогу для масс спектрометрии
def smoothing(y, smooth, window, cycles):
    """Smooth signal by moving average filter. New array is returned.
        signal (numpy array) - signal data points
        smooth (MA GA SG) - smoothing smooth: MA - moving average, GA - Gaussian, SG - Savitzky-Golay
        window (float) - m/z window size for smoothing
        cycles (int) - number of repeating cycles
    """
    # check signal data
    if len(y) == 0:
        return np.array([])

    # apply moving average filter
    if smooth == 'MA':
        return movaver(y, window, cycles, style='flat')
    
    # apply gaussian filter
    elif smooth == 'GA':
        return movaver(y, window, cycles, style='gaussian')
    
    # apply savitzky-golay filter
    elif smooth == 'SG':
        return savgol(y, window, cycles)
    
    # unknown smoothing smooth
    else:
        print("Смузинг нот юсд")
        return y

def movaver(y, window, cycles, style):
    """Smooth signal by moving average filter. New array is returned.
        signal (numpy array) - signal data points
        window (float) - m/z window size for smoothing
        cycles (int) - number of repeating cycles
    """
    

    if window < 3:
        return y.copy()
    if not window % 2:
        window -= 1
    
    # smooth the points
    while cycles:
        
        if style == 'flat':
            w = np.ones(window,'f')
        elif style == 'gaussian':
            r = np.array([(i-(window-1)/2.) for i in range(window)])
            w = np.exp(-(r**2/(window/4.)**2))
        else:
            w = eval('np.'+style+'(window)')
        
        s = np.r_[y[window-1:0:-1], y, y[-2:-window-1:-1]]
        yy = np.convolve(w/w.sum(), s, mode='same')
        y = yy[window-1:-window+1]
        cycles -=1


    return y

def savgol(y, window, cycles, order=3): #Не работает!!!!!!!!!!!!!!!!!!!!
    """Smooth signal by Savitzky-Golay filter. New array is returned.
        signal (numpy array) - signal data points
        window (float) - m/z window size for smoothing
        cycles (int) - number of repeating cycles
        order (int) - order of polynom used
    """
    
    if window <= order:
        return y
    
    # coefficients
    orderRange = range(order+1)
    halfWindow = (window-1) // 2
    b = np.mat([[k**i for i in orderRange] for k in range(-halfWindow, halfWindow+1)])
    m = np.linalg.pinv(b).A[0]
    window = len(m)
    halfWindow = (window-1) // 2
    
    # precompute the offset values for better performance
    offsets = range(-halfWindow, halfWindow+1)
    offsetData = zip(offsets, m)
    
    # smooth the data
    while cycles>0:
        smoothData = list()
        
        y = np.concatenate((np.zeros(halfWindow)+y[0], y, np.zeros(halfWindow)+y[-1]))
        for i in range(halfWindow, len(y) - halfWindow):
            
            value = 0.0
            for offset, weight in offsetData:
                value += weight * y[i + offset]
            smoothData.append(value)
        
        y = smoothData
        cycles -=1
    
    # return smoothed data
    y = np.array(y)
    return y

def MAD(y,nan_policy):
    return sqrt(2*math.log(len(y)))*median_abs_deviation(y,nan_policy)/0.6745 # from matlab "mad" algorithm noise description (but this is for y_h to filter out noisy components in the first high-band decomposition of DCWT peak picking)


### Utility functions for peakpicking
def mspeaks_opt(X, Y,spectra_ind, fwhhfilter=0,oversegmentationfilter=0,heightfilter=0,rel_heightfilter=0,peaklocation=1,noise_func = np.std ,noise_est_iterations = 3, SNR_threshold = 3.5,print_queue = None):
    """
    Общее описание
    ----
    Функция для получения пиклиста с характеристиками пиков из одного спектра. Оптимизирована под использование в получении пиклистов в дисконтинуальных данных без resample'а данных.

    :param X: mz
    :param Y: Intensity
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `0`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `0`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `0`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `0`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `3.5`
    :param noise_func: функция оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `np.std`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    
    :type X: `np.array`
    :type Y: `np.array`
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
    nY =Y.shape[0]
    # Robust valley finding
    h = np.concatenate(([-1], np.where(np.diff(Y) != 0)[0], [nY-1]))
    g = (np.diff(Y[[h[1],*h[1:]]]) <= 0) & (np.diff(Y[[*h[1:],h[-1]]]) >= 0)
    
    left_min = h[np.concatenate([g, [False]])] + 1
    right_min = h[np.concatenate([[False], g])]
    #print_queue.put('base')
    left_min = left_min[:-1]
    right_min = right_min[1:]
    # Compute max, and min for every peak
    size = left_min.shape
    #lfwhh = np.empty(size)
    #rfwhh = np.empty(size)
    val_max = np.empty(size)
    #pos_peak = np.empty(size)
    for idx, [lm, rm] in enumerate(zip(left_min, right_min)):
        #pp = lm + np.argmax(Y[lm:rm])
        vm = np.max(Y[lm:rm])
        #valmin = np.min(Y[lm:rm+1])
        val_max[idx] = vm 
        #pos_peak[idx] = pp
        
        #lfwhh[idx] = interp1d(Y[lm:pp+1]-valmin, X[lm:pp+1], kind='linear',fill_value='extrapolate')((vm-valmin)/2)
        #rfwhh[idx] = interp1d(Y[pp:rm+1][::-1]-valmin, X[pp:rm+1][::-1], kind='linear',fill_value="extrapolate")((vm-valmin)/ 2)


# Handle NaN values
#lfwhh = np.where(np.isnan(lfwhh), X[left_min], lfwhh)
#rfwhh = np.where(np.isnan(rfwhh), X[right_min], rfwhh)

    
    #print_queue.put('filt')
    # Remove peaks below the height, relative height
    k = (val_max >= heightfilter) & (val_max/max(Y) >= rel_heightfilter)
    val_max = val_max[k]
    #lfwhh = lfwhh[k]
    #rfwhh = rfwhh[k]
    left_min = left_min[k]
    right_min = right_min[k]
    
    # Remove peaks below the SNR thresholds
    
    noise_points = np.array([True]*nY) # Zero iteration

    for it in range(noise_est_iterations):
        
        for idx in np.where(((val_max-np.mean(Y[noise_points]))/noise_func(Y[noise_points])>=SNR_threshold))[0]:
            sl = slice(left_min[idx],right_min[idx]+1)
            noise_points[sl] = False
    
    noise = noise_func(Y[noise_points])
    mean_noise = np.mean(Y[noise_points])
    k = (val_max-mean_noise)/noise>=SNR_threshold

    val_max=val_max[k]
    #lfwhh=lfwhh[k] 
    #rfwhh=rfwhh[k]
    left_min=left_min[k]
    right_min=right_min[k]
    #print_queue.put('FWHM')

    # Compute FWHH for every peak
    size = left_min.shape
    lfwhh = np.empty(size)
    rfwhh = np.empty(size)
    pos_peak = np.empty(size)
    for idx, [lm, rm, vm] in enumerate(zip(left_min, right_min,val_max)):
        # pp = lm + np.argmax(Y[lm:rm])
        
        # valmin = np.min(Y[lm:rm+1]) 
        # pos_peak[idx] = pp
        
        # lfwhh[idx] = interp1d(Y[lm:pp+1]-valmin, X[lm:pp+1], kind='linear',fill_value='extrapolate')((vm-valmin)/2)
        # rfwhh[idx] = interp1d(Y[pp:rm+1][::-1]-valmin, X[pp:rm+1][::-1], kind='linear',fill_value="extrapolate")((vm-valmin)/ 2)
        pp = lm + np.argmax(Y[lm:rm])
        
        #valmin = np.min(Y[lm:rm+1])
        pos_peak[idx] = pp
        
        # lfwhh[idx] = interp1d(Y[lm:pp+1]-valmin, X[lm:pp+1], kind='linear',fill_value=X[lm])((vm-valmin)/2)
        # rfwhh[idx] = interp1d(Y[pp:rm+1][::-1]-valmin, X[pp:rm+1][::-1], kind='linear',fill_value=X[rm])((vm-valmin)/ 2)
        lfwhh[idx] = interp1d(Y[lm:pp+1], X[lm:pp+1], kind='linear',fill_value=X[lm],bounds_error=False)(vm/2)
        rfwhh[idx] = interp1d(Y[pp:rm+1][::-1], X[pp:rm+1][::-1], kind='linear',fill_value=X[rm],bounds_error=False)(vm/ 2)

    # Remove peaks with FWHH thresholds
    if fwhhfilter>0:
        if isinstance(fwhhfilter,tuple):
            k = ((rfwhh - lfwhh) >= fwhhfilter[0]) & ((rfwhh - lfwhh) <= fwhhfilter[1])
        else:
            k = (rfwhh - lfwhh) >= fwhhfilter
        val_max = val_max[k]
        lfwhh = lfwhh[k]
        rfwhh = rfwhh[k]
        left_min = left_min[k]
        right_min = right_min[k]
    # Remove oversegmented peaks
    if isinstance(oversegmentationfilter,str):
        oversegmentationfilter = np.median(rfwhh-lfwhh)
        #print(oversegmentationfilter)
    #val_max=np.array(val_max)
    while True:
        peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
        pkX = np.empty(left_min.shape)
        
        for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
            mask = Y[lm:rm] >= th
            if np.sum(mask) == 0:
                pkX[idx]=np.nan
            else:
                pkX[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
        dpkX = np.concatenate(([np.inf], np.diff(pkX), [np.inf]))
        
        j = np.where((dpkX[1:-1] <= oversegmentationfilter) & (dpkX[1:-1] <= dpkX[:-2]) & (dpkX[1:-1] < dpkX[2:]))[0]
        if j.size == 0:
            break
        left_min = np.delete(left_min, j + 1)
        right_min = np.delete(right_min, j)
        lfwhh = np.delete(lfwhh, j + 1)
        rfwhh = np.delete(rfwhh, j)
        
        val_max[j] = np.maximum(val_max[j], val_max[j + 1])
        val_max = np.delete(val_max, j + 1)
    signal_num = len(val_max)

    ## Area calculation
    pkA = np.empty((signal_num,))
    for idx in range(signal_num):
        sl = slice(left_min[idx],right_min[idx]+1,1)
        if min(Y[sl])<0:
            pkA[idx] = np.trapz(Y[sl] - min(Y[sl]),X[sl])
        else:
            pkA[idx] = np.trapz(Y[sl],X[sl])
    #print_queue.put(f"{[spectra_ind]*signal_num},{pkX},{val_max},{pkA},{val_max/noise},{X[left_min+1]},{X[right_min-1]},{lfwhh}, {rfwhh},{[noise]*signal_num},{[mean_noise]*signal_num}")
    #return np.column_stack(([spectra_ind]*signal_num,pkX, val_max, pkA,val_max/noise,X[left_min], X[right_min],lfwhh, rfwhh,[noise]*signal_num,[mean_noise]*signal_num))
    return np.column_stack(([spectra_ind]*signal_num,pkX, val_max, pkA,val_max/noise,X[left_min+1], X[right_min-1],lfwhh, rfwhh,[noise]*signal_num,[mean_noise]*signal_num))

def mspeaks_arrayopt(X, Y_array,spectra_ind, fwhhfilter=0,oversegmentationfilter=0,heightfilter=0,rel_heightfilter=0,peaklocation=1,noise_func = np.std ,noise_est_iterations = 3, SNR_threshold = 3.5,print_queue = None):
    """
    Общее описание
    ----
    Функция для получения пиклиста с характеристиками пиков из континуальных или обработанных resample'ом данных. Оптимизирована под использование в получении пиклистов в континуальных данных.

    :param X: mz
    :param Y_array: Intensity array
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `0`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `0`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `0`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `0`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `3.5`
    :param noise_func: функция оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `np.std`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    
    :type X: `np.array`
    :type Y: `np.array`
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

    auto_oversegmfilter_bool = isinstance(oversegmentationfilter,str)
    nspec, nY  = Y_array.shape
    peaklists = {}     
    # Robust valley finding

    valley_bool = np.diff(Y_array) != 0
    for ns in range(nspec):
        ind =spectra_ind[ns]
        Y=Y_array[ns,:]
        
        h = np.concatenate(([-1], np.where(valley_bool[ns,:])[0], [nY-1]))
        g = (np.diff(Y[[h[1],*h[1:]]]) <= 0) & (np.diff(Y[[*h[1:],h[-1]]]) >= 0)
    
        left_min = h[np.concatenate([g, [False]])] + 1
        right_min = h[np.concatenate([[False], g])]
        left_min = left_min[:-1]
        right_min = right_min[1:]

    # Compute max and min for every peak
        size = left_min.shape
        #lfwhh = np.empty(size)
        #rfwhh = np.empty(size)
        val_max = np.empty(size)
        #pos_peak = np.empty(size)
        for idx, [lm, rm] in enumerate(zip(left_min, right_min)):
            #pp = lm + np.argmax(Y[lm:rm])
            vm = np.max(Y[lm:rm])
            #valmin = np.min(Y[lm:rm+1])
            val_max[idx] = vm 
            #pos_peak[idx] = pp
            
            #lfwhh[idx] = interp1d(Y[lm:pp+1]-valmin, X[lm:pp+1], kind='linear',fill_value='extrapolate')((vm-valmin)/2)
            #rfwhh[idx] = interp1d(Y[pp:rm+1][::-1]-valmin, X[pp:rm+1][::-1], kind='linear',fill_value="extrapolate")((vm-valmin)/ 2)


    # Handle NaN values
    #lfwhh = np.where(np.isnan(lfwhh), X[left_min], lfwhh)
    #rfwhh = np.where(np.isnan(rfwhh), X[right_min], rfwhh)

        

        # Remove peaks below the height, relative height
        k = (val_max >= heightfilter) & (val_max/max(Y) >= rel_heightfilter)
        val_max = val_max[k]
        #lfwhh = lfwhh[k]
        #rfwhh = rfwhh[k]
        left_min = left_min[k]
        right_min = right_min[k]
        
        # Remove peaks below the SNR thresholds
        
        noise_points = np.array([True]*nY) # Zero iteration

        for it in range(noise_est_iterations):
            
            for idx in np.where(((val_max-np.mean(Y[noise_points]))/noise_func(Y[noise_points])>=SNR_threshold))[0]:
                
                #try:
                #    sl = slice(left_min[idx],right_min[idx]+1)
                #    noise_points[sl] = False
                #except:
                #    print_queue.put(f"{[left_min[idx],right_min[idx],idx]} and {slice(left_min[idx],right_min[idx]+1)} in spectra num {ind}")
                sl = slice(left_min[idx],right_min[idx]+1)
                noise_points[sl] = False
                #noise_points = [True]*nY #regenerate noise points (for this type of SNR estimation this step is ?useless?)
        
        noise = noise_func(Y[noise_points])
        mean_noise = np.mean(Y[noise_points])
        k = (val_max-mean_noise)/noise>=SNR_threshold

        val_max=val_max[k]
        #lfwhh=lfwhh[k] 
        #rfwhh=rfwhh[k]
        left_min=left_min[k]
        right_min=right_min[k]

        # Compute FWHH for every peak
        size = left_min.shape
        lfwhh = np.empty(size)
        rfwhh = np.empty(size)
        pos_peak = np.empty(size)
        for idx, [lm, rm, vm] in enumerate(zip(left_min, right_min,val_max)):
            pp = lm + np.argmax(Y[lm:rm])
            
            #valmin = np.min(Y[lm:rm+1])
            pos_peak[idx] = pp
            
            # lfwhh[idx] = interp1d(Y[lm:pp+1]-valmin, X[lm:pp+1], kind='linear',fill_value=X[lm])((vm-valmin)/2)
            # rfwhh[idx] = interp1d(Y[pp:rm+1][::-1]-valmin, X[pp:rm+1][::-1], kind='linear',fill_value=X[rm])((vm-valmin)/ 2)
            lfwhh[idx] = interp1d(Y[lm:pp+1], X[lm:pp+1], kind='linear',fill_value=X[lm],bounds_error=False)(vm/2)
            rfwhh[idx] = interp1d(Y[pp:rm+1][::-1], X[pp:rm+1][::-1], kind='linear',fill_value=X[rm],bounds_error=False)(vm/ 2)
        # Remove peaks higher FWHH thresholds
        if fwhhfilter>0:
            if isinstance(fwhhfilter,tuple):
                k = ((rfwhh - lfwhh) >= fwhhfilter[0]) & ((rfwhh - lfwhh) <= fwhhfilter[1])
            else:
                k = (rfwhh - lfwhh) >= fwhhfilter
            val_max = val_max[k]
            lfwhh = lfwhh[k]
            rfwhh = rfwhh[k]
            left_min = left_min[k]
            right_min = right_min[k]
        
        # Remove oversegmented peaks
        if auto_oversegmfilter_bool:
            oversegmentationfilter = np.median(rfwhh-lfwhh)
            #print(oversegmentationfilter)
        #val_max=np.array(val_max)
        while True:
            peak_thld = val_max * peaklocation - math.sqrt(np.finfo(float).eps)
            pkX = np.empty(left_min.shape)
            
            for idx, [lm, rm, th] in enumerate(zip(left_min, right_min, peak_thld)):
                mask = Y[lm:rm] >= th
                if np.sum(mask) == 0:
                    pkX[idx]=np.nan
                else:
                    pkX[idx] = np.sum(Y[lm:rm][mask] * X[lm:rm][mask]) / np.sum(Y[lm:rm][mask])
            dpkX = np.concatenate(([np.inf], np.diff(pkX), [np.inf]))
            
            j = np.where((dpkX[1:-1] <= oversegmentationfilter) & (dpkX[1:-1] <= dpkX[:-2]) & (dpkX[1:-1] < dpkX[2:]))[0]
            if j.size == 0:
                break
            left_min = np.delete(left_min, j + 1)
            right_min = np.delete(right_min, j)
            lfwhh = np.delete(lfwhh, j + 1)
            rfwhh = np.delete(rfwhh, j)
            
            val_max[j] = np.maximum(val_max[j], val_max[j + 1])
            val_max = np.delete(val_max, j + 1)
        signal_num = len(val_max)
        ## Area calculation
        pkA = np.empty((signal_num,))
        for idx in range(signal_num):
            sl = slice(left_min[idx],right_min[idx]+1,1)
            if min(Y[sl])<0:
                pkA[idx] = np.trapz(Y[sl] - min(Y[sl]),X[sl])
            else:
                pkA[idx] = np.trapz(Y[sl],X[sl])
        #peaklists[ns] = np.column_stack(([ind]*signal_num,pkX, val_max, pkA,val_max/noise,X[left_min], X[right_min],lfwhh, rfwhh,[noise]*signal_num,[mean_noise]*signal_num))  
        peaklists[ns] = np.column_stack(([ind]*signal_num,pkX, val_max, pkA,val_max/noise,X[left_min+1], X[right_min-1],lfwhh, rfwhh,[noise]*signal_num,[mean_noise]*signal_num))  
    #print_queue.put(f"{[spectra_ind]*sum(k)},\n{pkX[k]}, \n{val_max[k]},\n {pkA},\n{val_max[k]/noise,lfwhh[k]},\n {rfwhh[k]},\n{X[left_min[k]]}, \n{X[right_min[k]]},\n{noise},\n{mean_noise}")
    return np.vstack(tuple(peaklists.values()))
### Utility functions
def draw_data(data_obj_list,mz_diap4draw = None, num_specst = None):
    """
    Общее описание
    ----
    Функция для построения графиков интенсивностей и пик-листов из hdf5 в заданном диапазоне и в определённом спектре.

    :param data_obj_list: list of hdf5 objects
    :param mz_diap4draw: list of min and max range to draw graphs
    :param num_specst: num of the spectrum to draw. Otherwise it will be choosed randomly
    
    :type data_obj_list: list of paths to hdf5
    :type mz_diap4draw: `tuple` or `list`
    :type num_specst: `int`
    """
    for n, data_obj in enumerate(data_obj_list):
        for slide in data_obj.keys():
            for sample in data_obj[slide].keys():
                for roi in data_obj[slide][sample].keys():
                    if num_specst is not None:
                        num_spec = num_specst
                    else:
                        num_spec=np.random.randint(0,data_obj[slide][sample][roi]['xy'].len())
                    plt.figure().set_figwidth(25)
                    plt.gcf().set_figheight(5)
                    

                    print(f'Spectrum number: {num_spec}')
                    
                    try:
                        mz = data_obj[slide][sample][roi]["mz"][:]
                        intens = data_obj[slide][sample][roi]["int"][num_spec,:]
                        Label = "Processed spectra"
                    except:
                        mz = data_obj[slide][sample][roi]["mz_raw"][:]
                        intens = data_obj[slide][sample][roi]["int_raw"][num_spec,:]
                        Label = "Raw spectra"
            
                

                    if mz_diap4draw is not None:
                        diap=(np.array(mz>mz_diap4draw[0]) & np.array(mz<mz_diap4draw[1])) 
                    #diapnew=(np.array(mz[n]>mz_diap4draw[0]) & np.array(mz[n]<mz_diap4draw[1])) 
                    else:
                        diap=range(len(mz))
                    #diapnew=range(len(data_obj_procc[sample][roi]["mz"]))
                    plt.plot(mz[diap], intens[diap],alpha=0.75)
                    
                    try:
                        DataFeat=pd.DataFrame(data_obj[slide][sample][roi]['peaklists'][[0,1,2,5,6]], data_obj[slide][sample][roi]['peaklists'].attrs['Column headers'][[0,1,2,5,6]]).T
                        try:
                            DataFeat=DataFeat.astype({"spectra_ind": int})
                        except:
                            pass
                        DataFeat.query("mz>@mz_diap4draw[0] and mz<@mz_diap4draw[1] and spectra_ind == @num_spec").plot(x="mz",y="Intensity",ax = plt.gca(),style = "x", color='k')
                        left_intens=[]
                        for left_base in DataFeat.query("PextL>@mz_diap4draw[0] and PextL<@mz_diap4draw[1] and spectra_ind == @num_spec")['PextL']:
                            left_intens.append(intens[mz>left_base][0])
                        
                        right_intens = []
                        for right_base in DataFeat.query("PextR>@mz_diap4draw[0] and PextR<@mz_diap4draw[1] and spectra_ind == @num_spec")['PextR']:
                            right_intens.append(intens[mz<right_base][-1])
                        plt.plot(DataFeat.query("PextL>@mz_diap4draw[0] and PextL<@mz_diap4draw[1] and spectra_ind == @num_spec")['PextL'],
                        left_intens,'v', color='k')
                        plt.plot(DataFeat.query("PextR>@mz_diap4draw[0] and PextR<@mz_diap4draw[1] and spectra_ind == @num_spec")['PextR'],
                        right_intens,'^', color='k')
                    except:
                        pass
                    plt.grid(visible=True,which="both")

                    plt.legend([Label,"Peaks", 'Left peak base','Right peak base'])
                    plt.minorticks_on()
                    plt.xlabel("m/z")
                    plt.ylabel("Intensity")
                    plt.title(f"Slide: {slide}, sample: {sample}, roi: {roi}")
                    plt.show()

def draw_processing_example(data_obj_path, spec_num=None, baseliner_algo = 'asls', params2baseliner_algo={}, #penalized_poly - самый быстрый вариант. asls - меньше "отрицательных" точек по сравнению с penalized_poly, что лучше работает с пикпикингом с фильтрацией порогом по интенсивности, но в ~2 раза дольше считает
              align_peaks = None, weights_list=None, max_shift_mz=0.95,only_shift=True,params2align={},
              resample_to_dots = None, 
              smooth_algo = None, smooth_window=0.075, smooth_cycles=1,
              oversegmentationfilter = 0, fwhhfilter = 0, heightfilter=0, peaklocation=1,rel_heightfilter=0,
              SNR_threshold = 3.5, noise_est = "std",noise_est_iterations = 3,
              mz_diap4draw = None,dtypeconv='single'):
    """
    Общее описание
    ----
    Функция позволяет быстро оценить удовлетворительность результатов обработки спектров и пикпикинга для подобранных параметры. Функция работает аналогично Raw2peaklist, но только обрабатывает один случайный спектр во всех sample и roi и после строит график для оценки.

    :param data_obj_path: list of paths to root folders where to search imzml files in subfolders 
    :param baseliner_algo: Algorithm of baseline correction. Default: `"asls"`

        Fastest: `"penalized_poly"`.

        Optimal: `"asls"`. Slower, but intensities less frequently corrected to values <0

        See other algorithms: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html#
    :param params2baseliner_algo: dictionary of parametres for baseline correction algorithm (see: https://pybaselines.readthedocs.io/en/latest/api/Baseline.html). Default: `{}`

        .. Example: {"lam" : 500000, "diff_order" : 1}
    :param align_peaks: list of reference peaks for align. Default: `None`
    :param weights_list: list of weights for reference peaks in aligning. Default: `None`
    :param max_shift_mz: max spectrum shift at aligning in mz. Default: `0.95`
    :param params2align: Dictionary of parametres for aligning (see params: `align.py` in class `Aligner`). Default: `{}`

        .. Example: {"iterations" : 2, "only_shift" : False}
    :param resample_to_dots: resample spectra to number of dots. Default: `None`
    :param smooth_algo: spectrum smoothing algorithm. Default is `"None"`
        
        `"GA"` - is for gaussian

        `"MA"` - is for moving average

        `"SG"` - is for Savitzki-Golay (doesn't work for now)
    :param oversegmentationfilter: фильтр для близких друг к другу пиков. Default `0`
    :param fwhhfilter: Фильтр пиков по ширине на полувысоте пиков больше указанного значения. Default is `0`
    :param heightfilter: Фильтр пиков по абсолютному значению интенсивности ниже указанного значения. Default is `0`
    :param peaklocation: Параметр фильтрации пиков с oversegmentationfilter. Default is `1`
    :param rel_heightfilter: Фильтр пиков по относительному значению интенсивности. Default is `0`
    :param SNR_threshold: Фильтр пиков по их SNR. Default is `3.5`
    :param noise_est: алгоритм оценки шума. Пока только `std` и `mad` и для ускорения рассчётов, подсчёт идёт сразу по всему спектру в несколько итераций, где после каждой итерации определяются какие точки относятся к шуму, а какие к сигналу. Default is `"std"`
    :param noise_est_iterations: количество итераций определения шума. Оптимально более 3 итераций. Default is `3`
    :param smooth_window: window size in mz for smooth. Default:`0.075`
    :param smooth_cycles: Number of iterations for spectrum smooth. Default: `1`
    :param mz_diap4draw: Range for graphs draw. Default: `None`
    :param dtypeconv: convert data to `"double"`,`"single"` or `"half"` float type. The default is `"single"`

    :type data_obj_path: `list`
    :type max_shift_mz: `float`
    :type resample_to_dots: `int`
    :type baseliner_algo: `str`
    :type params2baseliner_algo: `dict`
    :type params2align: `dict`
    :type align_peaks: `list`
    :type weights_list: `list` or `pd.Series`
    :type dots_shift: `float`
    :type smooth_algo: {`"GA"`,`"MA"`,`"SG"`,`None`}
    :type oversegmentationfilter: `float`
    :type fwhhfilter: `float`
    :type heightfilter: `float`
    :type peaklocation: `float` and =<1
    :type rel_heightfilter: `float`
    :type SNR_threshold: `float`
    :type noise_est: {`"std"`,`"mad"`}
    :type noise_est_iterations: `int`
    :type smooth_window: `float`
    :type smooth_cycles: `int`
    :type mz_diap4draw: `list` or `None`
    :type dtypeconv: {`"double"`,`"single"`, `"half"`}

    :return: `None`
    :rtype: `NoneType`
    """

    logger("draw_processing_example",{**locals()})
    # Process args
    #defaults parametres for align
    pars = list(set(["width","iterations"])-set(params2align.keys()))
    if pars:
        params2align_default = {"iterations":3, "width":0.3}
        for par in pars:
            params2align[par]=params2align_default[par]
    params2align["only_shift"]=only_shift

    if not isinstance(peaklocation, (int, float)) or not np.isscalar(peaklocation) or peaklocation < 0 or peaklocation > 1:
        raise ValueError("mspeaks: Invalid peak location")
    if not isinstance(fwhhfilter, (int, float)) or not np.isscalar(fwhhfilter) or fwhhfilter < 0:
        raise ValueError("mspeaks: Invalid FWHH filter")
    if not isinstance(oversegmentationfilter, (int, float)) or not np.isscalar(oversegmentationfilter):
        if isinstance(oversegmentationfilter, str):
            oversegmentationfilter=oversegmentationfilter.lower()
        else:
            raise ValueError("mspeaks: Invalid oversegmentation filter")
    elif oversegmentationfilter < 0:
        raise ValueError("mspeaks: Invalid oversegmentation filter")
    if not isinstance(heightfilter, (int, float)) or not np.isscalar(heightfilter) or heightfilter < 0:
        raise ValueError("mspeaks: Invalid height filter")
    if not isinstance(rel_heightfilter, (int, float)) or not np.isscalar(rel_heightfilter) or rel_heightfilter < 0 or rel_heightfilter > 100:
        raise ValueError("mspeaks: Invalid relative height filter")

    if isinstance(data_obj_path,str):
        data_obj_path=[data_obj_path]

    ###I Finding slide directory with rawdata of samples (path_list) - DONE
    if noise_est == "MAD":
        noise_func= MAD
    elif noise_est == "std":
        noise_func=np.std
    # Working with slides
    for file_path in data_obj_path:
        sample_list =[]
        # Searching direct path to imzml files (samples) 
        if file_path.lower().endswith('.imzml'):
            sample_list.append(file_path)
        for root, dirs, files in os.walk(file_path):
            for file in files: 
                if file.lower().endswith('.imzml'):
                    sample_list.append(os.path.join(root,file))

        for sample_path2imzml in sample_list:
            folder_path2imzml = os.path.dirname(sample_path2imzml)
            sample_name = os.path.splitext(os.path.basename(sample_path2imzml))[0]
            folder_name = os.path.basename(folder_path2imzml)

            if folder_name == sample_name:

                sample=sample_name
                
            else:

                sample = folder_name+"_"+sample_name
            logger.log('imzml opening')
            sample_imzml=ImzMLParser(sample_path2imzml)
            logger.log('imzml opened')
            ### 1. Файл найден и открыт в sample_imzml файле - DONE
            ### 1. File found and opened in sample_imzml file - DONE
            ### Data extraction
            try: ### b. Extraction from _poslog and _info text files
                count=0
                idx_first=0
                roi_idx = {} 
                roi_idx[sample]={} # Информация sample по индексам спектров roi=(индекс первого спектра, кол-во спектров roi)

                base_path_str = os.path.join(folder_path2imzml,sample_name)
                roi_list = []
                try:
                    with open(base_path_str+"_info.txt") as f:
                        data_info = f.readlines()
                        #raw_data_points = int(data_info[12].split(' ')[1]) # Информация по кол-ву точек спектра
                        spectra_num = int(data_info[2].split(' ')[-1]) # Информация по кол-ву спектров в sample
                except:   
                    #raw_data_points = int(np.quantile(sample_imzml.mzLengths, 0.95))
                    spectra_num = len(sample_imzml.mzLengths)
                # try:
                #     dcont = sample_imzml.metadata.pretty()["file_description"]["continuous"]

                # except KeyError:
                #     dcont = not sample_imzml.metadata.pretty()["file_description"]["processed"]

                ###
                ### Так как файлы imzml с poslog исключительно континуальные (одна шкала mz для всех спектров), то здесь работаем исключительно в таком варианте
                ### Выгрузка данных с poslog
                ### 4.b. and 5.b. Extraction data from "poslog" coordinates and roi references of spectra
                with open(base_path_str+"_poslog.txt") as f:
                    data = f.readlines()

                    poslog_specdata = [None]*spectra_num #Данные строк в poslog с записью roi и координат снятого спектра.
                    ##первая итерация записи координат начиная с третьей строки
                    coords =  data[2].split(' ') 
                    roi_num = re.search('R(.+?)X', data[2]).group(1)
                    roi_list.append(roi_num)
                    poslog_specdata[count]=(roi_num,float(coords[-3]), float(coords[-2]))
                    
                    roi_idx[sample][roi_num] = idx_first
                    
                    count+=1
                    ## продолжение итераций    
                    for i in range(2,len(data)-1):
                        coords =  data[i+1].split(' ')
                        
                        if(coords[-4]!='__'):
                            roi_num = re.search('R(.+?)X', data[i+1]).group(1)
                            poslog_specdata[count]=(roi_num,float(coords[-3]), float(coords[-2]))
                            
                            ### Строгое положение из-за roi_list[-2] и условия в if
                            if roi_num not in roi_list[-1]:
                                roi_list.append(roi_num)
                                roi_idx[sample][roi_num] = []
                                roi_idx[sample][roi_list[-2]] = (idx_first, count-idx_first)
                                idx_first=count
                            ###
                            count +=1
                    roi_idx[sample][roi_num] = (idx_first, count-idx_first) ### 2.b. Num of spectra of roi/sample

            except FileNotFoundError: 
                roi = "00" # roi только один, так как там вроде нельзя настраивать и определять без poslog
                roi_list = []
                roi_list.append(roi)
                roi_idx[sample][roi] = (idx_first, spectra_num)

            plt.figure().set_figwidth(25)
            plt.gcf().set_figheight(5)
            for roi in roi_list:
                idx_start, numspec = roi_idx[sample][roi]
                if spec_num:
                    idx_spec=spec_num
                else:
                    idx_spec = np.random.randint(idx_start,idx_start+numspec)
                    
                print(f'Spectrum number: {idx_spec}')
                logger.log('getting spectra for draw')
                data_mz_old, data_int_old = sample_imzml.getspectrum(idx_spec)
                logger.log('getting spectra for draw ended')
                #data_int_old.shape = (1,data_int_old.shape[0])
                roi_idx_spec = idx_spec-idx_start
                loc_args2procc={"baseliner_algo": baseliner_algo, "params2baseliner_algo": params2baseliner_algo,"params2align":params2align, "align_peaks":align_peaks,"weights_list":weights_list,"smooth_algo":smooth_algo, "smooth_cycles":smooth_cycles}
                if resample_to_dots:
                    data_mz = np.array(list(np.linspace(min(data_mz_old),max(data_mz_old),resample_to_dots)))
                    if loc_args2procc["params2align"]["only_shift"]:
                        dots_shift = int(max_shift_mz/(np.median(np.diff(data_mz))))
                    else:
                        dots_shift = max_shift_mz
                    loc_args2procc['dots_shift']=dots_shift
                    loc_args2procc["smooth_window"]=int(smooth_window/(np.median(np.diff(data_mz))))
                    data_int = DataProc_resample1d(data_int_old,data_mz_old,data_mz,Baseline(data_mz),**loc_args2procc)

                else:
                    data_mz = data_mz_old
                    if loc_args2procc["params2align"]["only_shift"]:
                        dots_shift = int(max_shift_mz/(np.median(np.diff(data_mz))))
                    else:
                        dots_shift = max_shift_mz
                    loc_args2procc['dots_shift']=dots_shift
                    loc_args2procc["smooth_window"]=int(smooth_window/(np.median(np.diff(data_mz))))
                    data_int = DataProc_base1d(data_int_old,data_mz,Baseline(data_mz),**loc_args2procc)

                dataf = mspeaks_opt(data_mz, data_int, [idx_spec-idx_start],oversegmentationfilter=oversegmentationfilter, fwhhfilter=fwhhfilter,heightfilter=heightfilter,peaklocation=peaklocation,rel_heightfilter=rel_heightfilter,noise_func = noise_func,noise_est_iterations=noise_est_iterations,SNR_threshold=SNR_threshold)
                dataf = pd.DataFrame(dataf.T, ["spectra_ind","mz","Intensity","Area","SNR","PextL","PextR","FWHML","FWHMR","Noise","Mean noise"]).T
                dataf = dataf.astype({"spectra_ind": int})
                if mz_diap4draw:

                    diapold=(np.array(data_mz_old>mz_diap4draw[0]) & np.array(data_mz_old<mz_diap4draw[1]))
                    diap = (np.array(data_mz>mz_diap4draw[0]) & np.array(data_mz<mz_diap4draw[1]))
                    dataf.query("mz>@mz_diap4draw[0] and mz<@mz_diap4draw[1] and spectra_ind==@roi_idx_spec").plot(x="mz",y="Intensity",ax = plt.gca(), style = "x")
                    startg = mz_diap4draw[0]
                    endg = mz_diap4draw[1]
                else:
                    diapold=range(len(data_mz_old))
                    diap = range(len(data_mz))
                    startg = min(data_mz)
                    endg = max(data_mz)
                    dataf.query("spectra_ind==@roi_idx_spec").plot(x="mz",y="Intensity",ax = plt.gca(), style = "x")
                
                
                plt.plot(dataf.query("PextL>@startg and PextL<@endg and spectra_ind==@roi_idx_spec")['PextL'],
                        [0]*len(dataf.query("PextL>@startg and PextL<@endg and spectra_ind==@roi_idx_spec")['PextL']),'v')
                plt.plot(dataf.query("PextR>@startg and PextR<@endg and spectra_ind==@roi_idx_spec")['PextR'],
                        [0]*len(dataf.query("PextR>@startg and PextR<@endg and spectra_ind==@roi_idx_spec")['PextR']),'^')
                plt.plot(data_mz_old[diapold], data_int_old[diapold])
                plt.plot(data_mz[diap], data_int[diap])
                plt.grid(visible = True, which="both")
                plt.ylabel('Intensity')
                plt.title(f'Sample: {sample}, roi: {roi}')
                plt.legend(['Peaks', 'Peak`s left base', 'Peak`s right base', 'Original spectrum','Processed spectrum'])
                plt.minorticks_on()
                plt.xlim((startg,endg))
                plt.show()
                del diapold
    return
