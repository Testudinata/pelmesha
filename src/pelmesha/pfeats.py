import pandas as pd
import numpy as np
from itertools import product
from pelmesha.filling import specdata_Load, IMGfeats_concat, feat2DF, peakl2DF, logger, hdf5_metadata, del_hdf5
from pelmesha.utensils import split_pdtable_by_peaks_gap
import matplotlib.pyplot as plt
from h5py import File
from KDEpy import FFTKDE, TreeKDE
from pyimzml.ImzMLParser import ImzMLParser
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
import gc
import math
import os
import warnings
from functools import partial
try:
    from torch.multiprocessing import Pool, cpu_count
except Exception as error:
    warnings.warn(f"During import torch.multiprocessing package raised error {error}. Using python package multiprocessing instead")
    from multiprocessing import Pool, cpu_count
## pairwise for python versions below 10 
from sys import version_info
if version_info[0] < 3:
    raise Exception("Must be using Python 3")
else:
    if version_info[1]<10:
        from itertools import tee

        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)
    else:
        from itertools import pairwise

# from memory_profiler import profile # TODO: удалить перед релизом


FWHM_TO_SIGMA_FACTOR = 1 / np.sqrt(8 * np.log(2))  # Фактор пересчета FWHM в sigma


def Pgrouping_KD(ftable, extr_columns = None, KD_bandwidth = "med_fwhm", bwc = 1,KD_kernel = "gaussian", KD_func = None,CountF = 10, norm = (None,None), draw_borders = 1.5, 
                  dupl_drop = True,min_res = 10, pivoting4val = None, cpu_free=1, path2save=None,sample="unknwn",roi="00", coords4table=None, account_mzscale = True, draw=True, 
                  split_mz_min = 10, split_peaks_min = 25, **params2mspeaks_KD):
    """    
    Описание
    ----
    Функция группирует близкостоящие значения пиков mz к определённому значению Peak на основе оценки плотности вероятности встречаемости mz.  

    :param ftable: Источник данных. Может быть уже таблицей/датафреймом, а может быть ссылкой на папку с данными, которые надо выгрузить
    :param extr_columns: Лист столбцов, которые войдут в датафрейм фич, где `"spectra_ind"` и `"mz"` или `"Peak"` - экстрагируются всегда. Default: `None` - экстракция всех столбцов
    `"Intensity"`, `"Area"`, `"SNR"`, `"PextL"`, `"PextR"`, `"FWHML"`, `"FWHMR"`,`"Noise"`, `"Mean noise"`
    :param KD_bandwidth: {`"med_FWHM"`,`"mz_discret"`,`"ISJ"`,`"silverman"`,`"scott"`, `float`} - выбор полосы пропускания, либо алгоритм её определения.
    
        `"med_FWHM"` - полоса пропускания - медианное значение дисперсии пиков, которая определяется из медианы ширины на полувысоте. 
    
        `"mz_discret"` - полоса пропускания - медиана расстояний между точками шкалы mz, где лучше всего, если есть доступ к файлу hdf5 и imzml. 

        `"ISJ"`,`"silverman"`,`"scott"` - базовые для функции FFTKDE (из пакета KDEpy)
    :param bwc: коэффициент полученной/выбранной полосы пропускания
    :param KD_kernel: KDE ядро (see also: https://kdepy.readthedocs.io/en/latest/kernels.html#available-kernels)
    :param CountF: Параметр исключение из датасета редких пиков, чьё кол-во меньше данного значения
    :param params2mspeaks_KD: остальные параметры для функции mspeaks_KD
    :param norm: Default (None, None). First is type of normalization on spectrum: "l1", "l2", "max". Second is which column normalize (str or list)
    :param draw_borders: graphics borders extension ± m/z
    :param dupl_drop: `True` - полученная фича таблица фильтрует дупликаты пиков.
    :param min_res: минимальное разрешение прибора в ppm. Контролирует минимальное значение полосы пропускания для метода её определения `"mz_discret"` и максимальное кол-во точек при построении оценки плотности вероятности
    :param pivoting4val: list of columns or None (default) - resulted table is pivoted by index: spectra_ind, columns: Peak with fill_value = 0, and values: list of columns from pivoting4val. If None - do nothing about pivoting
    :param cpu_free: Number of CPU cores don't used in multiproccessing
    :param path2save: path to save if the ftable is not a path
    :param sample: works with path2saveand ftable is pandas DataFrame. It's a name for sample group in writing hdf5 file. Default: "unknwn"
    :param roi: works with path2save and ftable is pandas DataFrame. It's a name for roi group in writing hdf5 file. Default: "00"
    :param coords4table: works with path2save and ftable is pandas DataFrame. It's coordinates for saving in hdf5 file. Default: None

    :type ftable: `pd.Dataframe` or `path`/`str`
    :type columns: `list` or `None`
    :type KD_bandwidth: `float` or `str`
    :type bwc: `float`
    :type KD_kernel: `str`
    :type CountF: `int`
    :type params2mspeaks: `keyword` 
    :type norm: `tuple`
    :type draw_borders: `float`
    :type dupl_drop: `bool`
    :type min_res: `float`
    :type pivoting4val: `bool`
    :type cpu_free: `int`
    :type path2save: `str`
    :type sample: `str`
    :type roi: `str`
    :type coords4table: `pd.Dataframe` or `None`


    :return: peaklist DataFrame, where slide, sample and roi are in index. Return `tuple` if `extract_coords` is `True` with additional Coords

    :rtype: `dict` or `tuple`
    """
    split_params = {'split_mz_min': split_mz_min, 'split_peaks_min': split_peaks_min}
    logger("Pgrouping_KD",{**locals()},path2save)

    if not isinstance(extr_columns,list) and extr_columns:
        extr_columns = [extr_columns]
    cpu_num = cpu_count()-cpu_free
    min_res = min_res/1e+6
    if extr_columns:
        if KD_bandwidth.lower() =='med_fwhm':
            extr_columns=extr_columns+["FWHML", "FWHMR"]
        extr_columns=list(set(extr_columns))
        
    # Вариация для рассчётов для уже готовой таблицы или путей к таблицам features 
    if isinstance(ftable, str):
        ftableISAPATH = True
        data, path = peakl2DF([ftable],extr_columns=extr_columns, return_source_path = True,pivoting4val=None)
    elif isinstance(ftable, list):
        data, path = peakl2DF(ftable,extr_columns=extr_columns, return_source_path = True,pivoting4val=None)
        ftableISAPATH = True
    if isinstance(ftable, pd.DataFrame):
        ftableISAPATH = False

    if ftableISAPATH:
        logger.log(f"Data proccessed from file")
        ftable = Pgrouping_KD_file(data, path, 
                                   cpu_num, 
                                   KD_bandwidth, bwc, KD_kernel, KD_func, 
                                   CountF, norm,
                                   draw_borders, dupl_drop, split_params, account_mzscale,
                                   draw,
                                   **params2mspeaks_KD)
        if pivoting4val:
            for slide in ftable.keys():
                for sample in ftable[slide].keys():
                    for roi in ftable[slide][sample].keys():
                        ftable[slide][sample][roi]['features'] = ftable[slide][sample][roi]['features'].pivot_table(index="spectra_ind", columns="Peak",fill_value = 0, values =pivoting4val)
    else:
        
        # logger.log(f"Peaks are in range {plot_start}-{plot_end} with dtype{type(plot_start)}. Median distance: {median_dist}")
        
        logger.log(f"Data proccessed from table")
        
        ftable = Pgrouping_KD_table(ftable,
                                    cpu_num,
                                    KD_bandwidth, bwc, KD_kernel, KD_func, 
                                    CountF, 
                                    norm,
                                    draw_borders,
                                    dupl_drop, 
                                    split_params, 
                                    #discret_coeffs, 
                                    sample=sample,
                                    roi=roi, 
                                    account_mzscale = account_mzscale,
                                    draw=draw,
                                    **params2mspeaks_KD)
        if path2save:
            logger.log(f"Starting saving grouped data")
            try:
                path2hdf5 = os.path.join(path2save,f"{sample}_{roi}")+"_features.hdf5"
                del_hdf5(path2hdf5)
                hdf5file = File(path2hdf5, mode="a")
                hdf5file.create_dataset(sample + "/" + roi + "/features",data = ftable)
                logger.log(f"Grouped table data saved in hdf5")
                hdf5file[sample][roi]["features"].attrs["Column headers"] = list(ftable.columns)
                logger.log(f"Table headers saved in hdf5")

                if coords4table is not None:

                    hdf5file.create_dataset(sample + "/" + roi + "/xy",data = coords4table.values)
                    logger.log(f"Added coordinates saved in hdf5 succesfully")
                # try:
                #     hdf5file.create_dataset(sample + "/" + roi + "/z",data = Grouped_ftable[slide][sample][roi]["z"]) z-coordinates?
                # except:
                #     pass                        

                print(f"Processed features of sample {sample} roi {roi} is saved in hdf5 file")
                hdf5file.close()
                
                unique_num = Pgrouping_KD_table.unique_num
                num_bf_raredel = Pgrouping_KD_table.num_bf_raredel 
                num_raredel = Pgrouping_KD_table.num_raredel 
                num_res = Pgrouping_KD_table.num_res
                with open(os.path.join(path2save,"Peaks_grouping_settings.txt"), "a") as file:
                    file.write(f"\n\n##Grouping results of {sample} roi: {roi}\n")
                    file.write(f"KDE bandwidth: {KD_bandwidth}\n")
                    file.write(f"Grouping results:\nNumber of unique peaks before grouping: {unique_num}\nNumber of unique peaks after grouping: {num_bf_raredel}\nNumber of excluded peaks by count filter({CountF}): {num_raredel} ({num_raredel*100/num_bf_raredel:.2f}%)\nResulted feature peaks is {num_res}")             
            except:
                print(f"Processed features of sample {sample} roi {roi} doesn't saved in hdf5 file")
        if pivoting4val:
            ftable = ftable.pivot_table(index="spectra_ind", columns="Peak",fill_value = 0, values =pivoting4val)

    logger.ended()
    return ftable

def Getrefpeaks(ref_rois_paths,step=100,num_peaks_per_step=5, min_occurence = 0.1, return_weight=True,**Pgrouping_KD_kwargs):
    """    
    Описание
    ----
    Функция для получения списка референсных пиков на основе группировки пиков функции Pgrouping_KD. Пики выбираются из имаджа по всему диапазону mz. Диапазон mz разбивается на зоны с шагом step. Из каждого диапазона выбирается num_peaks_per_step самых часто встречаемых в имадже.   
    
    :param ref_rois_paths: dict = {path_1:[[sample_1,[roi_list_1]],[sample_2,[roi_list_2]],....],path_2:[[sample_3,[roi_list_3]],[sample_4,[roi_list_4]],....]}, "path" - path to hdf5 file directory, "sample_n" - какой именно sample (string), если None - берёт всё, "roi_list_n" - список каких roi использовать, если отсутствует, то берёт всё (example: dict value: list[sample_n])
    or list[path_1,path_2], "path" - path to hdf5 file directory, используются все sample и roi в них.
    :param step: промежуток, в котором выбираются наиболее частовстречаемые пики
    :param num_peaks_per_step: количество пиков, выбираемые из каждого промежутка
    :param min_occurence: минимальная частота встречаемости пиков для отбора
    :param return_weight: Также возвращается вес пиков
    :param Pgrouping_KD_kwargs: остальные значения параметров, соответствующие функции Pgrouping_KD при группировке пиков

    :type ref_rois_paths: `dict`,`path`,`list` or `str`
    :type step: `int`
    :type num_peaks_per_step: `int`
    :type min_occurence: `float`

    :return: Возвращает список референсных пиков или ещё и веса этих пиков на основе их встречаемости. 
    :rtype: `list` or tuple(`list`,`pd.Series`,`pd.DataFrame`) 
    """
    logger("Getrefpeaks",{**locals()})
    if isinstance(ref_rois_paths, pd.DataFrame):
        ref = ref_rois_paths
    else:
        ref = IMGfeats_concat(ref_rois_paths,extr_columns=None,extracts_coords=False,processed_feat = False)
    if 'Peak' not in ref.columns:
        ref = Pgrouping_KD(ref.reset_index(drop=True),**Pgrouping_KD_kwargs)
    ref_peaks = ref.pivot_table(columns= 'Peak', fill_value = 0,aggfunc={"spectra_ind":'count'}).rename({'spectra_ind':'P_occurence'}).astype(float)
    ref_peaks.loc['P_occurence'] = ref_peaks.loc['P_occurence']/ref_peaks.loc['P_occurence'].max()
    ref_peaks = ref_peaks.loc[:,ref_peaks.loc['P_occurence']>min_occurence]
    ref_peaklist = list(ref_peaks.columns)
    
    align_list = []
    ## Getting list of a most common peaks in step 
    for lim in pairwise(np.arange(min(ref_peaklist),max(ref_peaklist)+step+1,step)):
        align_list+=list(ref_peaks.loc[:,(ref_peaks.columns>=lim[0]) & (ref_peaks.columns<lim[1])].sort_values(by="P_occurence",axis=1).T.tail(num_peaks_per_step).index)
    ## Getting weights of choosed peaks
    if return_weight:
        weight_list=ref_peaks.loc["P_occurence",align_list]
        logger.log("Output weight list:")
        logger.log(weight_list)
        logger.ended()
        return align_list,weight_list
    else:
        logger.log("Return only align list")
        logger.ended()
        return align_list

def Roi_Pgrouping_KD(Paths, extr_columns=None,path2save=None,**Pgrouping_KD_kwargs):
    """
    Описание
    ----
    Функция объединяет указанные в paths датасеты (вплоть до одного roi из всех sample и roi в файле). В объединённых датасетах с помощью функции Pgrouping_KD формируются выравненные для всех датасетов значения mz пиков. Рекомендуется использовать датасеты предварительно выравненные одними и теми же референсными пиками, захватывающих весь диапазон пик-листов.
    
    :param paths: dict = {path_1:[[sample_1,[roi_list_1]],[sample_2,[roi_list_2]],....],path_2:[[sample_3,[roi_list_3]],[sample_4,[roi_list_4]],....]}, "path" - path to hdf5 file directory, "sample_n" - какой именно sample (string), если None - берёт всё, "roi_list_n" - список каких roi использовать, если отсутствует, то берёт всё (example: dict value: list[sample_n])
    или list[path_1,path_2], "path_n" - path to hdf5 file directory, используются все sample и roi в них.
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где экстрагируются всегда `"spectra_ind"` и `"mz"` или `"Peak"`. Default: `None` - экстракция всех столбцов
    `"Intensity"`,`"Area"`, `"SNR"`,`"PextL"`,`"PextR"`, `"FWHML"`, `"FWHMR"`,`"Noise"`,`"Mean noise"`
    :param Pgrouping_KD_kwargs: остальные значения параметров, соответствующие функции Pgrouping_KD при группировке пиков
    :param path2save: путь, куда сохранить объединённые данные в файл с названием "Images_{количество_имаджей}_grouped_MSIdata.hdf5". Если `None`, то данные не будут сохранены в файл.
    :param pivoting4val: list of columns or None (default) - resulted table is pivoted by index: `spectra_ind`, columns: `Peak` with fill_value = 0, and values: list of columns from pivoting4val. If `None` - do nothing about pivoting.

    :type paths: `dict`,`path`,`list` or `str`
    :type extr_columns: `list`
    :type path2save: `path` or `str`

    :return: возвращает tuple([таблица пиков],[Координаты имаджа])
    :rtype: tuple(`pd.DataFrame`,`dict`) 
    """
    if path2save:
        os.makedirs(path2save, exist_ok=True)
    logger("Roi_Pgrouping_KD",{**locals()},path2save)
    #if "path2save" not in Pgrouping_KD_kwargs.keys():
    #    Pgrouping_KD_kwargs["path2save"]=path2save

    if isinstance(extr_columns,(str)):
        extr_columns = [extr_columns]
    if  Pgrouping_KD_kwargs.get("KD_bandwidth", "med_FWHM") == "med_FWHM" and extr_columns is not None:
        extr_columns = list(set(extr_columns + ['FWHML',"FWHMR"]))
        extr_columns.sort()
    logger.log('Additional extraction columns list of values: "Intensity","Area", "SNR","PextL","PextR","FWHML", "FWHMR","Noise","Mean noise"')
    logger.log(f'Additional extraction columns:{extr_columns}')
    Total_peaks, coords = IMGfeats_concat(Paths,extr_columns,extracts_coords=True,processed_feat = False)

    if "pivoting4val" in Pgrouping_KD_kwargs.keys():
        if isinstance(Pgrouping_KD_kwargs["pivoting4val"], str):
            Pgrouping_KD_kwargs["pivoting4val"]=Pgrouping_KD_kwargs["pivoting4val"]
        pivoting4val = Pgrouping_KD_kwargs["pivoting4val"]
        del Pgrouping_KD_kwargs["pivoting4val"]
        # logger.info(f"'pivoting4val'= {pivoting4val}. Data saved in hdf5 is not pivoted")
    else:
        pivoting4val = None

    Aligned_rois = Pgrouping_KD(Total_peaks,**Pgrouping_KD_kwargs)
    ########## under construction
    indexes = Aligned_rois.index.unique()
    image_num = len(list(indexes))
    slides=[]
    for index in indexes:
        slides.append(index[0]) 
    slides = list(set(slides))
    slide_naming=''
    for slide in slides:
        slide_naming = slide_naming +"_"+slide
    if path2save:
        path2file = os.path.join(path2save,f"Images_{image_num}_from_slides"+ slide_naming+"_grouped_MSIdata.hdf5")
        del_hdf5(path2file)
        hdf5file = File(path2file, mode="a")
        logger.log("Grouped features is saved in hdf5 file of images:")
        for index in indexes:
            try:
                hdf5file.create_dataset(index[0] + "/" + index[1] + "/"+ index[2] + "/features", data = Aligned_rois.loc[index])
                hdf5file.create_dataset(index[0] + "/" + index[1] + "/"+ index[2] + "/xy", data = coords.loc[index])
                logger.log(f"Data saved: {list(index)}")
            except Exception as error:
                logger.warn(f"Error on {index}: \n{error}\nCoords: {coords.loc[index].info()}\nAligned_rois: {Aligned_rois.loc[index].info()}")
        hdf5file.attrs["Column headers"] = list(Aligned_rois.columns)
        print(f"Grouped features is saved in hdf5 file")
        hdf5file.close()

    ########## under construction
    if pivoting4val:
        Aligned_rois = Aligned_rois.pivot_table(index=[Aligned_rois.index,'spectra_ind'], columns="Peak",fill_value = 0, values =pivoting4val)
        Aligned_rois.index.names = ['MS_image','spectra_ind']
        logger.log(f"Returned feature table is pivoted")

    logger.log(f'Data used:')
    try:
        for path in Paths.keys():
            logger.log(f'\tfrom {path} used sample:')
            for sample in Paths[path]:
                if sample is None:
                    logger.log(f'\t\tfull data')
                else:
                    try: 
                        logger.log(f'\t\t{sample[0]} with rois {sample[1]}')
                    except:
                        logger.log(f'\t\t{sample[0]} with all rois')
                    
    except:
        for path in Paths:
            logger.log(f'\t\tfull data')

    logger.ended()
    return Aligned_rois, coords

### utils functions
def Pgrouping_KD_file(data, path,
                      cpu_num, 
                      KD_bandwidth, bwc, KD_kernel, KD_func,
                      CountF, norm, 
                      draw_borders, 
                      dupl_drop, split_params, account_mzscale, 
                      draw,
                      **params2mspeaks_KD):
    """    
    Описание
    ----
    Вспомогательная функция к основной `Pgrouping_KD`, которая работает с обработкой данных из файла, а также проводит запись итоговых данных в файл.  
    """
    Grouped_ftable = {}
    for slide in data.keys():
        Grouped_ftable[slide] = {}
    
        with open(os.path.join(os.path.dirname(path[slide]),"Peaks_grouping_settings.txt"), "w") as file:
            file.write(f"##General grouping settings\n")
            file.write(f"KD_bandwidth estimation: {KD_bandwidth} (or value mz)\nBandwith coeff: {bwc} \nKD kernel: {KD_kernel}\n")    
            file.write(f"Peaks count filter: {CountF}\nduplicated peaks dropping: {dupl_drop}\n")
            file.write(f"\nNormalizing data after grouping: type of normalization {norm[0]}, column: {norm[1]}")
        
        hdf5_feat_path = "_".join(path[slide].split("_")[:-1])+"_features.hdf5"

        del_hdf5(hdf5_feat_path)
        with File(path[slide]) as donor:# !!!! Или непосредственно взять из донора нужные метаданные (но универсальный перепаковщик всё ещё нужен)
            hdf5_metadata(hdf5_feat_path, donor)
        for sample in data[slide].keys():
            Grouped_ftable[slide][sample] = {}
            for roi in data[slide][sample].keys():
                Grouped_ftable[slide][sample][roi] = {}
                ftable = data[slide][sample][roi]["peaklists"]
        
                spec_data = specdata_Load(path[slide])
                # discret_coeffs = spec_data[slide][sample][roi].attrs['discret_coeffs']
                try:
                    source='imzml'
                    spectrums_source = spec_data[slide][sample][roi].attrs['source']
                    idx_start, num_spec = spec_data[slide][sample][roi].attrs['idxroi']
                    spec_data = ImzMLParser(spectrums_source)
                    
                except:
                    source='hdf5'

                Grouped_ftable[slide][sample][roi]["features"] = Pgrouping_KD_table(ftable, cpu_num,
                                                                                    KD_bandwidth, bwc, KD_kernel, KD_func,
                                                                                    CountF, norm, 
                                                                                    draw_borders, 
                                                                                    dupl_drop, split_params,# discret_coeffs,
                                                                                    sample, roi, account_mzscale,
                                                                                    draw, 
                                                                                    **params2mspeaks_KD)  
                if draw:
                    rand_spec_1 = Pgrouping_KD_table.rand_spec_1
                    rand_spec_2 = Pgrouping_KD_table.rand_spec_2
                    try:
                        if data[slide][sample][roi].get('int_mean', False):
                            mean_fmz = data[slide][sample][roi]['mz_scale']
                            mean_fint = data[slide][sample][roi]['int_mean']
                        else:
                            mean_fmz, mean_fint = get_mean_spectrum(spec_data[slide][sample][roi].attrs['source'], spec_data[slide][sample][roi].attrs['idxroi'])[roi]
                        mean_leg = ['Mean spectrum']
                    except Exception as e:
                        print(f'Error occur during extraction mean intensity: {e}')
                        mean_fmz = None
                        mean_fint = None
                        mean_leg = []
                    
                    if rand_spec_1:
                        rand_spec = rand_spec_1[-1]
                        plt.figure(num=plt.get_fignums()[-2])
                        try:
                            x = spec_data[slide][sample][roi]["mz"][:]        
                            y = spec_data[slide][sample][roi]["int"][rand_spec,:]

                        except:
                            x,y = spec_data.getspectrum(rand_spec+idx_start)
                        
                        mz_draw_borders = plt.xlim()
                        dots_bord_spec = (np.array(x)>=mz_draw_borders[0]) & (np.array(x)<=mz_draw_borders[1])
                        ax = plt.gca().twinx()
                        ax.plot(np.array(x)[dots_bord_spec],np.array(y)[dots_bord_spec],alpha=0.85)
                        #mean spectrum
                        if mean_fint:
                            mean_mz_dots = (np.array(mean_fmz)>=mz_draw_borders[0]) & (np.array(mean_fmz)<=mz_draw_borders[1])
                            ax.plot(mean_fmz[mean_mz_dots], mean_fint[mean_mz_dots]*(-1), color = 'g')
                        
                        plt.gcf().tight_layout()
                        plt.ylabel("Intensity")
                        plt.legend([f"Mass spectrum N{rand_spec}"] + mean_leg,loc='upper left')
                    
                    
                    plt.figure(num=plt.get_fignums()[-1])
                    try: 
                        x,y = spec_data.getspectrum(rand_spec_2+idx_start)

                    except:
                        x = spec_data[slide][sample][roi]["mz"][:]        
                        y = spec_data[slide][sample][roi]["int"][rand_spec_2,:]
                    mz_draw_borders = plt.xlim()
                    dots_bord_spec = (np.array(x)>=mz_draw_borders[0]) & (np.array(x)<=mz_draw_borders[1])
                    ax = plt.gca().twinx() 
                    ax.plot(np.array(x)[dots_bord_spec],np.array(y)[dots_bord_spec], alpha=0.85)
                    
                    # Mean spectrum
                    if mean_fint:
                        mean_mz_dots = (np.array(mean_fmz)>=mz_draw_borders[0]) & (np.array(mean_fmz)<=mz_draw_borders[1])
                        ax.plot(mean_fmz[mean_mz_dots], mean_fint[mean_mz_dots]*(-1), color = 'g')
                    plt.gcf().tight_layout()
                    plt.ylabel("Intensity")
                    plt.legend([f"Mass spectrum N{rand_spec_2}"] + mean_leg, loc='upper left')
                    if source == 'hdf5':
                        spec_data[slide].close()
                   

                ## Запись в hdf5 файл, если на входе был путь к hdf5 файлу
                try:
                    with File(hdf5_feat_path, mode="a") as hdf5file:
                        hdf5file.create_dataset(sample + "/" + roi + "/features",data = Grouped_ftable[slide][sample][roi]["features"])

                        hdf5file[sample][roi]["features"].attrs["Column headers"] = list(Grouped_ftable[slide][sample][roi]["features"].columns)
                        print(f"Processed features of sample {sample} roi {roi} is saved in hdf5 file")
                    unique_num = Pgrouping_KD_table.unique_num
                    num_bf_raredel = Pgrouping_KD_table.num_bf_raredel 
                    num_raredel = Pgrouping_KD_table.num_raredel 
                    num_res = Pgrouping_KD_table.num_res
                    with open(os.path.join(os.path.dirname(path[slide]),"Peaks_grouping_settings.txt"), "a") as file:
                        file.write(f"\n\n##Grouping results of {sample} roi: {roi}\n")
                        file.write(f"KDE bandwidth: {KD_bandwidth}\n")
                        file.write(f"Grouping results:\nNumber of unique peaks before grouping: {unique_num}\nNumber of unique peaks after grouping: {num_bf_raredel}\nNumber of excluded peaks by count filter({CountF}): {num_raredel} ({num_raredel*100/num_bf_raredel:.2f}%)\nResulted feature peaks is {num_res}")             
                except Exception as error:
                    print(f"Processed features of sample {sample} roi {roi} doesn't saved in hdf5 file by following error:")
                    raise(error)
    
    return Grouped_ftable
# @profile  # TODO: удалить перед релизом
def Pgrouping_KD_table(ftable,
                       cpu_num, 
                       KD_bandwidth, bwc, KD_kernel, KD_func, 
                       CountF, norm, draw_borders, 
                       dupl_drop, split_params, 
                       sample = None, roi=None, 
                       account_mzscale = True, 
                       draw = True, 
                       **params2mspeaks_KD):
    """    
    Описание
    ----
    Вспомогательная функция к основной `Pgrouping_KD`. В ней производятся основные рассчёты функции и работает только с табличными данными `pd.DataFrame`.  
    """
    ind_norm = len(ftable.index.names)==1 and not ftable.index.names[0] # Определяем тип индексации (Обычная без названия (TRUE) "as is" или информативная по принадлежности к чему-либо)
    FWHM2sigma = bwc*(FWHM_TO_SIGMA_FACTOR/5.5)
    ## Заглушка, если не определены sample и roi
    if not sample:
        sample = "unknwn"
    if not roi:
        roi = '00'
    ## Заглушка, если не определены sample и roi

    ### Нормализация
    if norm[0] is not None:
        batch_SpN=np.array_split(np.array(ftable.set_index('spectra_ind',append = not ind_norm).index.unique()),cpu_num*3)
        ftable=ftable.set_index('spectra_ind',append = not ind_norm)
        metadata = ftable.attrs
        ftable.attrs = {}
        SpecNorm_parargs = product([ftable.loc[batch_SpN[i]].copy() for i in range(len(batch_SpN))],[norm])
        with Pool(cpu_num) as p:
            ftable = p.starmap(SpecNorm,SpecNorm_parargs)

        ftable=pd.concat(ftable)
        ftable.reset_index(level='spectra_ind', inplace=True)
        ftable.attrs.update(metadata)
    ### Нормализация end
    discret_coeffs = ftable.attrs.get('discret_coeffs')
    mz_model = np.poly1d(discret_coeffs)

    # print(ftable['mz'])
    # Здесь заранее реализовать ширину полосы пропускания векторными операциями, которые же и пойдут в gap!!!
    if KD_bandwidth == "med_fwhm":
        ftable.loc[:,'KD_bandwidth'] = (ftable['FWHMR'] - ftable['FWHML'])*FWHM2sigma
        if account_mzscale:
            mz_discret = mz_model(ftable['mz'])

            low_band_bool = ftable['KD_bandwidth'] < mz_discret*2
            # unreal_discret_bool = mz_discret > ftable['KD_bandwidth']/FWHM2sigma # Для ситуации когда dist_loc нереалистично больше ширины на полувысоте пика TODO Оценить реалистичность такого сценария и добавить
            # total_bool = low_band_bool^unreal_discret_bool
            ftable.loc[low_band_bool,'KD_bandwidth'] = (mz_discret[low_band_bool]*bwc).astype(ftable['KD_bandwidth'].dtype)
    elif KD_bandwidth == "mz_discret":
        ftable.loc[:,'KD_bandwidth'] = mz_model(ftable['mz'])*bwc
    else:
        ftable.loc[:,'KD_bandwidth'] = KD_bandwidth*bwc
    # double-splitting array 
    KD_data = split_pdtable_by_peaks_gap(ftable, split_mz_min = split_params['split_mz_min'], split_peaks_min = split_params['split_peaks_min'])

    # TODO: Проблемы с median_dist:
    # 1) нужны из сырых данных получить настоящие dist между точками
    # 2) Если нет возможности получить из сырых данных... 
    #   а. Брать между разницей пиков. Вариант хороший и упор будет на FWHM тогда, а если его нет - тоже неплохо. Но плохо, если у нас в данных сугубо шумы,
    #      которые далеко друг от друга и в итоге fwhm шума будет сильно меньше dist и dist будет нашей bw, что создаст ложноположительный результат
    #  NO б. Использовать только ppm прибора. Из плюсов: ложноположительного результата не будет. Минусы: если вдруг bw полученная от fwhm будет сильно ниже расстояния между точками, то спектры точно будут overfitted
    # TODO !!!!!!!!!!!! Написать защиту от малого кол-ва точек в сегменте при определении сигма и дистанции
    #################################################
    
    #             # TODO: попробовать не с медианной разницей между пиками, а с:
    #             # 1) min
    #             # 2) первый квартиль
    #             #     !!!! нужно создать мат обоснование верхней границы полосы пропускания
    #             # 1) медианное расстояние между точками никак не может быть больше fwhm, так как иначе fwhm и не построился бы
    #             # # 2) Но максимальное расстояние между точками возможно при ситуации, если пик состоит из трёх точек и дистанция там fwhm/0.53
    #             # # 3) Максимальное кол-во пиков в сегменте: нужно как-то ограничить. Если оно уже попадает по CountF - не считать
    #             # # 4) Максимальное кол-во пиков в сегменте: попытатьтся взять ближайшее достоверное значение сигмы с корректировкой в ppm?

    #     par_args[reversed_n] = (ftable_segment, KD_bandwidth_loc)
    mz_sequence = np.sort(ftable['mz'].unique())
    X_plot = []
    Y_plot = []
    partial_worker = partial(peaks_probability_distribution, KD_func, KD_kernel, discret_coeffs)
    with Pool(cpu_num) as p:
        Last_segment_max_mz = 0
        for X_plot_segment, Y_plot_segment in tqdm(p.imap_unordered(partial_worker,KD_data), total = len(KD_data), unit = 'segment', desc = 'Peaks probability distribution calculation'):
            assert Last_segment_max_mz < X_plot_segment.min() #TODO удалить при выкладывании
            X_plot.append(X_plot_segment)
            Y_plot.append(Y_plot_segment)
            
            # Y_plot[plot_slice] += result
    X_plot = np.hstack(X_plot)
    Y_plot = np.hstack(Y_plot)
    idx_sort = np.argsort(X_plot)
    X_plot = X_plot[idx_sort] 
    Y_plot = Y_plot[idx_sort]

    Y_plot = normalize(Y_plot.reshape(1,-1), norm = 'l1').squeeze() # Normalize the probability distribution
    

    ##Определение пиков графика плотности вероятности и их оснований
    ##### ПОпробовать разбить спектр на доли, чтобы более комфортно что-то строить
    logger.log("Peaks search is started")
    Xp_data = mspeaks_KD(X_plot,Y_plot,**params2mspeaks_KD)
    Xp = Xp_data[0]
    Xl = Xp_data[1]
    Xr = Xp_data[2]
    logger.log("Peaks search is ended")
    ##Проверка попадания нескольких пиков с одного спектра в этот диапазон
    ##Batching по диапазону
    ### Определим индексы значений m/z для диапазонов
    mz_num = len(mz_sequence)
    if mz_num < cpu_num*3:
        batches_num = mz_num
    else:
        batches_num = cpu_num*3
    idxmz_batches = list(pairwise(np.linspace(0,mz_sequence.shape[0],batches_num,dtype=int))) 
    ### Организуем аргументы для параллельного назначения
    par_args=[None]*len(idxmz_batches)
    metadata = ftable.attrs
    ftable.attrs = {}
    logger.log("Organizing arguments for parallelization")
    for batch_n,idx_batch in enumerate(idxmz_batches):
        mzb_min = mz_sequence[idx_batch[0]]
        mzb_max = mz_sequence[idx_batch[1]-1]
        Xl_min = Xl[Xl<=mzb_min][-1]
        Xr_max = Xr[Xr>=mzb_max][0]
        batch_indexes = (Xp>=Xl_min) & (Xp<=Xr_max)
        par_args[batch_n] = (ftable.loc[(ftable['mz'] >= mzb_min) & (ftable['mz'] <= mzb_max)], Xp_data[:,batch_indexes])
    # print(len(par_args))
    logger.log("Arguments for parallelization is organized")
    
    gc.collect()
    logger.log("Peaks assigment is started")
    with Pool(cpu_num) as p:
        grftable = p.starmap(Peak_assignment,par_args)
    grftable=pd.concat(grftable)
    grftable.attrs = metadata
    # print(f'here grftable {grftable}')
    logger.log("Peaks assigment is ended")
    # print(grftable)
    # print(ftable)
    assert len(grftable)==len(ftable), f"batching is not working properly grouped table length is {len(grftable)} but ftable length is {len(ftable)}"
    del Xp_data
    del par_args
    gc.collect()
    
    logger.log("Table pivoting started")
    ## Пивотинг таблицы для подсчётов и определения дублей пиков в одном спектре (по spectra_ind) для дальнейшего их удаления. Также учитываем разделение при работе с объединёнными таблицами пиков нескольких имаджей
    if ind_norm: # Проверка на наличие стандартного индекса, который неинформативен
        logger.log('DataFrame indexes is standart noname')
        temp_pivo = pd.pivot_table(grftable,index=["spectra_ind","Peak"],values="mz",aggfunc=["count"])
    else:
        logger.log('DataFrame indexes is complex named indexing')
        temp_pivo = pd.pivot_table(grftable,index=[grftable.index,"spectra_ind","Peak"],values="mz",aggfunc=["count"])
    #logger.log(f'temp_pivo after pivo: {temp_pivo}')
    ##Проверка на различие в индексации от стандартного
    idx_level_diff = list(set(grftable.index.names)-set(['sample','roi','slide']))
    if len(idx_level_diff)>0 and not (len(idx_level_diff)==1 and not grftable.index.names[0]):
        textw = f'The indexes of the data frame is not the default names:{idx_level_diff}\nThe removal of duplicate peaks in spectra can have unpredictable results.'
        logger.warn(textw)
    #results.set_index(pd.Index(range(results.shape[0]),name='idx'),append=True,inplace=True)## Обязательно в этом месте добавляем индекс к таблице, чтобы temp_pivo смог подсчитать одинаковые пики из-за одинаковой индексации
    logger.log("Table pivoting ended")
    Total_num_spec=len(temp_pivo.droplevel('Peak').index.unique()) #Количество спектров имаджа
    temp_pivo = temp_pivo.iloc[(temp_pivo["count"]>1)["mz"].values]
    logger.log(f"Pivoted table is empty:{temp_pivo.empty}")
    if not temp_pivo.empty:
        duplicated_num=len(temp_pivo.index.unique(level="Peak")) #Определяем кол-во дублированных пиков чисто для справки
        logger.log("Drawing graphs")
        
        num_of_uniq_spectras = len(temp_pivo.droplevel('Peak').index.unique()) #Определяем кол-во спектров, где обнаружены дубликаты чисто для справки
        textw=f"At the specified peak grouping settings in the peak list of {sample} {roi}, {temp_pivo['count']['mz'].sum()-temp_pivo.shape[0]} duplicates were identified, of which {duplicated_num} were unique peaks in {num_of_uniq_spectras} of mass spectra ({num_of_uniq_spectras*100/(Total_num_spec):.2f}% of the total spectra)."
        logger.warn(textw)
        if temp_pivo['count']["mz"].value_counts().index.max() > 2 and draw:
            plt.figure(figsize=(3, 2))
            plt.bar(temp_pivo['count']["mz"].value_counts().index.astype(str),temp_pivo['count']["mz"].value_counts().astype(int))
            plt.xlabel('Quantity')
            plt.ylabel('Num of duplicates')
            plt.gca().set_title(f"Peaks occurence in one group in mass spectrum. Sample: {sample} {roi}")        
            plt.grid(visible=True,which="both",axis="y")
        
        

        if draw:
            
            rand_num = np.random.randint(0,temp_pivo.shape[0])
            rand_spec = temp_pivo.iloc[rand_num].name
            peak_mz = rand_spec[-1]
            rand_spec = rand_spec[:-1]
            mz_draw_borders = (peak_mz-draw_borders,peak_mz+draw_borders)
            dots_bord = (np.array(X_plot)>=mz_draw_borders[0]) & (np.array(X_plot)<=mz_draw_borders[1])
            plt.figure(figsize=(25, 6), dpi=600)
            # plt.figure(figsize=(17.5, 7.5/1.5), dpi=600)
            plt.plot(np.array(X_plot)[dots_bord],np.array(Y_plot)[dots_bord],color='k',alpha = 0.85)
            if len(ftable.index.names)==1:
                query_table = ftable.query("spectra_ind==@rand_spec[-1] and mz>=@mz_draw_borders[0] and mz<=@mz_draw_borders[1]")["mz"]
                query_table_excluded = ftable.query("spectra_ind!=@rand_spec[-1] and mz>=@mz_draw_borders[0] and mz<=@mz_draw_borders[1]")["mz"]
            else:
                query_table = ftable.loc[rand_spec[:-1]].query("spectra_ind==@rand_spec[-1] and mz>=@mz_draw_borders[0] and mz<=@mz_draw_borders[1]")["mz"]
                query_table_excluded = ftable.loc[~((ftable["spectra_ind"]==rand_spec[-1]) & (ftable.index==rand_spec[:-1]))].query("mz>=@mz_draw_borders[0] and mz<=@mz_draw_borders[1]")["mz"]
            plt.plot(query_table_excluded,[0]*query_table_excluded.shape[0],'|', color= 'g',alpha=0.25)
            plt.plot(query_table,[0]*query_table.shape[0],'|', color= 'r',alpha=1)
            del query_table, query_table_excluded
            gc.collect()
            plt.xlabel('m/z')
            plt.ylabel("Probability Density")
            plt.xlim(mz_draw_borders[0],mz_draw_borders[1])
            if ind_norm:
                text = ["Probability function graphic",f"All peaks except mass spectrum N {rand_spec[-1]}",f"Peaks of mass spectrum N {rand_spec[-1]}"]
            else:
                text = ["Probability function graphic",f"All peaks except mass spectrum index {rand_spec[:-1]} and N {rand_spec[-1]}",f"Peaks of mass spectrum index {rand_spec[:-1]} and N {rand_spec[-1]}"]
            plt.legend(text)
            Pgrouping_KD_table.rand_spec_1 = rand_spec
            plt.minorticks_on()
            plt.grid(visible=True,which="both")
            plt.gca().set_title(f"Duplicated peak {peak_mz:.3f} in the spectra. Sample: {sample} {roi}.")
        else:
            Pgrouping_KD_table.rand_spec_1 = None
    else:
        Pgrouping_KD_table.rand_spec_1 = None
    logger.log("Drawing graphs ended")
    logger.log(f'Merging with peaks column: shape before {grftable.shape}')
    merged_full = grftable.merge(grftable['Peak'].value_counts(),left_on="Peak",right_index=True)
    logger.log(f'and shape after {grftable.shape} {merged_full.shape})')
    result = merged_full.query("count>=@CountF").copy()
    logger.log(f'Shape after filtering by CountF {result.shape} (excluded num of peaks: {merged_full.shape[0]-result.shape[0]})')
    excluded = merged_full.query("count<@CountF").copy()
    del merged_full
    num_res = len(result["Peak"].unique())
    num_bf_raredel = len(grftable["Peak"].unique())
    num_raredel = len(excluded["Peak"].unique())
    unique_num =len(ftable["mz"].unique())
    textw=f"Grouping results of {sample} {roi}:\nUnique peaks before grouping: {unique_num}\nUnique peaks after grouping: {num_bf_raredel}\nExcluded peaks by count filter({CountF}): {num_raredel} ({num_raredel*100/num_bf_raredel:.2f}%)\nResulted feature peaks is {num_res}"
    logger.log(textw)
    print(textw)

    gc.collect()

    ## Проверочный график. Строим график плотности вероятности в некотором диапазоне случайного пика

    if draw:
        rand_num = np.random.randint(0,result.shape[0])
        rand_spec, peak_mz = result.iloc[rand_num][["spectra_ind","Peak"]]
        
        # TODO: !!!!!!! Дописать, чтобы были все положительные шкалы игрик
        plt.figure(figsize=(25, 6), dpi=600)
        # plt.figure(figsize=(17.5, 7.5/1.5), dpi=600)
        mz_draw_borders = (peak_mz-draw_borders,peak_mz+draw_borders)
        dots_bord = (np.array(X_plot)>=mz_draw_borders[0]) & (np.array(X_plot)<=mz_draw_borders[1])
        plt.plot(np.array(X_plot)[dots_bord],np.array(Y_plot)[dots_bord], color="k",alpha=0.85)
        leg = ["Probability density function"]
        plt.xlim(mz_draw_borders)
        quered_results = result.query("mz>=@mz_draw_borders[0] and mz<=@mz_draw_borders[1]")
        Peaks_list=quered_results["Peak"].sort_values().unique()
        plt.xlabel('m/z')
        plt.ylabel("Probability Density")
        plt.gca().set_title(f"Grouping results around peak {peak_mz:.3f}. Sample: {sample} {roi}.")

        for peak in Peaks_list:
            temp_query = quered_results.query("Peak == @peak")
            plt.plot(temp_query["mz"],[0]*temp_query.shape[0],'|',alpha=0.85)
            leg+=[f"{peak:.3f} peak"]
        plt.plot(excluded["mz"],[0]*excluded.shape[0],"|",c="k",alpha=1)
        leg+=["Excluded peaks"]
        plt.legend(leg, loc = 'upper right')
        plt.minorticks_on()
        plt.grid(visible=True,which="both")
    
        Pgrouping_KD_table.rand_spec_2=int(rand_spec)
    else:
        Pgrouping_KD_table.rand_spec_2=None
    Pgrouping_KD_table.unique_num = unique_num
    Pgrouping_KD_table.num_bf_raredel = num_bf_raredel
    Pgrouping_KD_table.num_raredel = num_raredel
    Pgrouping_KD_table.num_res = num_res

    if params2mspeaks_KD.get('return_pkY', False):
        drop_list = ['count']
    else:
        drop_list = ['count','mz']
    result.drop(columns=drop_list,inplace=True)

    headers=[]
    Column_headers = result.columns

    for header in zip(['SNR',"Intensity","Area","PextL","PextR","FWHML","FWHMR"],["max","max","sum","min","max","min","max"]):
        if header[0] in Column_headers:
            headers.append(header)
    dict4drop = dict(headers)
    oth_col = list(set(Column_headers) -set(list(dict4drop.keys())+['spectra_ind','Peak']))

    if oth_col:
        dict4drop.update(dict(zip(oth_col,len(oth_col)*['first'])))

    if dupl_drop:
        if ind_norm:
            result=pd.pivot_table(result,index=['spectra_ind','Peak'],aggfunc=dict4drop).reset_index(level=['spectra_ind','Peak'])
        else:
            result=pd.pivot_table(result,index=[*result.index.names,'spectra_ind','Peak'],aggfunc=dict4drop).reset_index(level=['spectra_ind','Peak'])

    logger.log(f"Shape of the dataframe on enter:{ftable.shape}")
    logger.log(f"Shape of the dataframe before drop:{grftable.shape} ({grftable.shape[0]-result.shape[0]})")
    logger.log(f"Shape of the dataframe after drop:{result.shape} ({grftable.shape[0]-result.shape[0]})")
    return result#.reindex(columns=ftable_colarrange)

def peaks_probability_distribution(KDE_func, KD_kernel, mz_discret_coeffs, KD_data): # Пока медленно
    mz, KD_bandwidth = KD_data
    ftable_min = mz[0] - KD_bandwidth[0]*6
    ftable_max = mz[-1] + KD_bandwidth[-1]*6
    min_dist = np.poly1d(mz_discret_coeffs)(mz).min()
    X_plot_segment = _set_KDE_X_plot(ftable_min, ftable_max, min_dist = min_dist)
    if KDE_func == 'FFT':# or (len(mz_discret_coeffs) == 1):
        KD_bandwidth = np.median(KD_bandwidth)
        KDE_func = FFTKDE
    else:
        KDE_func = TreeKDE
    Y_plot_segment = KDE_func(kernel = KD_kernel, bw = KD_bandwidth).fit(mz)(X_plot_segment)*len(mz)
    return X_plot_segment, Y_plot_segment

def batching(features4batch,mz):
    """
    Вспомогательная функция разбивки на батчи по зонам mz
    """
    return features4batch.loc[(features4batch['mz']<=mz[1]) and (features4batch['mz']>=mz[0])]

def SpecNorm(ftable,norm):
    """    
    Описание
    ----
    Вспомогательная функция к основной `Pgrouping_KD`. Проводит нормализацию пиклиста относительно одного спектра.  
    """
    for idx in ftable.index.unique():
        if isinstance(norm[1],str):
            col_list=[norm[1]]
        else:
            col_list=norm[1]

        data_array = ftable.loc[idx,col_list].to_numpy()
        if len(data_array.shape)==1:
            data_array.shape = (data_array.shape[0],1)
        ftable.loc[idx,col_list] = normalize(data_array,norm=norm[0],axis=0)
    return ftable

def Peak_assignment(ftable_batch,Xp_batch):
    """    
    Описание
    ----
    Вспомогательная функция к основной `Pgrouping_KD`. Определяет принадлежность значений mz к определённому значению пика  
    """
    if not ftable_batch.empty:
        if len(Xp_batch) == 3:
            
            for peak, xl, xr in Xp_batch.T:
                bool_mask = (ftable_batch['mz']>=xl) & (ftable_batch['mz']<=xr)
                ftable_batch.loc[bool_mask,"Peak"] = peak
        else:

            for peak, xl, xr, density in Xp_batch.T:
                bool_mask = (ftable_batch['mz']>=xl) & (ftable_batch['mz']<=xr)
                ftable_batch.loc[bool_mask,"Peak"] = peak
                ftable_batch.loc[bool_mask,"Density"] = density
    return ftable_batch

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

def _set_KDE_X_plot(plot_start, plot_end, min_dist):
    mz_range = plot_end - plot_start
    num_of_dots = int((mz_range)*5/min_dist)+1
    # print(num_of_dots)
    X_plot = np.linspace(np.float64(plot_start),np.float64(plot_end),num_of_dots)
    diffs = np.diff(X_plot)

    while not np.allclose(np.ones_like(diffs) * diffs[0], diffs):
        logger.warn(f"X_plot is not uniform between {plot_start} and {plot_end} with num of dots: {num_of_dots} and distances between points {np.unique_values(diffs)}. Reducing number of dots for X_plot by 2 times")
        num_of_dots=int(num_of_dots/2)
        X_plot = np.linspace(plot_start,plot_end,num_of_dots)
        diffs = np.diff(X_plot)
        if num_of_dots<=1:
            raise AssertionError("Cannot get uniform data for KDE. See logs for info")
    logger.log(f"X_plot is uniform between {plot_start} and {plot_end} with dtype {type(plot_start)} and num of dots: {num_of_dots} and distances between points {np.unique(diffs, equal_nan=False)}.")
    return X_plot