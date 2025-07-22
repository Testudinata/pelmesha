
import pandas as pd
import numpy as np
from itertools import product,pairwise
from torch.multiprocessing import Pool, cpu_count
from pelmesha.loaders import specdata_Load, IMGfeats_concat, feat2DF, peakl2DF, logger
import matplotlib.pyplot as plt
from h5py import File
from KDEpy import FFTKDE
from pyimzml.ImzMLParser import ImzMLParser
from sklearn.preprocessing import normalize
import gc
import math
import os
import warnings

### Base functions
def Pgrouping_KD(ftable, columns = None,KD_bandwidth = "med_fwhm", bwc = 1,KD_kernel = "gaussian",CountF = 10,tol = 500, norm = (None,None),draw_borders = 1.5,
                  dupl_drop = True,min_res = 10, pivoting4val = None, cpu_free=1, path2save=None,sample="unknwn",roi="00", coords4table=None, draw=True, **params2mspeaks_KD):
    """    
    Описание
    ----
    Функция группирует близкостоящие значения пиков mz к определённому значению Peak на основе оценки плотности вероятности встречаемости mz.  

    :param ftable: Источник данных. Может быть уже таблицей/датафреймом, а может быть ссылкой на папку с данными, которые надо выгрузить
    :param columns: Лист номеров столбцов, которые войдут в датафрейм фич, где 0 и 1 - экстрагируются всегда (`"spectra_ind"` и `"mz"` или `"Peak"`). Default: `None` - экстракция всех столбцов
    2 - `"Intensity"`, 3 -`"Area"`, 4 - `"SNR"`, 5 - `"PextL"`, 6 - `"PextR"`, 7 - `"FWHML"`, 8 - `"FWHMR"`, 9-`"Noise"`, 10-`"Mean noise"`
    :param KD_bandwidth: {`"med_FWHM"`,`"mz_discret"`,`"ISJ"`,`"silverman"`,`"scott"`, `float`} - выбор полосы пропускания, либо алгоритм её определения.
    
        `"med_FWHM"` - полоса пропускания - медианное значение дисперсии пиков, которая определяется из медианы ширины на полувысоте. 
    
        `"mz_discret"` - полоса пропускания - медиана расстояний между точками шкалы mz, где лучше всего, если есть доступ к файлу hdf5 и imzml. 

        `"ISJ"`,`"silverman"`,`"scott"` - базовые для функции FFTKDE (из пакета KDEpy)
    :param bwc: коэффициент полученной/выбранной полосы пропускания
    :param KD_kernel: KDE ядро (see also: https://kdepy.readthedocs.io/en/latest/kernels.html#available-kernels)
    :param CountF: Параметр исключение из датасета редких пиков, чьё кол-во меньше данного значения
    :param params2mspeaks_KD: остальные параметры для функции mspeaks_KD
    :param tol: tolerance in ppm of m/z (used for batching dots by window)
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
    :type tol: `float`
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
    
    logger("Pgrouping_KD",{**locals()},path2save)
    if not columns:
        columns=list(range(11))
    if not isinstance(columns,list):
        columns = [columns]
    cpu_num = cpu_count()-cpu_free
    tol = tol/1e+6
    min_res = min_res/1e+6
    if KD_bandwidth.lower() =='med_fwhm':
        columns=columns+[7,8]
        
    # Вариация для рассчётов для уже готовой таблицы или путей к таблицам features 

    columns=list(set([0,1]+columns))

    if isinstance(ftable, str):
        ftableISAPATH=True
        data, path = peakl2DF([ftable],extr_columns=columns, return_source_path = True,pivoting4val=None)
    elif isinstance(ftable, list):
        data, path = peakl2DF(ftable,extr_columns=columns, return_source_path = True,pivoting4val=None)
        ftableISAPATH=True
    if isinstance(ftable, pd.DataFrame):
        ftableISAPATH=False
        median_dist = ftable['mz'].sort_values().diff().loc[ftable['mz'].sort_values().diff()>0].median()
        plot_start = ftable['mz'].min()-1
        plot_end = ftable['mz'].max()+1
        logger.log(f"Peaks are in range {plot_start}-{plot_end}. Median distance: {median_dist}")

    if ftableISAPATH:
        logger.log(f"Data proccessed from file")
        ftable = Pgrouping_KD_file(data,path,cpu_num, KD_bandwidth, bwc,KD_kernel,CountF,tol, norm,draw_borders,dupl_drop, min_res,draw,**params2mspeaks_KD)
        if pivoting4val:
            for slide in ftable.keys():
                for sample in ftable[slide].keys():
                    for roi in ftable[slide][sample].keys():
                        ftable[slide][sample][roi]['features'] = ftable[slide][sample][roi]['features'].pivot_table(index="spectra_ind", columns="Peak",fill_value = 0, values =pivoting4val)
    else:
        logger.log(f"Data proccessed from table")
        ftable = Pgrouping_KD_table(ftable,cpu_num,median_dist,plot_start,plot_end,KD_bandwidth, bwc,KD_kernel,CountF,tol, norm,draw_borders,dupl_drop, min_res,sample=sample,roi=roi,draw=draw,**params2mspeaks_KD)
        if path2save:
            logger.log(f"Starting saving grouped data")
            try:
                if os.path.exists(os.path.join(path2save,f"{sample}_{roi}")+"_features.hdf5"):
                    os.remove(os.path.join(path2save,f"{sample}_{roi}")+"_features.hdf5")
                    logger.log(f"Old file deleted")
                hdf5file = File(os.path.join(path2save,f"{sample}_{roi}")+"_features.hdf5", mode="a")
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
    :rtype: `list` or tuple(`list`,`pd.Series`) 
    """
    logger("Getrefpeaks",{**locals()})
    ref = IMGfeats_concat(ref_rois_paths,extr_columns=None,extracts_coords=False,processed_feat = False)
    ref = Pgrouping_KD(ref.reset_index(drop=True),**Pgrouping_KD_kwargs)
    ref_peaks = ref.pivot_table(columns= 'Peak', fill_value = 0,aggfunc={"spectra_ind":'count'}).rename({'spectra_ind':'P_occurence'}).astype(float)
    ref_peaks.loc['P_occurence'] = ref_peaks.loc['P_occurence']/ref_peaks.loc['P_occurence'].max()
    ref_peaks = ref_peaks.loc[:,ref_peaks.loc['P_occurence']>min_occurence]
    ref_peaklist = list(ref_peaks.columns)
    
    align_list = []
    for lim in pairwise(np.arange(min(ref_peaklist),max(ref_peaklist)+step+1,step)):
        align_list+=list(ref_peaks.loc[:,(ref_peaks.columns>lim[0]) & (ref_peaks.columns<lim[1])].sort_values(by="P_occurence",axis=1).T.tail(num_peaks_per_step).index)
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
    :param extr_columns: Лист номеров столбцов для экстракции из `hdf5`, где 0 и 1 - экстрагируются всегда (`"spectra_ind"` и `"mz"` или `"Peak"`). Default: `None` - экстракция всех столбцов
    2 - `"Intensity"`, 3 -`"Area"`, 4 - `"SNR"`, 5 - `"PextL"`, 6 - `"PextR"`, 7 - `"FWHML"`, 8 - `"FWHMR"`, 9-`"Noise"`, 10-`"Mean noise"`
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
    if path2save:
        try:
            os.remove(path2save+"_grouped_MSIdata.hdf5")
            print(f"Previous grouped features data is deleted")
        except:
            pass
    try:
        if Pgrouping_KD_kwargs["KD_bandwidth"] == "med_FWHM" and extr_columns is not None:
            extr_columns=list(set(extr_columns + [7,8]))
            extr_columns.sort()
    except:
        pass
    logger.log('Additional extraction columns list of values: 2 - "Intensity", 3 -"Area", 4 - "SNR", 5 - "PextL", 6 - "PextR", 7 - "FWHML", 8 - "FWHMR", 9-"Noise", 10-"Mean noise"')
    logger.log(f'Additional extraction columns:{extr_columns}')
    Total_peaks, coords = IMGfeats_concat(Paths,extr_columns,extracts_coords=True,processed_feat = False)

    if "pivoting4val" in Pgrouping_KD_kwargs.keys():
        if isinstance(Pgrouping_KD_kwargs["pivoting4val"], str):
            Pgrouping_KD_kwargs["pivoting4val"]=Pgrouping_KD_kwargs["pivoting4val"]
        pivoting4val = Pgrouping_KD_kwargs["pivoting4val"]
        del Pgrouping_KD_kwargs["pivoting4val"]
        logger.warn(f"'pivoting4val'= {pivoting4val}. Data saved in hdf5 is not pivoted")
        warnings.warn(f"'pivoting4val'= {pivoting4val}. Data saved in hdf5 is not pivoted")
    else:
        pivoting4val = None

    Aligned_rois = Pgrouping_KD(Total_peaks,**Pgrouping_KD_kwargs)
    ########## under construction

    try:
        if path2save:
            
            indexes = Aligned_rois.index.unique()
            image_num = len(list(indexes))
            slides=[]
            for index in indexes:
                slides.append(index[0])
            slides = list(set(slides)) 
            hdf5file = File(os.path.join(path2save,f"Images_{image_num}_grouped_MSIdata.hdf5"), mode="a")
            logger.log("Grouped features is saved in hdf5 file of images:")
            for index in indexes:
                hdf5file.create_dataset(index[0] + "/" + index[1] + "/"+ index[2] + "/features",data = Aligned_rois.loc[index])
                hdf5file.create_dataset(index[0] + "/" + index[1] + "/"+ index[2] + "/xy",data = coords.loc[index])
                logger.log(f"Data saved: {list(index)}")
            hdf5file.attrs["Column headers"] = list(Aligned_rois.columns)
            print(f"Grouped features is saved in hdf5 file")
            hdf5file.close()
    except:
        logger.warn(f"Error on {index}")

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
def Pgrouping_KD_file(data,path,cpu_num, KD_bandwidth, bwc,KD_kernel, CountF,tol, norm, draw_borders, dupl_drop, min_res, draw,**params2mspeaks_KD):
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
            file.write(f"Tolerance for parallel peak assigment: {tol*1e+6} ppm")
            file.write(f"\nNormalizing data after grouping: type of normalization {norm[0]}, column: {norm[1]}")
        try:
            os.remove(path[slide][:-14]+"_features.hdf5")
            print(f"Previous processed features data is deleted")
        except:
            pass
        

        for sample in data[slide].keys():
            Grouped_ftable[slide][sample] = {}
            for roi in data[slide][sample].keys():
                Grouped_ftable[slide][sample][roi] = {}

                ftable=data[slide][sample][roi]["peaklists"]
                ## normalization
                
                ##Определение шкалы построение функции плотности вероятности.
                Grouped_ftable[slide][sample][roi]["xy"] = data[slide][sample][roi]["xy"]
                try:
                    Grouped_ftable[slide][sample][roi]["z"] = data[slide][sample][roi]["z"]
                except:
                    pass
        
                spec_data = specdata_Load(path[slide])
                try:
                    source='imzml'

                    spectrums_source=spec_data[slide][sample][roi].attrs['source']
                    idx_start, num_spec = spec_data[slide][sample][roi].attrs['idxroi']
                    dcont = spec_data[slide][sample][roi].attrs['continuous']
                    spec_data = ImzMLParser(spectrums_source)
                    
                    if dcont:
                        idx = np.random.randint(idx_start,idx_start+num_spec)
                        mz = spec_data.getspectrum(idx)[0]
                        median_dist = np.median(np.diff(mz))
                        plot_start = mz[0]-1
                        plot_end = mz[-1]+1
                        #min_dist = min(np.diff(mz))
                    else:
                        mz= spec_data.getspectrum(idx_start)[0]
                        #min_dist=min(np.diff(mz))
                        median_dist = [None]*num_spec
                        median_dist[0] = np.median(np.diff(mz))
                        plot_start = mz[0]-1
                        plot_end = mz[-1]+1
                        for n,idx in enumerate(range(idx_start+1,idx_start+num_spec)):
                            mz= spec_data.getspectrum(idx)[0]
                            #min_dist=min(min_dist,min(np.diff(mz)))
                            median_dist[n+1]=np.median(np.diff(mz))
                            plot_start = min(mz[0]-1,plot_start)
                            plot_end = max(mz[-1]+1,plot_end)
                        median_dist=np.mean(median_dist)
                    pass
                    
                except:
                    source='hdf5'
                    min_dist = np.diff(np.sort(spec_data[slide][sample][roi]["mz"][:]))
                    
                    #min_dist = min_dist[min_dist>0]
                    median_dist = np.median(min_dist[min_dist>0])
                    
                    plot_start = spec_data[slide][sample][roi]["mz"][:].min()-1
                    plot_end = spec_data[slide][sample][roi]["mz"][:].max()+1
                
                Grouped_ftable[slide][sample][roi]["features"] = Pgrouping_KD_table(ftable,cpu_num,median_dist,plot_start,plot_end, KD_bandwidth, bwc,KD_kernel, CountF,tol, norm, draw_borders, dupl_drop,min_res, sample, roi,draw, **params2mspeaks_KD)  
                plt.figure(num=plt.get_fignums()[-2])
                rand_spec = Pgrouping_KD_table.rand_spec_1[-1]
                try:
                    x = spec_data[slide][sample][roi]["mz"][:]        
                    y = spec_data[slide][sample][roi]["int"][rand_spec,:]

                except:
                    x,y = spec_data.getspectrum(rand_spec+idx_start)

                mz_draw_borders = plt.xlim()
                dots_bord_spec = (np.array(x)>=mz_draw_borders[0]) & (np.array(x)<=mz_draw_borders[1])
                ax = plt.gca().twinx()
                ax.plot(np.array(x)[dots_bord_spec],np.array(y)[dots_bord_spec],c="dimgray",alpha=0.75)
                plt.gcf().tight_layout()
                plt.ylabel("Intensity")
                plt.legend([f"Graph of the {rand_spec} mass spectrum"],loc='upper left')
                rand_spec = Pgrouping_KD_table.rand_spec_2
                
                plt.figure(num=plt.get_fignums()[-1])
                try: 
                    x,y = spec_data.getspectrum(rand_spec+idx_start)

                except:
                    x = spec_data[slide][sample][roi]["mz"][:]        
                    y = spec_data[slide][sample][roi]["int"][rand_spec,:]
                mz_draw_borders = plt.xlim()
                dots_bord_spec = (np.array(x)>=mz_draw_borders[0]) & (np.array(x)<=mz_draw_borders[1])

                ax = plt.gca().twinx() 
                ax.plot(np.array(x)[dots_bord_spec],np.array(y)[dots_bord_spec], c="dimgray",alpha=0.75)
                plt.gcf().tight_layout()
                plt.ylabel("Intensity")
                plt.legend([f"Graph of the {rand_spec} mass spectrum"], loc='upper left')
                if source == 'hdf5':
                    spec_data[slide].close()
                ## Удаление старого hdf5 файла

                ## Запись в hdf5 файл, если на входе был путь к hdf5 файлу
                try:
                    
                    hdf5file = File(path[slide][:-14]+"_features.hdf5", mode="a")
                    hdf5file.create_dataset(sample + "/" + roi + "/features",data = Grouped_ftable[slide][sample][roi]["features"])
                    hdf5file.create_dataset(sample + "/" + roi + "/xy",data = Grouped_ftable[slide][sample][roi]["xy"])
                    try:
                        hdf5file.create_dataset(sample + "/" + roi + "/z",data = Grouped_ftable[slide][sample][roi]["z"])
                    except:
                        pass                        
                    hdf5file[sample][roi]["features"].attrs["Column headers"] = list(Grouped_ftable[slide][sample][roi]["features"].columns)
                    print(f"Processed features of sample {sample} roi {roi} is saved in hdf5 file")
                    hdf5file.close()
                    unique_num = Pgrouping_KD_table.unique_num
                    num_bf_raredel = Pgrouping_KD_table.num_bf_raredel 
                    num_raredel = Pgrouping_KD_table.num_raredel 
                    num_res = Pgrouping_KD_table.num_res
                    with open(os.path.join(os.path.dirname(path[slide]),"Peaks_grouping_settings.txt"), "a") as file:
                        file.write(f"\n\n##Grouping results of {sample} roi: {roi}\n")
                        file.write(f"KDE bandwidth: {KD_bandwidth}\n")
                        file.write(f"Grouping results:\nNumber of unique peaks before grouping: {unique_num}\nNumber of unique peaks after grouping: {num_bf_raredel}\nNumber of excluded peaks by count filter({CountF}): {num_raredel} ({num_raredel*100/num_bf_raredel:.2f}%)\nResulted feature peaks is {num_res}")             
                except:
                    print(f"Processed features of sample {sample} roi {roi} doesn't saved in hdf5 file")
    
    return Grouped_ftable

def Pgrouping_KD_table(ftable,cpu_num,median_dist,plot_start,plot_end, KD_bandwidth, bwc,KD_kernel, CountF,tol, norm, draw_borders, dupl_drop,min_res, sample = None, roi=None, draw=True,**params2mspeaks_KD):
    """    
    Описание
    ----
    Вспомогательная функция к основной `Pgrouping_KD`. В ней производятся основные рассчёты функции и работает только с табличными данными `pd.DataFrame`.  
    """
    ind_norm = len(ftable.index.names)==1 and not ftable.index.names[0] # Определяем тип индексации (Обычная без названия (TRUE) "as is" или информативная по принадлежности к чему-либо)

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
        SpecNorm_parargs = product([ftable.loc[batch_SpN[i]].copy() for i in range(len(batch_SpN))],[norm])
        with Pool(cpu_num) as p:
            ftable = p.starmap(SpecNorm,SpecNorm_parargs)

        ftable=pd.concat(ftable)
        ftable.reset_index(level='spectra_ind', inplace=True)
    ### Нормализация
    ftable_colarrange = ftable.rename(columns={'mz':'Peak'}, inplace=False).columns
    
    if median_dist<min_res*(plot_start+plot_end)/2:
        median_dist=min_res*(plot_start+plot_end)/2
        #logging.warning(f'Warning. The value of {min_res*1e+6} ppm is used as the minimum distance between points to build the density distribution. If you want to build a more accurate probability distribution, change the "min_res" parameter. (Example: accuracy of Orbitrap ~ 10 ppm)')
        textw=f'Warning. The value of {min_res*1e+6} ppm is used as the minimum distance between points to build the density distribution. If you want to build a more accurate probability distribution, change the "min_res" parameter. (Example: accuracy of Orbitrap ~ 10 ppm)'
        logger.warn(textw)
        warnings.warn(textw)
    #X_plot = np.arange(plot_start,plot_end+1,median_dist/10)
    X_plot = np.linspace(plot_start,plot_end,int((plot_end-plot_start)*10/median_dist)+1)
    #assert np.allclose(np.ones_like(diffs) * diffs[0], diffs)
    logger.log(f"KD bandwith value or estimation method = {KD_bandwidth}")
    if KD_bandwidth == "mz_discret":
        #KD_bandwidth = min_dist*bwc
        KD_bandwidth = median_dist*bwc

    elif KD_bandwidth.lower() == "med_fwhm":
        KD_bandwidth =np.median(np.array(ftable["FWHMR"])-np.array(ftable["FWHML"]))*bwc*0.1431
    logger.log(f"KD bandwidth value\\estimated value = {KD_bandwidth}, with coeff: {bwc}")
    ##
    ##Построение графика функции плотности вероятности по всей шкале mz
    
    Y_plot = FFTKDE(kernel=KD_kernel,bw=KD_bandwidth).fit(ftable['mz'].values)(X_plot)
    
    ##
    ##Определение пиков графика плотности вероятности и их оснований
    ##### ПОпробовать разбить спектр на доли, чтобы более комфортно что-то строить
    logger.log("Peaks search is started")
    Xp, Xl, Xr = mspeaks_KD(X_plot,Y_plot,**params2mspeaks_KD)
    logger.log("Peaks search is ended")
    ##Проверка попадания нескольких пиков с одного спектра в этот диапазон
    ##Batching по диапазону
    ### Определим индексы значений m/z для диапазонов
  
    sorted_mz = ftable['mz'].sort_values(ascending=True).unique()

    idxmz_batches = list(pairwise(np.linspace(0,sorted_mz.shape[0],cpu_num*3,dtype=int))) 
    ### Определяем сами m/z диапазону по сортированному списку m/z
    par_args=[None]*len(idxmz_batches)
    #sorted_mz = ftable["mz"].sort_values(ascending=True).reset_index(drop=True)
    ### Организуем аргументы для параллельного назначения
    logger.log("Organizing arguments for parallelization")
    for batch_n,idx_batch in enumerate(idxmz_batches[:-1]):
        mzb_min = sorted_mz[idx_batch[0]]
        mzb_max = sorted_mz[idx_batch[1]-1]
        batch_indexes = (Xp>=mzb_min-mzb_min*tol) & (Xp<=mzb_max+mzb_max*tol)
        par_args[batch_n] = (ftable.loc[(ftable['mz']>=mzb_min) & (ftable['mz']<=mzb_max)],np.array(Xp)[batch_indexes],np.array(Xl)[batch_indexes],np.array(Xr)[batch_indexes])
       
    #print(f'Time 1 is: {time.time()-start_time}')
    mzb_min = sorted_mz[idxmz_batches[-1][0]]
    mzb_max = sorted_mz[idxmz_batches[-1][1]-1]
    batch_indexes = (Xp>=mzb_min-mzb_min*tol) & (Xp<=mzb_max+mzb_max*tol)
    par_args[-1] = (ftable.loc[(ftable['mz']>=mzb_min) & (ftable['mz']<=mzb_max)],np.array(Xp)[batch_indexes],np.array(Xl)[batch_indexes],np.array(Xr)[batch_indexes])

    logger.log("Arguments for parallelization is organized")
    del Xp, Xl, Xr
    gc.collect()
    logger.log("Peaks assigment is started")
    with Pool(cpu_num) as p:
        grftable = p.starmap(Peak_assignment,par_args)
    grftable=pd.concat(grftable)
    logger.log("Peaks assigment is ended")

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
        warnings.warn(textw)
    #results.set_index(pd.Index(range(results.shape[0]),name='idx'),append=True,inplace=True)## Обязательно в этом месте добавляем индекс к таблице, чтобы temp_pivo смог подсчитать одинаковые пики из-за одинаковой индексации
    logger.log("Table pivoting ended")
    Total_num_spec=len(temp_pivo.droplevel('Peak').index.unique()) #Количество спектров имаджа
    temp_pivo = temp_pivo.iloc[(temp_pivo["count"]>1)["mz"].values]
    logger.log(f"Pivoted table is empty:{temp_pivo.empty}")
    if not temp_pivo.empty:
        duplicated_num=len(temp_pivo.index.unique(level="Peak")) #Определяем кол-во дублированных пиков чисто для справки
        logger.log("Drawing graphs")
        
        num_of_uniq_spectras = len(temp_pivo.droplevel('Peak').index.unique()) #Определяем кол-во спектров, где обнаружены дубликаты чисто для справки
        logger.log(f"{temp_pivo.droplevel('Peak').index.unique()}") 
        textw=f"\nWarning! At the specified peak grouping settings in the peak list of {sample} {roi}, {temp_pivo['count']['mz'].sum()-temp_pivo.shape[0]} duplicates were identified, of which {duplicated_num} were unique peaks in {num_of_uniq_spectras} of mass spectra ({num_of_uniq_spectras*100/(Total_num_spec):.2f}% of the total spectra)."
        logger.warn(textw)
        warnings.warn(textw)
        if temp_pivo['count']["mz"].value_counts().index.max() > 2 and draw:
            plt.figure(figsize=(3, 2))
            plt.bar(temp_pivo['count']["mz"].value_counts().index.astype(str),temp_pivo['count']["mz"].value_counts().astype(int))
            plt.xlabel('Quantity')
            plt.ylabel('Num of duplicates')
            plt.gca().set_title(f"Occurence of peaks in one group in mass spectrum. Sample: {sample} {roi}")        
            plt.grid(visible=True,which="both",axis="y")
        
        
        rand_num = np.random.randint(0,temp_pivo.shape[0])
        rand_spec = temp_pivo.iloc[rand_num].name
        peak_mz = rand_spec[-1]
        rand_spec = rand_spec[:-1]
        mz_draw_borders = (peak_mz-draw_borders,peak_mz+draw_borders)
        dots_bord = (np.array(X_plot)>=mz_draw_borders[0]) & (np.array(X_plot)<=mz_draw_borders[1])
        if draw:
            plt.figure(figsize=(25, 4), dpi=600)
            plt.plot(np.array(X_plot)[dots_bord],np.array(Y_plot)[dots_bord])
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
                text = ["Probability function graphic",f"All peaks except mass spectra with num of spectra {rand_spec[-1]}",f"The mass spectrum peaks of spectra with num of spectra {rand_spec[-1]}"]
            else:
                text = ["Probability function graphic",f"All peaks except mass spectra with index {rand_spec[:-1]} and num of spectra {rand_spec[-1]}",f"The mass spectrum peaks of spectra with index {rand_spec[:-1]} and num of spectra {rand_spec[-1]}"]
            plt.legend(text)
            Pgrouping_KD_table.rand_spec_1=rand_spec
            plt.minorticks_on()
            plt.grid(visible=True,which="both")
            plt.gca().set_title(f"Duplicated peak {peak_mz:.3f} in the spectra of sample {sample} {roi}.")
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
    textw=f"Grouping results of {sample} {roi}:\nNumber of unique peaks before grouping: {unique_num}\nNumber of unique peaks after grouping: {num_bf_raredel}\nNumber of excluded peaks by count filter({CountF}): {num_raredel} ({num_raredel*100/num_bf_raredel:.2f}%)\nResulted feature peaks is {num_res}"
    logger.log(textw)
    print(textw)

    gc.collect()

    ## Проверочный график. Строим график плотности вероятности в некотором диапазоне случайного пика
    rand_num = np.random.randint(0,result.shape[0])
    rand_spec, peak_mz = result.iloc[rand_num][["spectra_ind","Peak"]]

    mz_draw_borders = (peak_mz-draw_borders,peak_mz+draw_borders)
    if draw:
        plt.figure(figsize=(25,4), dpi=600)
        dots_bord = (np.array(X_plot)>=mz_draw_borders[0]) & (np.array(X_plot)<=mz_draw_borders[1])
        plt.plot(np.array(X_plot)[dots_bord],np.array(Y_plot)[dots_bord])
        leg = ["Probability function graphic"]
        plt.xlim(mz_draw_borders)
        quered_results = result.query("mz>=@mz_draw_borders[0] and mz<=@mz_draw_borders[1]")
        Peaks_list=quered_results["Peak"].sort_values().unique()
        plt.xlabel('m/z')
        plt.ylabel("Probability Density")
        plt.gca().set_title(f"Grouping results around peak {peak_mz:.3f} (as example) of sample {sample} {roi}.")

        for peak in Peaks_list:
            temp_query = quered_results.query("Peak == @peak")
            plt.plot(temp_query["mz"],[0]*temp_query.shape[0],'|',alpha=0.85)
            leg+=[f"m/z dots of {peak:.3f} peak"]
        plt.plot(excluded["mz"],[0]*excluded.shape[0],"|",c="k",alpha=1)
        leg+=["Dots of excluded peaks"]
        plt.legend(leg)
        plt.minorticks_on()
        plt.grid(visible=True,which="both")
    
    Pgrouping_KD_table.rand_spec_2=int(rand_spec)
    Pgrouping_KD_table.unique_num = unique_num
    Pgrouping_KD_table.num_bf_raredel = num_bf_raredel
    Pgrouping_KD_table.num_raredel = num_raredel
    Pgrouping_KD_table.num_res = num_res

    result.drop(columns=['count','mz'],inplace=True)

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
            result=pd.pivot_table(result,index=['spectra_ind','Peak'],aggfunc=dict4drop)
        else:
            result=pd.pivot_table(result,index=[*result.index.names,'spectra_ind','Peak'],aggfunc=dict4drop)

    logger.log(f"Shape of the dataframe on enter:{ftable.shape}")
    logger.log(f"Shape of the dataframe before drop:{result.shape} ({grftable.shape[0]-result.shape[0]})")
    logger.log(f"Shape of the dataframe after drop:{result.shape} ({grftable.shape[0]-result.shape[0]})")
    return result.reset_index(level=['spectra_ind','Peak']).reindex(columns=ftable_colarrange)

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
        #for spectra_ind in ftable.loc[idx,"spectra_ind"].unique():
        #    mask = ftable.loc[idx,"spectra_ind"]==spectra_ind
        if isinstance(norm[1],str):
            col_list=[norm[1]]
        else:
            col_list=norm[1]
        #for colname in col_list:
        # logger.log(f"{ftable.loc[idx,col_list]}")
        data_array = ftable.loc[idx,col_list].to_numpy()
        if len(data_array.shape)==1:
            data_array.shape = (data_array.shape[0],1)
        ftable.loc[idx,col_list] = normalize(data_array,norm=norm[0],axis=0)
        # logger.log(f"{ftable.loc[idx,col_list]}")
    return ftable

def Peak_assignment(ftable_batch,Xp_batch,Xl_batch,Xr_batch):
    """    
    Описание
    ----
    Вспомогательная функция к основной `Pgrouping_KD`. Определяет принадлежность значений mz к определённому значению пика  
    """
    if not ftable_batch.empty:
        for idx, peak in enumerate(Xp_batch):
            ftable_batch.loc[(ftable_batch['mz']>=Xl_batch[idx]) & (ftable_batch['mz']<=Xr_batch[idx]),"Peak"] = peak
    return ftable_batch

def mspeaks_KD(X, Y,oversegmentationfilter=0,peaklocation=1):
    """    
    Описание
    ----
    Вспомогательная функция к основной `Pgrouping_KD`. Находит пики и их границы в оценке плотности вероятности. 
    """
    n = X.size
    # Robust valley finding
    h = np.concatenate(([-1], np.where(np.diff(Y) != 0)[0], [n-1]))
    g = (np.diff(Y[[h[1],*h[1:]]]) <= 0) & (np.diff(Y[[*h[1:],h[-1]]]) >= 0)
    
    left_min = h[np.concatenate([g, [False]])] + 1
    right_min = h[np.concatenate([[False], g])]
    left_min = left_min[:-1]
    right_min = right_min[1:]
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
        
        val_max[j] = np.maximum(val_max[j], val_max[j + 1])
        val_max = np.delete(val_max, j + 1)
    return pkX,X[left_min], X[right_min]
