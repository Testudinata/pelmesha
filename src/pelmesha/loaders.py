import os
from h5py import File
import gc
import warnings
import logging
import pandas as pd 
# Иерархия структур в наименованиях и в HDF5: 
# 1.Slide - слайд или пара слайдов, на котором/ых находятся образцы (Sample) (= одному пайплану эксперимента (образцы->стекло->нанесение матрицы->измерение), это может быть корневой папкой, в которой сохраняются все измерения одного такого эксперимента) 
# 2.Sample - образец измерения, в котором может быть несколько изучаемых областей (roi: region of interest). (= одному измерению области/ей, которые пользователь сам выбрал как отдельные по каким-либо параметрам) 
# 3.ROI - изучаемые области в образце. Для Rapiflex'а в одном файле .imzml может быть несколько ROI, которые могут иметь и разные настройки и прочее, но сохранены пользователем в одном файле .imzml.
# В ряде случаев Sample = ROI с одним roi "00"

# Более кратко: Slide - корневая папка со всеми измерениями -> Sample - сами измерения записанные в один файл slide, сохранённый в папке Slide -> ROI - отдельные области измерения.

### Base
def hdf5_Load(path_list, file_end=''):
    """
    Общее описание
    ----
    Базовый загрузчик hdf5
    
    :param path_list: list of str with paths to `hdf5` file
    :param file_end: Поиск файлов с определённым окончанием в названии помимо ".hdf5"
     
    :type path_list: list
    :type file_end: str

    :return: dictionary with hdf5 file objects
    :rtype: dict
    """
    logger("hdf5_Load",{**locals()})
    file_end=file_end+".hdf5"
    if isinstance(path_list, str):
        path_list=[path_list]
        
    hdf5path_list=find_paths(path_list,file_end=file_end)
    
    Slide_data={}
    for path in hdf5path_list:
        Slide_name=os.path.basename(path.replace(file_end,""))
        Slide_data[Slide_name] = File(path,"r")
    if not hdf5path_list:
        logger.warn(f"Data not readed due to missing hdf5 with spectra data (hdf5 with end \"{file_end}\" in the name is missing)", stacklevel=2)
    logger.ended()
    return Slide_data

def specdata_Load(path_list):
    """
    Функция открытия в режиме чтения `hdf5` данных обработанных спектров.
    """
    return hdf5_Load(path_list,file_end='_specdata')
def features_Load(path_list):
    """
    Функция открытия в режиме чтения `hdf5` данных пиклистов сгруппированных по mz.
    """
    return hdf5_Load(path_list,file_end='_features')
def rawdata_Load(path_list):
    """
    Функция открытия в режиме чтения `hdf5` данных сырых спектров.
    """
    return hdf5_Load(path_list,file_end='_rawdata')
def grouped_MSIdata_Load(path):
    """
    Функция открытия в режиме чтения `hdf5` данных пиклистов сгруппированных по mz по нескольким областям.
    :param path: direct path to hdf5 file ending by '_grouped_MSIdata.hdf5' 
    """
    gr_fdata = hdf5_Load(path,file_end='_grouped_MSIdata')
    group_list = list(gr_fdata.keys())
    return gr_fdata[group_list[0]]


def peakl2DF(batch_path, extr_columns=None,extract_coords = True, return_source_path = False, pivoting4val = None):
    """
    Общее описание
    ----
    Функция преобразует данные пиклисты `hdf5` в словарь с датафреймами пиклистов образцов согласно выставленным параметрам.

    :param batch_path: лист путей или путь к папке/файлу с `hdf5`. 
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где `"spectra_ind"` и `"mz"` или `"Peak"` экстрагируются всегда. Default: `None` - экстракция всех столбцов
    `"Intensity"`, `"Area"`, `"SNR"`, `"PextL"`, `"PextR"`, `"FWHML"`, `"FWHMR"`, `"Noise"`, `"Mean noise"`
    :param extract_coords: `True` - extracting to dict coordinates Dataframe, `False` - coordinates doesn't extracting. Default: `True`
    :param pivoting4val: list of columns or None (default) - extracted data is pivoted by index: spectra_ind, columns: Peak with fill_value = 0, and values: list of columns from pivoting4val. If None - do nothing about pivoting
    :param return_source_path: If `True` - return full path to source. Optional. Used in some functions.
    
    :type batch_path: `str` or `list`
    :type extr_columns: `list`
    :type extract_coords: `bool`
    :type pivoting4val: `list`
    :type return_source_path: `bool`

    :return: dictionary with peaklist and coordinates dataframes. `dict` structure as in `hdf5`. Return `tuple` if return_source_path is `True` with additional source pathes

    :rtype: `dict` or `tuple`
    """
    
    logger("peakl2DF",{**locals()})

    if isinstance(batch_path, str):
        batch_path=[batch_path]

    ### hdf5 load
    feat_type = "peaklists"
    Slide_data = specdata_Load(batch_path)
    if return_source_path:
        table,sourcse_path = table2DF(Slide_data,feat_type,extr_columns,extract_coords, return_source_path, pivoting4val)
        logger.ended()
        return table,sourcse_path
    else:
        logger.ended()
        return table2DF(Slide_data,feat_type,extr_columns,extract_coords, return_source_path, pivoting4val) 

def feat2DF(batch_path, extr_columns=None,extract_coords = True, return_source_path = False, pivoting4val = None):
    """
    Общее описание
    ----
    Функция преобразует данные фича-матрицы `hdf5` в словарь с датафреймами пиклистов образцов согласно выставленным параметрам.

    :param batch_path: лист путей или путь к папке/файлу с `hdf5`. 
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где экстрагируются всегда `"spectra_ind"` и `"mz"` или `"Peak"`. Default: `None` - экстракция всех столбцов
    `"Intensity"`,`"Area"`,`"SNR"`,`"PextL"`,`"PextR"`,`"FWHML"`,`"FWHMR"`,`"Noise"`,`"Mean noise"`
    :param extract_coords: `True` - extracting to dict coordinates Dataframe, `False` - coordinates doesn't extracting. Default: `True`
    :param pivoting4val: list of columns or None (default) - extracted data is pivoted by index: spectra_ind, columns: Peak with fill_value = 0, and values: list of columns from pivoting4val. If None - do nothing about pivoting
    :param return_source_path: If `True` - return full path to source. Optional. Used in some functions.
    
    :type batch_path: `str` or `list`
    :type extr_columns: `list`
    :type extract_coords: `bool`
    :type pivoting4val: `list`
    :type return_source_path: `bool`

    :return: dictionary with peaklist and coordinates dataframes. `dict` structure as in `hdf5`. Return `tuple` if return_source_path is `True` with additional source pathes

    :rtype: `dict` or `tuple`
    """
    logger("feat2DF",{**locals()})
    
    if isinstance(batch_path, str):
        batch_path=[batch_path]

    ### hdf5 load
    feat_type = "features"
    Slide_data = features_Load(batch_path)
    if return_source_path:
        table,sourcse_path = table2DF(Slide_data,feat_type,extr_columns,extract_coords, return_source_path, pivoting4val)
        logger.ended()
        return table,sourcse_path
    else:
        logger.ended()
        return table2DF(Slide_data,feat_type,extr_columns,extract_coords, return_source_path, pivoting4val) 
    
def table2DF(Slide_data, feat_type , extr_columns=None,extract_coords = True, return_source_path = False, pivoting4val = None):
    logger("table2DF",{**locals()})
    if isinstance(extr_columns, str):
        extr_columns=[extr_columns]
    DataFeat ={}
    Source_path={}
    for slides in list(Slide_data.keys()):
        DataFeat[slides]={}
        Source_path[slides] = Slide_data[slides].filename
        
        for sample in Slide_data[slides].keys():
            DataFeat[slides][sample]={}
            for roi in Slide_data[slides][sample].keys():
                try:
                    headers = list(Slide_data[slides][sample][roi][feat_type].attrs['Column headers'])
                except KeyError as error:
                    logger.warn(f"An error occurred on {slides} {sample} {roi}: {error}")
                    try:
                        logger.warn(f'HDF5 file from source {Source_path[slides]} on {slides} {sample} {roi} has datasets: {Slide_data[slides][sample][roi].keys()}')
                    except:
                        pass
                    continue

                DataFeat[slides][sample][roi]={}
                ## Setting columns to extract
                if "Peak" in headers:
                    mz_type = "Peak"
                else:
                    mz_type = "mz"
                if extr_columns is None:
                    extr_columns = headers
                    column_nums = range(len(extr_columns))
                else:
                    column_nums=[]

                    if "mz" in extr_columns:
                        del extr_columns[extr_columns.index("mz")]
                    if "Peak" in extr_columns:
                        del extr_columns[extr_columns.index("Peak")]
                    if "spectra_ind" not in extr_columns:
                        extr_columns.append("spectra_ind")
                    temp_extr_column = extr_columns.copy()
                    temp_extr_column.append(mz_type)

                    for head in headers:
                        if head in extr_columns:
                            column_nums.append(headers.index(head))
                            del temp_extr_column[temp_extr_column.index(head)]
                    if temp_extr_column:
                        logger.warn(f"Columns: {temp_extr_column} - are not extracted. Columns in loading dataset is {headers}")
                ## Setting columns to extract - Ended

                if Slide_data[slides][sample][roi][feat_type].shape[1] == len(headers):

                    DataFeat[slides][sample][roi][feat_type]=pd.DataFrame(Slide_data[slides][sample][roi][feat_type][:,column_nums], columns= (headers[column_num] for column_num in column_nums)).sort_values(['spectra_ind',mz_type])
                else:
                    DataFeat[slides][sample][roi][feat_type]=pd.DataFrame(Slide_data[slides][sample][roi][feat_type][column_nums,:].T, columns= (headers[column_num] for column_num in column_nums)).sort_values(['spectra_ind',mz_type])
                try:
                    DataFeat[slides][sample][roi][feat_type]=DataFeat[slides][sample][roi][feat_type].astype({"spectra_ind": int})
                except:
                    logger.warn("spectra_ind to int is unsuccessful")
                    pass
                if extract_coords:
                    try:
                        #print(Slide_data[slides][sample][roi]['xy'][:])
                        DataFeat[slides][sample][roi]["xy"] = pd.DataFrame(Slide_data[slides][sample][roi]['xy'][:],columns=["x","y"], index=pd.Index(range(Slide_data[slides][sample][roi]['xy'].shape[0]),name="spectra_ind"))
                        
                        logger.info(f"{slides}, {sample} and roi {roi}. x and y coordinates were extracted")
                        DataFeat[slides][sample][roi]["z"] = pd.Series(Slide_data[slides][sample][roi]['z'][:],columns=['z'], index=pd.Index(range(Slide_data[slides][sample][roi]['z'].shape[0]),name="spectra_ind"))
                        logger.info(f"{slides}, {sample} and roi {roi}. z coordinates were extracted")
                    except:
                        pass#print(f"{slides}, {sample} and roi {roi}. The extraction of other coordinates was unsuccessful")
                if pivoting4val:
                    DataFeat[slides][sample][roi][feat_type] = DataFeat[slides][sample][roi][feat_type].pivot_table(index="spectra_ind", columns="Peak",fill_value = 0, values =pivoting4val)
            if not DataFeat[slides][sample]:
                DataFeat[slides].pop(sample,None)
        if not DataFeat[slides]:
            DataFeat.pop(slides,None)
        Slide_data[slides].close()
    if not DataFeat:
        warnings.warn(f"Warning. Any dataset doesn't have {feat_type} data")
        logger.ended()
        return
    if return_source_path:
        logger.ended()
        return DataFeat, Source_path
    logger.ended()
    return DataFeat

def grouped_feat2DF(path, extr_columns=None,extract_coords = True, pivoting4val = None):
    """
    Общее описание
    ----
    Функция для работы со сгруппированными между имаджами пиклистами (признаками). Преобразует данные из `hdf5` в датафрейм.

    :param batch_path: лист путей или путь к папке/файлу с `hdf5`. 
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где 0 и 1 - экстрагируются всегда (`"spectra_ind"` и `"mz"` или `"Peak"`). Default: `None` - экстракция всех столбцов
    2 - `"Intensity"`, 3 -`"Area"`, 4 - `"SNR"`, 5 - `"PextL"`, 6 - `"PextR"`, 7 - `"FWHML"`, 8 - `"FWHMR"`, 9-`"Noise"`, 10-`"Mean noise"`
    :param extract_coords: `True` - extracting to dict coordinates Dataframe, `False` - coordinates doesn't extracting. Default: `True`
    :param pivoting4val: list of columns or None (default) - extracted data is pivoted by index: spectra_ind, columns: Peak with fill_value = 0, and values: list of columns from pivoting4val. If None - do nothing about pivoting
    
    :type batch_path: `str` or `list`
    :type extr_columns: `list`
    :type extract_coords: `bool`
    :type pivoting4val: `list`

    :return: dictionary with peaklist and coordinates dataframes. `dict` structure as in `hdf5`. Return `tuple` if return_source_path is `True` with additional source pathes

    :rtype: `dict` or `tuple`
    """
    logger("grouped_feat2DF",{**locals()},path)

    if isinstance(path, str):
        path=[path]
    
    grouped_images_DF=pd.DataFrame()
    Coords = pd.DataFrame(columns=['x','y'])

    Slide_data = grouped_MSIdata_Load(path)

    feat_type = "features"
    logger.log('Data converting to pandas DataFrame started')
    headers = Slide_data.attrs['Column headers']
    ####Probably delete this code. mz_type must be only "Peak"
    if "Peak" in headers:
        mz_type = "Peak"
    else:
        mz_type = "mz"
    ####
    for slides in list(Slide_data.keys()):
        for sample in Slide_data[slides].keys():
            for roi in Slide_data[slides][sample].keys():
                if extr_columns is None:
                    column_list = range(len(headers))
                    
                else:
                    column_list = extr_columns

                if Slide_data[slides][sample][roi][feat_type].shape[1] == len(headers):

                    DataFeat=pd.DataFrame(Slide_data[slides][sample][roi][feat_type], columns= headers).sort_values(['spectra_ind',mz_type])[Slide_data.attrs['Column headers'][column_list]]
                else:
                    DataFeat=pd.DataFrame(Slide_data[slides][sample][roi][feat_type][column_list,:].T, columns= headers).sort_values(['spectra_ind',mz_type])#[Slide_data[slides][sample][roi]['peaklists'].attrs['Column headers'][column_list]]
                try:
                    DataFeat=DataFeat.astype({"spectra_ind": int})
                except:
                    print("spectra_ind to int is unsuccessful")
                    pass
                ##concatenate
                n =DataFeat.shape[0]
                DataFeat.set_index([pd.Index([slides]*n),pd.Index([sample]*n),pd.Index([roi]*n)],inplace = True)

                if not set(DataFeat.columns).issubset(set(grouped_images_DF.columns)):
                    for col in list(DataFeat.columns):
                        if col not in set(grouped_images_DF.columns):
                            grouped_images_DF[col]=[]
                grouped_images_DF = pd.concat([DataFeat,grouped_images_DF])
                logger.log(f'Extracted data {slides}, {sample} roi {roi}. And concatenated')
                if extract_coords:
                    n = Slide_data[slides][sample][roi]['xy'].shape[0]
                    temp_coords = pd.DataFrame(Slide_data[slides][sample][roi]['xy'][:],columns=["x","y"]).set_index([pd.Index([slides]*n),pd.Index([sample]*n),pd.Index([roi]*n),pd.Index(range(n))])
                    logger.log(f"{slides}, {sample} and roi {roi}. x and y coordinates were extracted")
                    Coords = pd.concat([temp_coords,Coords])
    grouped_images_DF.index.names = ['slide','sample','roi']
    grouped_images_DF=grouped_images_DF.astype({'spectra_ind':int})
    logger.log('Data converting to pandas DataFrame ended')
    if pivoting4val:
        logger.log(f'Data pivoting started for values:{pivoting4val}')
        grouped_images_DF=grouped_images_DF.pivot_table(index=[grouped_images_DF.index,'spectra_ind'], columns="Peak",fill_value = 0, values =pivoting4val)
        grouped_images_DF.index.names = ['MS_image','spectra_ind']
        logger.log(f'Data pivoting ended')
    Slide_data.close()
    

    logger.ended()
    if extract_coords:
        Coords.index.names = ['slide','sample','roi',"spectra_ind"]
        return grouped_images_DF, Coords
    return grouped_images_DF

def IMGfeats_concat(paths,extr_columns,extracts_coords=True,processed_feat = False):
    """    
    Общее описание
    ----
    Функция объединяет данные пиклистов в разных `hdf5` в датафрейм пиклистов образцов согласно выставленным параметрам.

    :param paths: dict = {path_1:[[sample_1,[roi_list_1]],[sample_2,[roi_list_2]],....],path_2:[[sample_3,[roi_list_3]],[sample_4,[roi_list_4]],....]}, "path" - path to hdf5 file directory, "sample_n" - какой именно sample (string), если None - берёт всё, "roi_list_n" - список каких roi использовать, если отсутствует, то берёт всё (example: dict value: list[sample_n])
    :param extr_columns: Лист столбцов для экстракции из `hdf5`, где экстрагируются всегда `"spectra_ind"` и `"mz"` или `"Peak"`. Default: `None` - экстракция всех столбцов
    `"Intensity"`,`"Area"`,`"SNR"`,`"PextL"`,`"PextR"`,`"FWHML"`,`"FWHMR"`,`"Noise"`,`"Mean noise"`
    :param processed_feat: `True` - Dataframe from grouped peaklists, `False` - Dataframe from raw image peaklists. Default: `False`
    :param extract_coords: `True` - extracting to dict coordinates Dataframe, `False` - coordinates doesn't extracting. Default: `True`

    :type batch_path: `str` or `list`
    :type extr_columns: `list`
    :type processed_feat: `bool`
    :type extract_coords: `bool`

    :return: peaklist DataFrame, where slide, sample and roi are in index. Return `tuple` if `extract_coords` is `True` with additional Coords

    :rtype: `dict` or `tuple`
    """
    
    ### Data_loading
    logger("IMGfeats_concat",locals())
    grouped_images_DF=pd.DataFrame()
    Coords = pd.DataFrame(columns=['x','y'])
    if isinstance(paths,list):
        path_list=paths
        samples = False
        rois =  None
    elif isinstance(paths,dict):
        path_list=paths.keys()
        samples = True
    elif isinstance(paths,str):
        path_list = [paths]
    if isinstance(extr_columns, str):
        extr_columns=[extr_columns]
    for path in path_list:
       
        ### hdf5 load
        if processed_feat:
            feat_type = 'features'
            Slide_data = features_Load([path])
        else:
            feat_type = "peaklists"
            Slide_data = specdata_Load([path])
        ### slide load
        if not Slide_data:
            logger.warn(f'For path: {path}, data doesn\'t loaded')
        for slide in list(Slide_data.keys()):

            ### samples to load
            if samples:
                s_iter = paths[path]
                if s_iter is None:
                    s_iter = Slide_data[slide].keys()
            else:
                s_iter = Slide_data[slide].keys()

            for sample in s_iter:
                if len(sample)>1 and isinstance(sample,tuple):
                    rois = sample[1]
                    
                    sample = sample[0]
                else:
                    rois = Slide_data[slide][sample].keys()

                for roi in rois:                    
                    headers = Slide_data[slide][sample][roi][feat_type].attrs['Column headers']
                    if "Peak" in headers:
                        mz_type = "Peak"
                    else:
                        mz_type = "mz"
                    ## Setting columns to extract
                    if extr_columns is None:
                        column_nums = range(len(headers))
                    else:
                        
                        column_nums=[]


                        if "mz" in extr_columns:
                            del extr_columns[extr_columns.index("mz")]
                        if "Peak" in extr_columns:
                            del extr_columns[extr_columns.index("Peak")]
                        if "spectra_ind" not in extr_columns:
                            extr_columns.append("spectra_ind")
                        temp_extr_column = extr_columns.copy()
                        temp_extr_column.append(mz_type)

                        for head in headers:
                            if head in extr_columns:
                                column_nums.append(headers.index(head))
                                del temp_extr_column[temp_extr_column.index(head)]
                        if temp_extr_column:
                            logger.warn(f"Columns: {temp_extr_column} - are not extracted. Columns in loading dataset is {headers}")
                    ## Setting columns to extract - Ended
                    if Slide_data[slide][sample][roi][feat_type].shape[1] == len(headers):

                        #DataFeat[slide][sample][roi]=pd.DataFrame(Slide_data[slide][sample][roi][feat_type], columns= headers).sort_values(['spectra_ind',mz_type])[Slide_data[slide][sample][roi][feat_type].attrs['Column headers'][column_list]]
                        DataFeat=pd.DataFrame(Slide_data[slide][sample][roi][feat_type][:,column_nums], columns= (headers[column_num] for column_num in column_nums)).sort_values(['spectra_ind',mz_type])
                    else:
                        DataFeat=pd.DataFrame(Slide_data[slide][sample][roi][feat_type][column_nums,:].T, columns= (headers[column_num] for column_num in column_nums)).sort_values(['spectra_ind',mz_type])
                    
                    try:
                        #DataFeat[slide][sample][roi]=DataFeat[slide][sample][roi].astype({"spectra_ind": int})
                        DataFeat=DataFeat.astype({"spectra_ind": int})
                    except:
                        print("spectra_ind to int is unsuccessful")
                        pass
                    #n =DataFeat[slide][sample][roi].shape[0]
                    n =DataFeat.shape[0]
                    #DataFeat[slide][sample][roi].set_index([pd.Index([slide]*n,name='slide'),pd.Index([sample]*n,name='sample'),pd.Index([roi]*n,name='roi')],inplace = True)
                    DataFeat.set_index([pd.Index([slide]*n),pd.Index([sample]*n),pd.Index([roi]*n)],inplace = True)
                    
                    if not set(DataFeat.columns).issubset(set(grouped_images_DF.columns)):
                        for col in list(DataFeat.columns):
                            if col not in set(grouped_images_DF.columns):
                                grouped_images_DF[col]=[]
                    #grouped_images_DF = pd.concat([DataFeat[slide][sample][roi],grouped_images_DF])
                    grouped_images_DF = pd.concat([DataFeat,grouped_images_DF])
                    if extracts_coords:
#                        Coords[slide][sample][roi] = Slide_data[slide][sample][roi]['xy'][:]
                        n = Slide_data[slide][sample][roi]['xy'].shape[0]
                        temp_coords = pd.DataFrame(Slide_data[slide][sample][roi]['xy'][:],columns=["x","y"]).set_index([pd.Index([slide]*n),pd.Index([sample]*n),pd.Index([roi]*n),pd.Index(range(n))])
                        Coords = pd.concat([temp_coords,Coords])
                    #DataFeat[slides][sample][roi][feat_type]["xy"] = Slide_data[slides][sample][roi]['xy'][:]

            Slide_data[slide].close()
    grouped_images_DF.index.names = ['slide','sample','roi']

    if extracts_coords:
        Coords.index.names = ['slide','sample','roi','spectra_ind']
        logger.ended()
        return grouped_images_DF.astype({'spectra_ind':int}), Coords
    logger.ended()
    return grouped_images_DF

def hdf5_close():
    """
    Функция для закрытия всех hdf5 файлов разом
    """
    gc.collect()
    for obj in gc.get_objects():   # Browse through ALL objects
        try:
            if isinstance(obj, File):   # Just HDF5 files
                try:
                    obj.close()
                except:
                    pass # Was already closed
        except:
            pass

### utils functions
def find_paths(path_list,file_end = '.imzML'):
    """
    Общее описание
    ----
    Поисковик файлов
    
    :param path_list: list of str with paths to files
    :param file_end: Поиск файлов с определённым окончанием.
     
    :type path_list: list
    :type file_end: str

    :return: list with full paths to files
    :rtype: list
    """
    logger("find_paths",{**locals()})
    file_end = file_end.lower()
    files_path_list = []
    for path in path_list:
        if path.lower().endswith(file_end) and os.path.exists(path):
            files_path_list.append(path)
        else:
            for root, dirs, files in os.walk(path):
                for file in files: 
                    if file.lower().endswith(file_end):
                        files_path_list.append(os.path.join(root,file))
    logger.ended()
    return files_path_list

class logger:
    """
    logging messages in local package format:

    `logger.warn(text)` - write text as warning message to log

    `logger.log(text)` - write text message to log

    `logger.ended()` - write message of successful end of function to log
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name=[]
    getted_log=logging.getLogger()
    for handler in getted_log.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.WARN)
    def __init__(self,func_name,args,path = None):        
        if not path:
            try:
                os.mkdir('logs')
            except:
                pass
            path_to_file_handler = os.path.join("logs",str(func_name))+"_log.log"
        else:
            path_to_file_handler = os.path.join(path,str(func_name))+"_log.log"
        
        h_not_exist = True
        for handler in self.getted_log.handlers:
            if isinstance(handler, logging.FileHandler):
                h_not_exist = False
                handler.setLevel(logging.INFO)
            # if isinstance(handler, logging.StreamHandler):
            #     handler.setLevel(logging.WARN)
        if h_not_exist:
            fhandler = logging.FileHandler(filename=path_to_file_handler, mode="w")
            fhandler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            fhandler.setLevel(logging.INFO)
            self.getted_log.addHandler(fhandler)

            
        self.name.append(func_name)
        self.getted_log.info(f"====================================Function {func_name} arguments========================================")
        for arg in args.keys():
            self.getted_log.info(f"{arg} = {args[arg]}")
        self.getted_log.info(f"====================================Function {func_name} STARTED========================================")
    def warn(text):
        logger.getted_log.warn(f"{text}")
    def log(text):
        logger.getted_log.info(f"{text}")
    def ended():
        logger.getted_log.info(f"====================================Function {logger.name[-1]} ENDED==========================================")
        if len(logger.name)<2:
            del logger.name[-1]

            for handler in logger.getted_log.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.getted_log.removeHandler(handler)
        else:
            del logger.name[-1]
    