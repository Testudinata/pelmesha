from pelmesha.serving import DataSet
from pelmesha.cookbook import PipelineConfigurator
if __name__ == '__main__':

    signal_distortions = [(99, 100.2),
                        (102.09, 103.86),
                        (107.79, 109.4),
                        (112.65, 112.9),
                        (123.2, 123.6),
                        (127, 128.2),
                        (135.4, 136.12), 
                        (155.7, 156.6), 
                        (168.5, 170.8),
                        (188.75, 189.15),
                        (202.00, 202.55),
                        (203.13, 204.65),
                        (206.4, 206.8), 
                        (210.30, 210.60), 
                        (220.10, 221.02), 
                        (229.36, 231.53), 
                        (241.92, 243.0),
                        (277.4, 278.4), 
                        (285.83, 287.11),
                        (321.6, 322.0),
                        (375.0, 375.6),
                        (398.18, 401.15), 
                        (472.29, 472.89), 
                        (508.34, 512.61), 
                        (620.00, 622.81), 
                        (643.60, 645.03), 
                        (674.0, 676.17), 
                        (677.37, 683.2), 
                        (906.92, 908.83), 
                        (510.0, 511.2), 
                        (622.5, 627.5), 
                        (643.0, 644.5), 
                        (676.0, 679.5)]


    path = r"C:\Job_and_Literature\Esi_test_2"
    data = DataSet(path, rebuild_metadata = False, KDE_algo="tree",bwc=1,SNR_threshold=5,smooth_algo='GA', align_shift_range = (-0.25,0.25), zero_points_to_peaks_ext=True, resample_mz_step = 0.0125)
    data.set_reference_source(data.sources['Esi_test_2_3_t'], KDE_algo = 'tree', bwc = 1, SNR_threshold = 10, smooth_algo = 'GA', zero_points_to_peaks_ext=True)
    for source in data.sources.values():
        for roi in source.roi_configs.keys():
            if roi in [2,3,"2","3"]:
                source.roi_configs[roi].delete('Baseline')
                source.roi_configs[roi]['resample_mz_step'] = 0.006125
                source.roi_configs[roi]['align_shift_range'] = (-0.15,0.15)
                source.roi_configs[roi]['mz_segments_to_zero'] = signal_distortions
    ref_source = data.reference_source
    for roi in ref_source.roi_configs.keys():
        if roi in [2,3,"2","3"]:
            ref_source.roi_configs[roi].delete('Baseline')
            ref_source.roi_configs[roi]['resample_mz_step'] = 0.006125
            ref_source.roi_configs[roi]['align_shift_range'] = (-0.15,0.15)
            ref_source.roi_configs[roi]['mz_segments_to_zero'] = signal_distortions
    # data.set_reference_source(data.sources['1_test1'], KDE_algo = 'tree', bwc = 1, SNR_threshold = 8)

    rois = ['0','1',"2","3"]
    for roi in rois:
        data.get_reference_peaks(roi = roi,step=500,num_peaks_per_step=3)
        data.set_align_peaks_from_ref(rois = roi)
    data.peakpick(draw_mz_range=(750,800))
    data.estimate_peak_density_kde(draw_borders= 3)
    rois = ['0','1',"2","3"]
    features = {}
    for roi in rois:
        features[roi] = data.feature_matrix(rois = roi, countf=25, draw_borders= 3)
    # path = r"D:\Testing\Our_data\Rapiflex"

    # data = DataSet(path)

    # data.estimate_peak_density_kde()