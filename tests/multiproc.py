from pelmesha.serving import DataSet
from pelmesha.cookbook import PipelineConfigurator
if __name__ == '__main__':

    # Example of a complex configuration where each ROI is treated as a separate channel
    # ROIs 0, 1 → negative and positive modes in ion trap mass analyzer
    # ROIs 2, 3 → negative and positive modes in Orbitrap mass analyzer

    # Parameter for zeroing induced spurious signal in mass spectra
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


    path = r"C:\Job_and_Literature\Esi_test"
    # Create a dataset with common parameters for all sources, which we’ll later fine‑tune for individual ROIs.
    data = DataSet(path, KDE_algo="tree",
                bwc=1,
                SNR_threshold=5,
                smooth_algo='GA', 
                align_shift_range = (-0.25,0.25), 
                zero_points_to_peaks_ext=True, 
                resample_mz_step = 0.0125,
                fwhm_merge_factor = 0.85,
                mz_crop_range = (100,1000))
    data.set_reference_source(data.sources['Esi_test_3_t'], KDE_algo = 'tree', bwc = 1, SNR_threshold = 10, smooth_algo = 'GA', zero_points_to_peaks_ext=True)
    for source in data.sources.values():
        for roi in source.roi_configs.keys():
            if roi in ["2","3"]: #Fine-tuning for ROIs 2 and 3
                source.roi_configs[roi].delete('Baseline')
                source.roi_configs[roi]['resample_mz_step'] = 0.006125
                source.roi_configs[roi]['align_shift_range'] = (-0.15,0.15)
                source.roi_configs[roi]['mz_segments_to_zero'] = signal_distortions
                source.roi_configs[roi]['smooth_algo'] = None            
    ref_source = data.reference_source
    for roi in ref_source.roi_configs.keys():
        if roi in ["2","3"]: #Fine-tuning for ROIs 2 and 3 of the reference source
            ref_source.roi_configs[roi].delete('Baseline')
            ref_source.roi_configs[roi]['resample_mz_step'] = 0.006125
            ref_source.roi_configs[roi]['align_shift_range'] = (-0.15,0.15)
            ref_source.roi_configs[roi]['mz_segments_to_zero'] = signal_distortions
            ref_source.roi_configs[roi]['smooth_algo'] = None

    allowed_coords = {"RT":[(0.25*60,2.69*60),(3.09*60,5.5*60)]} 
    rois = ['0','1',"2","3"]
    for roi in rois: # Setting the reference peaks for each ROI
        data.get_reference_peaks(roi = roi,step=300,num_peaks_per_step=3, min_occurence = 0.95, allowed_coords = allowed_coords, free_cpus=19)
        data.set_align_peaks_from_ref(rois = roi)

    data.peakpick(sample_name="Esi_test_31",draw_mz_range=(750,800), allowed_coords = allowed_coords)
    data.estimate_peak_density_kde(draw_borders= 3)

    # Creating feature matrix for each ROI/channel
    rois = ['0','1',"2","3"]
    features = {}
    parq_path = r"C:\Job_and_Literature\Esi_test\features_roi_"
    for roi in rois:
        path = parq_path + roi + ".parq"
        features[roi] = data.feature_matrix(save_path = path, rois = roi, countf=25, draw_borders= 3)