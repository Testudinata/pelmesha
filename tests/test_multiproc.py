from pelmesha.serving import DataSet
from pelmesha.cookbook import PipelineConfigurator
if __name__ == '__main__':
    # dataset = DataSet()
    # path = r"D:\Testing\Our_data\Rapiflex"
    # configs = PipelineConfigurator()
    # configs['align_peaks'] = [128,768,769]
    # configs['smooth_algo'] = 'GA'
    # configs['smooth_window'] = 5
    # configs['SNR_threshold'] = 3
    # configs['resample_mz_step'] = 0.075

    # dataset.add_sources(path, configs) # or dataset(path)

    # dataset.peakpick(draw_mz_range=[600,800],free_cpus=18)

    configs = PipelineConfigurator(SNR_threshold=9)
    configs.delete('Baseline')
    print(configs)
    path = r'D:\Testing\Our_data\Orbitrap\1'
    dataset = DataSet()
    dataset.add_sources(path, configs)
    dataset.peakpick()