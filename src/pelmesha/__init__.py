"""
pelmesha

A Python library for loading, preprocessing, processing, storing, and
extracting peak lists of Mass Spectrometry Imaging (MSI) data.

Public API
----------
- :class:`~pelmesha.cookbook.PipelineConfigurator` — configuration manager for
  the MSI processing pipeline.
- :class:`~pelmesha.filling.DataSource` — handles a single raw data source
  (imzML / CDF) and its metadata.
- :class:`~pelmesha.serving.DataSet` — central class for managing and
  processing multiple data sources.
"""

__author__ = 'Andrey Kuzin'
__credits__ = 'Moscow Institute of Physics and Technology'

from pelmesha.cookbook import PipelineConfigurator
from pelmesha.serving import DataSet
from pelmesha.filling import DataSource

__all__ = ['PipelineConfigurator', 'DataSource', 'DataSet']
