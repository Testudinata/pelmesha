"""
Tests for DataSource.split_idxs RAM-budget batching.

Covers the continuous and discontinuous data branches, the CPU-count
rebalancing fallback, index normalization, and the RAM budget parameters
(cpu_count, Ramcap_GB, size_per_spec, dtype).

The tests build a lightweight DataSource stub (bypassing ``__init__``,
which would require an actual data file and its HDF5 metadata) so the
batching logic can be exercised in isolation.
"""

import types

import numpy as np
import pandas as pd
import pytest
from pelmesha.dough import Indexator
from pelmesha.filling import DataSource
pytestmark = pytest.mark.filling
# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def _make_datasource(dcont, dtype="uint8", ramcap_bytes=100, spectrum_sizes=None,
                     roi_segments=None):
    """Create a DataSource stub with only the attributes split_idxs needs."""
    ds = DataSource.__new__(DataSource)
    ds.loader = types.SimpleNamespace(dcont=dcont)
    ds.Ramcap = ramcap_bytes
    ds.metadata = pd.DataFrame({"dtype_raw": [dtype]}, index=["sample"])
    if roi_segments is None:
        roi_segments = np.array([[0, 1]])
    ds.roi_metadata = pd.DataFrame(
        {"idxroi": [Indexator(roi_segments)]}, index=["ROI0"]
    )
    if spectrum_sizes is None:
        ds.get_spectrum_sizes = lambda idxs: [1] * Indexator(idxs).count
    else:
        ds.get_spectrum_sizes = lambda idxs: spectrum_sizes
    return ds


def _expand(segments):
    """Expand index segments into the full ordered list of covered indices."""
    segments = np.asarray(segments, dtype=np.int64)
    return np.concatenate([np.arange(start, end) for start, end in segments])


def _flat_batches(batches):
    """Stack all batch segments into a single (N, 2) array."""
    return np.vstack([np.asarray(batch) for batch in batches])


def _assert_covers(batches, idxs):
    """Assert that batches partition the input index segments exactly."""
    got = _expand(_flat_batches(batches))
    want = _expand(idxs)
    assert np.array_equal(got, want), f"batches cover {got} but input is {want}"


def _assert_count(batches, idxs):
    """Assert that the total number of spectra is preserved."""
    got = Indexator(_flat_batches(batches)).count
    want = Indexator(idxs).count
    assert got == want, f"spectra lost: {got} != {want}"


# --------------------------------------------------------------------------- #
#  Tests: continuous data (dcont=True)                                        #
# --------------------------------------------------------------------------- #

class TestSplitIdxsContinuous:
    """split_idxs on continuous data (shared m/z scale, d=1)."""

    def test_splits_single_segment_into_fixed_chunks(self):
        """A segment larger than the chunk size is cut into equal chunks."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100)
        batches = ds.split_idxs(idxs=np.array([[0, 100]]), size_per_spec=5)
        expected = np.array([[0, 20], [20, 40], [40, 60], [60, 80], [80, 100]])
        assert np.array_equal(_flat_batches(batches), expected)

    def test_merges_small_segments_into_shared_remainder_batch(self):
        """Small adjacent segments are merged into one batch up to the chunk size."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=6)
        batches = ds.split_idxs(idxs=np.array([[0, 4], [4, 8]]), size_per_spec=1)
        assert len(batches) == 2
        assert np.array_equal(batches[0], [[0, 4], [4, 6]])
        assert np.array_equal(batches[1], [[6, 8]])

    @pytest.mark.parametrize(
        "ramcap_bytes,size_per_spec,idxs",
        [
            (100, 5, [[0, 100]]),                      # exact chunks
            (6, 1, [[0, 4], [4, 8]]),                  # remainder merge
            (60, 3, [[0, 100], [150, 170]]),           # big + small segments
            (33, 2, [[0, 16], [40, 44], [100, 140]]),  # several segments
            (1024, 7, [[3, 13], [20, 23], [50, 51]]),  # tiny segments -> one batch
        ],
    )
    def test_preserves_all_spectra(self, ramcap_bytes, size_per_spec, idxs):
        """No spectra are lost or duplicated regardless of segment layout."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=ramcap_bytes)
        idxs = np.array(idxs)
        batches = ds.split_idxs(idxs=idxs, size_per_spec=size_per_spec,
                                cpu_count=2)
        _assert_covers(batches, idxs)
        _assert_count(batches, idxs)

    def test_chunk_size_depends_on_raw_dtype(self):
        """Wider dtypes hold fewer spectra per batch for the same RAM budget."""
        ds_u8 = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100)
        ds_f64 = _make_datasource(dcont=True, dtype="float64", ramcap_bytes=100)
        idxs = np.array([[0, 100]])
        b_u8 = ds_u8.split_idxs(idxs=idxs, size_per_spec=10)
        b_f64 = ds_f64.split_idxs(idxs=idxs, size_per_spec=10)
        assert len(b_u8) < len(b_f64)

    def test_returns_single_batch_for_large_budget(self):
        """A segment below the chunk size stays in a single batch."""
        ds = _make_datasource(dcont=True, dtype="float64", ramcap_bytes=2 ** 20,
                              spectrum_sizes=[1000] * 120)
        batches = ds.split_idxs(idxs=np.array([[0, 120]]), cpu_count=1)
        assert len(batches) == 1
        assert np.array_equal(batches[0], [[0, 120]])


# --------------------------------------------------------------------------- #
#  Tests: discontinuous data (dcont=False)                                    #
# --------------------------------------------------------------------------- #

class TestSplitIdxsDiscontinuous:
    """split_idxs on discontinuous data (per-spectrum m/z, d=2)."""

    def test_splits_by_accumulated_spectrum_sizes(self):
        """Batches are cut when the accumulated spectrum sizes reach the budget."""
        ds = _make_datasource(dcont=False, dtype="uint8", ramcap_bytes=10,
                              spectrum_sizes=[3, 1, 1, 2, 3, 1, 1, 2])
        batches = ds.split_idxs(idxs=np.array([[0, 8]]))
        assert np.array_equal(_flat_batches(batches), [[0, 2], [2, 4], [4, 8]])

    def test_keeps_gap_between_segments(self):
        """Index gaps are preserved: separate segments never share a batch."""
        ds = _make_datasource(dcont=False, dtype="uint8", ramcap_bytes=2 ** 20,
                              spectrum_sizes=[1] * 6)
        batches = ds.split_idxs(idxs=np.array([[0, 3], [10, 13]]))
        assert len(batches) == 2
        assert np.array_equal(batches[0], [[0, 3]])
        assert np.array_equal(batches[1], [10, 13])

    @pytest.mark.parametrize(
        "ramcap_bytes,spectrum_sizes,idxs",
        [
            (10, [3, 1, 1, 2, 3, 1, 1, 2], [[0, 8]]),
            (12, [4, 2, 2, 4, 1, 1, 5, 1], [[0, 4], [10, 14]]),
            (2 ** 10, [7, 3, 1, 9, 2, 2, 2, 2], [[1, 5], [8, 12]]),
        ],
    )
    def test_preserves_all_spectra(self, ramcap_bytes, spectrum_sizes, idxs):
        """No spectra are lost across mixed spectrum sizes and segment gaps."""
        ds = _make_datasource(dcont=False, dtype="uint8",
                              ramcap_bytes=ramcap_bytes,
                              spectrum_sizes=spectrum_sizes)
        idxs = np.array(idxs)
        batches = ds.split_idxs(idxs=idxs)
        _assert_covers(batches, idxs)
        _assert_count(batches, idxs)

    @pytest.mark.xfail(
        strict=False,
        reason="Known edge case (TODO in split_idxs): the last spectrum is dropped "
        "when it exactly fills the final batch",
    )
    def test_exact_final_batch_is_a_known_bug(self):
        """Documenting the known boundary bug where the trailing flush loses a spectrum."""
        ds = _make_datasource(dcont=False, dtype="uint8", ramcap_bytes=6,
                              spectrum_sizes=[1] * 6)
        batches = ds.split_idxs(idxs=np.array([[0, 6]]))
        _assert_count(batches, np.array([[0, 6]]))

    def test_size_per_spec_switches_to_chunk_batching(self):
        """Passing size_per_spec routes discontinuous data through chunk batching."""
        ds = _make_datasource(dcont=False, dtype="uint8", ramcap_bytes=6,
                              spectrum_sizes=[1] * 8)
        idxs = np.array([[0, 8]])
        batches = ds.split_idxs(idxs=idxs, size_per_spec=1)
        _assert_covers(batches, idxs)
        _assert_count(batches, idxs)


# --------------------------------------------------------------------------- #
#  Tests: rebalancing when batches are fewer than CPU count                  #
# --------------------------------------------------------------------------- #

class TestSplitIdxsRebalance:
    """Fallback that re-splits segments evenly into cpu_count * 3 batches."""

    def test_rebalances_when_batch_count_below_cpu_count(self):
        """Too few batches for the CPU count trigger an even re-split."""
        ds = _make_datasource(dcont=True, dtype="float64", ramcap_bytes=2 ** 20,
                              spectrum_sizes=[100] * 120)
        batches = ds.split_idxs(idxs=np.array([[0, 120]]), cpu_count=4)
        assert len(batches) == 12  # cpu_count * 3
        assert np.array_equal(batches[0], [[0, 10]])
        assert np.array_equal(batches[-1], [[110, 120]])
        _assert_count(batches, np.array([[0, 120]]))

    def test_rebalance_preserves_segment_gaps(self):
        """The even re-split still keeps the original index gaps intact."""
        ds = _make_datasource(dcont=True, dtype="float64", ramcap_bytes=2 ** 20,
                              spectrum_sizes=[1] * 120)
        idxs = np.array([[0, 60], [100, 160]])
        batches = ds.split_idxs(idxs=idxs, cpu_count=4)
        assert len(batches) == 12
        _assert_covers(batches, idxs)
        _assert_count(batches, idxs)

    def test_cpu_count_scales_down_batch_capacity(self):
        """More CPU cores means a smaller per-batch budget and more batches."""
        ds = _make_datasource(dcont=True, dtype="float64", ramcap_bytes=2 ** 20,
                              spectrum_sizes=[100] * 5000)
        idxs = np.array([[0, 5000]])
        b1 = ds.split_idxs(idxs=idxs, cpu_count=1)
        b2 = ds.split_idxs(idxs=idxs, cpu_count=2)
        assert len(b1) == 4
        assert len(b2) == 8

    def test_explicit_d_factor_scales_capacity(self):
        """Passing d=2 halves the per-batch budget compared with d=1."""
        ds = _make_datasource(dcont=True, dtype="float64", ramcap_bytes=2 ** 20,
                              spectrum_sizes=[100] * 5000)
        idxs = np.array([[0, 5000]])
        d1_batches = ds.split_idxs(idxs=idxs, d=1)
        d2_batches = ds.split_idxs(idxs=idxs, d=2)
        assert len(d2_batches) > len(d1_batches)
        _assert_count(d2_batches, idxs)


# --------------------------------------------------------------------------- #
#  Tests: index normalization (_normalize_indices)                            #
# --------------------------------------------------------------------------- #

class TestSplitIdxsInputNormalization:
    """Accepted idxs input forms: int, slice, list/tuple, Indexator, None."""

    @pytest.mark.parametrize("idx", [5, np.int64(5), 5.0])
    def test_accepts_single_spectrum_index(self, idx):
        """A single index is treated as the segment [idx, idx + 1)."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100)
        batches = ds.split_idxs(idxs=idx, size_per_spec=5)
        assert np.array_equal(_flat_batches(batches), [[5, 6]])

    @pytest.mark.parametrize("idxs", [slice(0, 10), (0, 10), [0, 10],
                                      np.array([0, 10])])
    def test_accepts_slice_tuple_list_and_flat_array(self, idxs):
        """slice, tuple, list and flat 1-D arrays all normalize to one segment."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100)
        batches = ds.split_idxs(idxs=idxs, size_per_spec=5)
        _assert_covers(batches, [[0, 10]])

    def test_accepts_indexator_input(self):
        """An Indexator is unwrapped to a plain ndarray view."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100)
        batches = ds.split_idxs(idxs=Indexator([[0, 10]]), size_per_spec=5)
        assert np.array_equal(_flat_batches(batches), [[0, 10]])

    def test_none_idxs_uses_all_roi_spectra(self):
        """With idxs=None the whole roi_metadata index ranges are used."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100,
                              roi_segments=np.array([[0, 50], [100, 150]]))
        batches = ds.split_idxs(idxs=None, size_per_spec=1)
        _assert_covers(batches, [[0, 50], [100, 150]])
        _assert_count(batches, [[0, 50], [100, 150]])

    @pytest.mark.xfail(
        strict=False,
        reason="Known limitation: a ROI name string yields an Indexator from "
        "_normalize_indices, and iterating an Indexator yields ints (crash)",
    )
    def test_roi_name_string_is_a_known_limitation(self):
        """Documenting that passing a ROI name string currently crashes."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100,
                              roi_segments=np.array([[0, 10]]))
        batches = ds.split_idxs(idxs="ROI0", size_per_spec=5)
        _assert_count(batches, [[0, 10]])


# --------------------------------------------------------------------------- #
#  Tests: RAM budget parameters and output contract                           #
# --------------------------------------------------------------------------- #

class TestSplitIdxsBudgetParams:
    """Ramcap_GB override, spectrum-size queries and the output contract."""

    def test_ramcap_gb_override_matches_instance_ramcap(self):
        """An explicit Ramcap_GB produces the same batches as self.Ramcap."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=6)
        batches = ds.split_idxs(idxs=np.array([[0, 4], [4, 8]]),
                                size_per_spec=1, Ramcap_GB=6 / 1024 ** 3)
        assert np.array_equal(_flat_batches(batches), [[0, 4], [4, 6], [6, 8]])

    def test_spectrum_sizes_queried_for_normalized_indices(self):
        """get_spectrum_sizes receives the normalized (N, 2) index array."""
        calls = []
        ds = _make_datasource(dcont=False, dtype="uint8", ramcap_bytes=2 ** 20)
        ds.get_spectrum_sizes = (
            lambda idxs: calls.append(np.asarray(idxs).copy())
            or [1] * Indexator(idxs).count
        )
        ds.split_idxs(idxs=5)
        assert len(calls) == 1
        assert calls[0].shape == (1, 2)
        assert calls[0].tolist() == [[5, 6]]

    def test_empty_segment_is_allowed(self):
        """A zero-length segment contributes nothing and does not break batching."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100)
        idxs = np.array([[5, 5]])
        batches = ds.split_idxs(idxs=idxs, size_per_spec=5)
        _assert_covers(batches, idxs)
        _assert_count(batches, idxs)

    def test_returns_list_of_segment_arrays(self):
        """The return value is a list whose entries are index-segment arrays."""
        ds = _make_datasource(dcont=True, dtype="uint8", ramcap_bytes=100)
        batches = ds.split_idxs(idxs=np.array([[0, 100]]), size_per_spec=5)
        assert isinstance(batches, list)
        for batch in batches:
            arr = np.asarray(batch)
            assert arr.ndim >= 1
            assert arr.shape[-1] == 2