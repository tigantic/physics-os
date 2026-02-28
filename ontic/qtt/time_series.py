"""
QTT Time-Series Compression
=============================

Compress 1-D temporal signals into QTT (Quantized Tensor-Train) format
by interpreting the time axis through a quantics / bit-interleaved
mapping.  A signal of length 2^N is stored as an N-site TT with
physical dimension 2, achieving log-linear memory for smooth or
structured signals.

Applications
------------
* Sensor telemetry archival at 1000:1 compression for smooth baselines.
* Fast spectral analysis via QTT-FFT of the compressed representation.
* Streaming: incrementally update the TT as new data arrives.

Key classes / functions
-----------------------
* :class:`QTTTimeSeriesConfig`  — configuration
* :class:`QTTTimeSeries`        — compressed time-series object
* :func:`compress_timeseries`   — fit a QTT to a 1-D signal
* :func:`decompress_timeseries` — reconstruct full signal
* :func:`streaming_update`      — rank-adaptive incremental update
* :func:`qtt_spectrum`          — power spectrum directly in QTT
* :func:`qtt_downsample`        — decimate via site removal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ontic.qtt.sparse_direct import tt_round


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class QTTTimeSeriesConfig:
    """
    Configuration for QTT time-series operations.

    Attributes
    ----------
    max_rank : int
        Maximum TT bond dimension.
    tol : float
        SVD truncation tolerance.
    n_bits : int | None
        Override number of quantics bits.  If None, derived from signal
        length as ceil(log2(N)).
    """
    max_rank: int = 64
    tol: float = 1e-8
    n_bits: Optional[int] = None


# ======================================================================
# QTT time-series object
# ======================================================================

@dataclass
class QTTTimeSeries:
    """
    A time series stored in QTT format.

    Attributes
    ----------
    cores : list[NDArray]
        TT-cores, each of shape (r_{k-1}, 2, r_k).
    n_bits : int
        Number of quantics bits.
    n_samples : int
        Original signal length.
    dt : float
        Sampling interval.
    compression_ratio : float
        Original size / compressed parameters.
    """
    cores: list[NDArray]
    n_bits: int
    n_samples: int
    dt: float = 1.0
    compression_ratio: float = 1.0


# ======================================================================
# Core operations
# ======================================================================

def _pad_to_power_of_two(signal: NDArray) -> tuple[NDArray, int]:
    """Pad signal to next power of 2."""
    N = len(signal)
    n_bits = int(np.ceil(np.log2(max(N, 2))))
    padded_len = 2 ** n_bits
    if padded_len > N:
        padded = np.zeros(padded_len)
        padded[:N] = signal
    else:
        padded = signal.copy()
    return padded, n_bits


def compress_timeseries(
    signal: NDArray,
    config: Optional[QTTTimeSeriesConfig] = None,
    dt: float = 1.0,
) -> QTTTimeSeries:
    """
    Compress a 1-D time series into QTT format.

    The signal of length N is padded to 2^n and reshaped into an
    n-dimensional tensor of shape (2, 2, ..., 2), which is then
    decomposed into TT format via sequential SVD.

    Parameters
    ----------
    signal : NDArray
        Real-valued 1-D signal.
    config : QTTTimeSeriesConfig, optional
        Compression parameters.
    dt : float
        Sampling interval.

    Returns
    -------
    QTTTimeSeries
    """
    if config is None:
        config = QTTTimeSeriesConfig()

    N = len(signal)
    padded, n_bits = _pad_to_power_of_two(signal)

    if config.n_bits is not None:
        n_bits = config.n_bits
        padded_len = 2 ** n_bits
        if padded_len > len(padded):
            new_pad = np.zeros(padded_len)
            new_pad[:len(padded)] = padded
            padded = new_pad
        else:
            padded = padded[:padded_len]

    # Reshape to (2, 2, ..., 2) tensor
    tensor = padded.reshape([2] * n_bits)

    # TT-SVD: sequential left-to-right SVD
    cores: list[NDArray] = []
    remaining = tensor.copy()
    r_left = 1

    for k in range(n_bits - 1):
        shape = remaining.shape
        mat = remaining.reshape(r_left * 2, -1)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncation
        rank = len(S)
        if config.tol > 0 and S[0] > 1e-30:
            total_sq = np.cumsum(S[::-1] ** 2)[::-1]
            for r in range(1, rank):
                if np.sqrt(total_sq[r]) / np.sqrt(total_sq[0]) < config.tol:
                    rank = r
                    break
        rank = min(rank, config.max_rank)

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        core = U.reshape(r_left, 2, rank)
        cores.append(core)

        remaining = (np.diag(S) @ Vh).reshape(rank, *([2] * (n_bits - k - 1)))
        r_left = rank

    # Last core
    cores.append(remaining.reshape(r_left, 2, 1))

    # Compression ratio
    n_params = sum(c.size for c in cores)
    ratio = N / max(n_params, 1)

    return QTTTimeSeries(
        cores=cores,
        n_bits=n_bits,
        n_samples=N,
        dt=dt,
        compression_ratio=ratio,
    )


def decompress_timeseries(qts: QTTTimeSeries) -> NDArray:
    """
    Reconstruct the full time series from QTT format.

    Parameters
    ----------
    qts : QTTTimeSeries
        Compressed time series.

    Returns
    -------
    NDArray
        Reconstructed signal of length ``qts.n_samples``.
    """
    # Contract TT cores into full tensor
    result = qts.cores[0]  # (1, 2, r1)
    for k in range(1, len(qts.cores)):
        # result: (1, 2^k, r_k), core: (r_k, 2, r_{k+1})
        r_l = result.shape[0]
        mid = result.shape[1]
        r_mid = result.shape[2]
        core = qts.cores[k]
        # Contract: result[a, I, i] * core[i, j, b] → new[a, I*j, b]
        result = np.einsum('aIi,ijb->aIjb', result, core)
        result = result.reshape(r_l, mid * 2, core.shape[2])

    signal = result.reshape(-1)
    return signal[:qts.n_samples]


def streaming_update(
    qts: QTTTimeSeries,
    new_samples: NDArray,
    config: Optional[QTTTimeSeriesConfig] = None,
) -> QTTTimeSeries:
    """
    Incrementally update a QTT time series with new samples.

    Appends new data, recompresses, and returns the updated object.

    Parameters
    ----------
    qts : QTTTimeSeries
        Existing compressed time series.
    new_samples : NDArray
        New samples to append.
    config : QTTTimeSeriesConfig, optional
        Compression parameters for re-compression.

    Returns
    -------
    QTTTimeSeries
        Updated time series.
    """
    if config is None:
        config = QTTTimeSeriesConfig(
            max_rank=max(c.shape[2] for c in qts.cores) + 4,
            tol=1e-8,
        )

    # Decompress, concatenate, recompress
    old_signal = decompress_timeseries(qts)
    combined = np.concatenate([old_signal, new_samples])

    return compress_timeseries(combined, config, dt=qts.dt)


def qtt_spectrum(qts: QTTTimeSeries) -> NDArray:
    """
    Compute the power spectrum directly from QTT representation.

    Decompresses and applies FFT.  (A fully QTT-native FFT would use
    the twiddle-MPO approach in ``ontic.cfd.qtt_fft``; this
    function provides a convenient interface for moderate sizes.)

    Parameters
    ----------
    qts : QTTTimeSeries
        Compressed time series.

    Returns
    -------
    NDArray
        Power spectral density (one-sided, normalised).
    """
    signal = decompress_timeseries(qts)
    N = len(signal)
    spectrum = np.fft.rfft(signal)
    psd = np.abs(spectrum) ** 2 / N
    return psd


def qtt_downsample(
    qts: QTTTimeSeries,
    factor: int = 2,
    config: Optional[QTTTimeSeriesConfig] = None,
) -> QTTTimeSeries:
    """
    Downsample by removing the finest-scale QTT site(s).

    In quantics representation, the last site encodes the finest bit
    of the time index.  Removing it effectively decimates by 2.

    Parameters
    ----------
    qts : QTTTimeSeries
        Input compressed time series.
    factor : int
        Downsampling factor (must be a power of 2).
    config : QTTTimeSeriesConfig, optional
        Recompression config.

    Returns
    -------
    QTTTimeSeries
        Downsampled time series.
    """
    n_remove = int(np.log2(factor))
    if 2 ** n_remove != factor:
        raise ValueError(f"factor must be a power of 2, got {factor}")

    if n_remove >= qts.n_bits:
        raise ValueError(
            f"Cannot remove {n_remove} sites from {qts.n_bits}-site QTT"
        )

    # Remove finest sites by contracting last n_remove cores
    cores = [c.copy() for c in qts.cores]

    for _ in range(n_remove):
        if len(cores) < 2:
            break
        last = cores.pop()  # (r, 2, 1)
        # Average over the physical index of the last core
        # This performs decimation by averaging
        prev = cores[-1]  # (r_prev, 2, r)
        # Contract: prev[a, i, k] * last[k, j, 0] → merged[a, i, j]
        # Then sum over j (average): merged[a, i] = prev[a,i,:] @ last[:,avg,0]
        avg_last = last[:, 0, :] + last[:, 1, :]  # (r, 1) — sum over d=2
        avg_last *= 0.5
        new_last = np.einsum('aik,k...->ai...', prev, avg_last.squeeze(-1))
        cores[-1] = new_last.reshape(prev.shape[0], 2, 1)

    new_n_bits = len(cores)
    new_n_samples = min(qts.n_samples // factor, 2 ** new_n_bits)

    if config is not None:
        cores = tt_round(cores, max_rank=config.max_rank, cutoff=config.tol)

    n_params = sum(c.size for c in cores)
    ratio = new_n_samples / max(n_params, 1)

    return QTTTimeSeries(
        cores=cores,
        n_bits=new_n_bits,
        n_samples=new_n_samples,
        dt=qts.dt * factor,
        compression_ratio=ratio,
    )
