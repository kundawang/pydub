"""
True peak (inter-sample peak) measurement.

Digital sample peaks can sit under 0 dBFS while the reconstructed analog
waveform overshoots full scale between samples -- which is why a master that
never clips on a sample-peak meter can still clip after lossy encoding or
D/A conversion. This module estimates the true peak per channel by
oversampling the signal with a fixed 4x polyphase interpolation filter and
reporting the highest peak found, alongside the plain sample peak, so both
can be told apart (dBFS vs dBTP).

Long audio is processed in chunks; the FIR filter state (the last
``TAPS_PER_PHASE`` samples of each channel) is carried across chunk
boundaries so the result is independent of the chunk size. The source
``AudioSegment`` data is never modified.
"""

from __future__ import division

import math
from array import array as _array

from .utils import audioop, ratio_to_db

try:
    import numpy as _np
except ImportError:
    _np = None

# Fixed oversampling factor (>= 4x as required for true peak estimation)
OVERSAMPLE_FACTOR = 4

# FIR taps per polyphase branch. The full interpolation filter has
# OVERSAMPLE_FACTOR * TAPS_PER_PHASE + 1 taps.
TAPS_PER_PHASE = 32

_DEFAULT_CHUNK_MS = 1000


def _design_interpolation_filter():
    """
    Windowed-sinc lowpass for OVERSAMPLE_FACTOR-times interpolation.

    Returns a list of OVERSAMPLE_FACTOR polyphase branches, each padded with
    zeros to TAPS_PER_PHASE + 1 coefficients. The filter is a Nyquist filter
    (h[0] == 1, h[k * OVERSAMPLE_FACTOR] == 0), so samples at the original
    positions pass through unchanged and the true peak is always >= the
    sample peak (up to floating point rounding).
    """
    oversample = OVERSAMPLE_FACTOR
    taps_per_phase = TAPS_PER_PHASE
    n_taps = oversample * taps_per_phase + 1
    center = (n_taps - 1) / 2.0

    taps = []
    for n in range(n_taps):
        x = n - center
        if x == 0:
            sinc = 1.0
        else:
            sinc = oversample * math.sin(math.pi * x / oversample) / (math.pi * x)
        # Blackman window
        window = (0.42
                  - 0.5 * math.cos(2.0 * math.pi * n / (n_taps - 1))
                  + 0.08 * math.cos(4.0 * math.pi * n / (n_taps - 1)))
        taps.append(sinc * window)

    phases = []
    for p in range(oversample):
        coeffs = [taps[oversample * k + p]
                  for k in range(taps_per_phase + 1)
                  if oversample * k + p < n_taps]
        coeffs.extend([0.0] * (taps_per_phase + 1 - len(coeffs)))
        phases.append(coeffs)
    return phases


_PHASES = _design_interpolation_filter()
_PHASES_REVERSED = [list(reversed(coeffs)) for coeffs in _PHASES]


def _finite(value):
    """Replace non-finite input with 0.0 so results can never go NaN."""
    return value if math.isfinite(value) else 0.0


def _channel_samples(data, sample_width, channels, channel):
    """
    Extract one channel as a list of floats normalized to full scale == 1.0.

    8-bit and 24-bit PCM are widened with audioop so every bit depth ends up
    in a signed array type (8-bit data is signed, matching pydub's internal
    representation). Non-finite values are sanitized to 0.0.
    """
    if sample_width == 1:
        converted = audioop.lin2lin(data, 1, 2)
        typecode, norm = "h", 32768.0
    elif sample_width == 2:
        converted = data
        typecode, norm = "h", 32768.0
    elif sample_width == 3:
        converted = audioop.lin2lin(data, 3, 4)
        typecode, norm = "i", 2147483648.0
    elif sample_width == 4:
        converted = data
        typecode, norm = "i", 2147483648.0
    else:
        raise ValueError("unsupported sample width: %r" % (sample_width,))

    samples = _array(typecode, converted)
    channel_samples = samples[channel::channels]
    return [_finite(v / norm) for v in channel_samples]


def _measure_channel_python(samples, chunk_frames):
    """Pure-python oversampled peak measurement (reference implementation)."""
    taps_per_phase = TAPS_PER_PHASE
    history = [0.0] * taps_per_phase
    peak = 0.0
    n_samples = len(samples)
    for start in range(0, n_samples, chunk_frames):
        chunk = samples[start:start + chunk_frames]
        buf = history + chunk
        n_chunk = len(chunk)
        for i in range(n_chunk):
            idx = taps_per_phase + i
            for coeffs in _PHASES:
                acc = 0.0
                for j, c in enumerate(coeffs):
                    acc += buf[idx - j] * c
                if acc > peak:
                    peak = acc
                elif -acc > peak:
                    peak = -acc
        # carry the filter state across the chunk boundary
        history = buf[n_chunk:]
    return peak


def _measure_channel_numpy(samples, chunk_frames):
    """numpy-accelerated oversampled peak measurement."""
    taps_per_phase = TAPS_PER_PHASE
    phases = [_np.asarray(coeffs) for coeffs in _PHASES_REVERSED]
    samples = _np.asarray(samples, dtype=_np.float64)
    history = _np.zeros(taps_per_phase, dtype=_np.float64)
    peak = 0.0
    n_samples = len(samples)
    for start in range(0, n_samples, chunk_frames):
        chunk = samples[start:start + chunk_frames]
        buf = _np.concatenate([history, chunk])
        for coeffs in phases:
            # np.correlate with the reversed branch computes the same
            # polyphase dot products as _measure_channel_python
            out = _np.correlate(buf, coeffs, "valid")
            chunk_peak = _np.max(_np.abs(out))
            if chunk_peak > peak:
                peak = float(chunk_peak)
        history = buf[len(chunk):]
    return peak


def _measure_channel(samples, chunk_frames):
    if _np is not None:
        return _measure_channel_numpy(samples, chunk_frames)
    return _measure_channel_python(samples, chunk_frames)


class TruePeakMeasurement(object):
    """
    Result of a true peak measurement.

    ``sample_peaks`` / ``true_peaks`` are per-channel linear peaks relative
    to full scale (1.0 == 0 dBFS); the dB variants express the same values
    in dBFS / dBTP. Silence and empty audio report 0.0 (-inf dB), never NaN.
    """

    def __init__(self, sample_peaks, true_peaks, oversample=OVERSAMPLE_FACTOR):
        self.sample_peaks = tuple(float(p) for p in sample_peaks)
        self.true_peaks = tuple(float(p) for p in true_peaks)
        self.oversample = int(oversample)

    @property
    def channels(self):
        return len(self.sample_peaks)

    @property
    def sample_peaks_dbfs(self):
        return tuple(ratio_to_db(p) for p in self.sample_peaks)

    @property
    def true_peaks_dbtp(self):
        return tuple(ratio_to_db(p) for p in self.true_peaks)

    @property
    def max_sample_peak(self):
        return max(self.sample_peaks) if self.sample_peaks else 0.0

    @property
    def max_true_peak(self):
        return max(self.true_peaks) if self.true_peaks else 0.0

    @property
    def max_sample_peak_dbfs(self):
        return ratio_to_db(self.max_sample_peak)

    @property
    def max_true_peak_dbtp(self):
        return ratio_to_db(self.max_true_peak)

    def __eq__(self, other):
        if not isinstance(other, TruePeakMeasurement):
            return NotImplemented
        return (self.sample_peaks == other.sample_peaks
                and self.true_peaks == other.true_peaks
                and self.oversample == other.oversample)

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    def __repr__(self):
        return ("TruePeakMeasurement(sample_peaks=%r, true_peaks=%r, "
                "oversample=%r)" % (self.sample_peaks, self.true_peaks,
                                    self.oversample))


def measure_true_peak(data, sample_width, channels, frame_rate,
                      chunk_ms=_DEFAULT_CHUNK_MS):
    """
    Measure per-channel sample peaks and true (inter-sample) peaks of raw
    interleaved PCM ``data``.

    Returns a :class:`TruePeakMeasurement`. Empty input, silence and
    non-finite samples yield stable (0.0 / -inf dB) results. A ``channels``
    value below 1 or an unsupported ``sample_width`` raises ValueError.
    Trailing bytes that do not form a complete frame are ignored.
    """
    if channels < 1:
        raise ValueError("channels must be >= 1, got %r" % (channels,))
    if sample_width not in (1, 2, 3, 4):
        raise ValueError("unsupported sample width: %r" % (sample_width,))

    frame_width = sample_width * channels
    n_frames = len(data) // frame_width
    if n_frames == 0:
        zeros = (0.0,) * channels
        return TruePeakMeasurement(zeros, zeros)
    data = data[:n_frames * frame_width]

    if frame_rate and frame_rate > 0:
        chunk_frames = max(1, int(round(frame_rate * chunk_ms / 1000.0)))
    else:
        chunk_frames = n_frames

    sample_peaks = []
    true_peaks = []
    for channel in range(channels):
        samples = _channel_samples(data, sample_width, channels, channel)
        sample_peak = 0.0
        for v in samples:
            if v > sample_peak:
                sample_peak = v
            elif -v > sample_peak:
                sample_peak = -v
        sample_peaks.append(sample_peak)
        true_peaks.append(_measure_channel(samples, chunk_frames))

    return TruePeakMeasurement(sample_peaks, true_peaks)
