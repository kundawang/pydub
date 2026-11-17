import sys
import math
import array
import warnings
from .utils import (
    db_to_float,
    ratio_to_db,
    register_pydub_effect,
    make_chunks,
    audioop,
    get_min_max_value
)
from .silence import split_on_silence
from .exceptions import TooManyMissingFrames, InvalidDuration

if sys.version_info >= (3, 0):
    xrange = range


@register_pydub_effect
def apply_mono_filter_to_each_channel(seg, filter_fn):
    n_channels = seg.channels

    channel_segs = seg.split_to_mono()
    channel_segs = [filter_fn(channel_seg) for channel_seg in channel_segs]

    out_data = seg.get_array_of_samples()
    for channel_i, channel_seg in enumerate(channel_segs):
        for sample_i, sample in enumerate(channel_seg.get_array_of_samples()):
            index = (sample_i * n_channels) + channel_i
            out_data[index] = sample

    return seg._spawn(out_data)


@register_pydub_effect
def normalize(seg, headroom=0.1):
    """
    headroom is how close to the maximum volume to boost the signal up to (specified in dB)
    """
    peak_sample_val = seg.max
    
    # if the max is 0, this audio segment is silent, and can't be normalized
    if peak_sample_val == 0:
        return seg
    
    target_peak = seg.max_possible_amplitude * db_to_float(-headroom)

    needed_boost = ratio_to_db(target_peak / peak_sample_val)
    return seg.apply_gain(needed_boost)


@register_pydub_effect
def speedup(seg, playback_speed=1.5, chunk_size=150, crossfade=25):
    # we will keep audio in 150ms chunks since one waveform at 20Hz is 50ms long
    # (20 Hz is the lowest frequency audible to humans)

    # portion of AUDIO TO KEEP. if playback speed is 1.25 we keep 80% (0.8) and
    # discard 20% (0.2)
    atk = 1.0 / playback_speed

    if playback_speed < 2.0:
        # throwing out more than half the audio - keep 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size * (1 - atk) / atk)
    else:
        # throwing out less than half the audio - throw out 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size)
        chunk_size = int(atk * chunk_size / (1 - atk))

    # the crossfade cannot be longer than the amount of audio we're removing
    crossfade = min(crossfade, ms_to_remove_per_chunk - 1)

    # DEBUG
    #print("chunk: {0}, rm: {1}".format(chunk_size, ms_to_remove_per_chunk))

    chunks = make_chunks(seg, chunk_size + ms_to_remove_per_chunk)
    if len(chunks) < 2:
        raise Exception("Could not speed up AudioSegment, it was too short {2:0.2f}s for the current settings:\n{0}ms chunks at {1:0.1f}x speedup".format(
            chunk_size, playback_speed, seg.duration_seconds))

    # we'll actually truncate a bit less than we calculated to make up for the
    # crossfade between chunks
    ms_to_remove_per_chunk -= crossfade

    # we don't want to truncate the last chunk since it is not guaranteed to be
    # the full chunk length
    last_chunk = chunks[-1]
    chunks = [chunk[:-ms_to_remove_per_chunk] for chunk in chunks[:-1]]

    out = chunks[0]
    for chunk in chunks[1:]:
        out = out.append(chunk, crossfade=crossfade)

    out += last_chunk
    return out
    

@register_pydub_effect
def strip_silence(seg, silence_len=1000, silence_thresh=-16, padding=100):
    if padding > silence_len:
        raise InvalidDuration("padding cannot be longer than silence_len")

    chunks = split_on_silence(seg, silence_len, silence_thresh, padding)
    crossfade = padding / 2

    if not len(chunks):
        return seg[0:0]

    seg = chunks[0]
    for chunk in chunks[1:]:
        seg = seg.append(chunk, crossfade=crossfade)

    return seg


@register_pydub_effect
def compress_dynamic_range(seg, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0):
    """
    Keyword Arguments:
        
        threshold - default: -20.0
            Threshold in dBFS. default of -20.0 means -20dB relative to the
            maximum possible volume. 0dBFS is the maximum possible value so
            all values for this argument sould be negative.

        ratio - default: 4.0
            Compression ratio. Audio louder than the threshold will be 
            reduced to 1/ratio the volume. A ratio of 4.0 is equivalent to
            a setting of 4:1 in a pro-audio compressor like the Waves C1.
        
        attack - default: 5.0
            Attack in milliseconds. How long it should take for the compressor
            to kick in once the audio has exceeded the threshold.

        release - default: 50.0
            Release in milliseconds. How long it should take for the compressor
            to stop compressing after the audio has falled below the threshold.

    
    For an overview of Dynamic Range Compression, and more detailed explanation
    of the related terminology, see: 

        http://en.wikipedia.org/wiki/Dynamic_range_compression
    """

    thresh_rms = seg.max_possible_amplitude * db_to_float(threshold)
    
    look_frames = int(seg.frame_count(ms=attack))
    def rms_at(frame_i):
        return seg.get_sample_slice(frame_i - look_frames, frame_i).rms
    def db_over_threshold(rms):
        if rms == 0: return 0.0
        db = ratio_to_db(rms / thresh_rms)
        return max(db, 0)

    output = []

    # amount to reduce the volume of the audio by (in dB)
    attenuation = 0.0
    
    attack_frames = seg.frame_count(ms=attack)
    release_frames = seg.frame_count(ms=release)
    for i in xrange(int(seg.frame_count())):
        rms_now = rms_at(i)
        
        # with a ratio of 4.0 this means the volume will exceed the threshold by
        # 1/4 the amount (of dB) that it would otherwise
        max_attenuation = (1 - (1.0 / ratio)) * db_over_threshold(rms_now)
        
        attenuation_inc = max_attenuation / attack_frames
        attenuation_dec = max_attenuation / release_frames
        
        if rms_now > thresh_rms and attenuation <= max_attenuation:
            attenuation += attenuation_inc
            attenuation = min(attenuation, max_attenuation)
        else:
            attenuation -= attenuation_dec
            attenuation = max(attenuation, 0)
        
        frame = seg.get_frame(i)
        if attenuation != 0.0:
            frame = audioop.mul(frame,
                                seg.sample_width,
                                db_to_float(-attenuation))
        
        output.append(frame)
    
    return seg._spawn(data=b''.join(output))


def _import_numpy():
    try:
        import numpy
        return numpy
    except ImportError:
        raise ImportError(
            "multiband_compress requires numpy. "
            "Install it with: pip install numpy"
        )


def _validate_band_params(name, values, check):
    try:
        values = tuple(values)
    except TypeError:
        raise ValueError(
            "%s must be a sequence of 3 values (low, mid, high), got %r"
            % (name, values))
    if len(values) != 3:
        raise ValueError(
            "%s must contain exactly 3 values (low, mid, high), got %d"
            % (name, len(values)))
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("%s values must be finite numbers, got %r"
                             % (name, v))
        check(v)
    return tuple(float(v) for v in values)


def _check_threshold(v):
    if v > 0:
        raise ValueError("thresholds values must be <= 0 dBFS, got %r" % (v,))


def _check_ratio(v):
    if v < 1.0:
        raise ValueError("ratios values must be >= 1.0, got %r" % (v,))


def _check_positive_time(v):
    if v <= 0:
        raise ValueError("attacks/releases values must be > 0, got %r" % (v,))


def _design_lowpass_taps(np, cutoff, frame_rate, numtaps):
    """Windowed-sinc lowpass FIR (linear phase, constant group delay)."""
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    fc = cutoff / float(frame_rate)
    h = 2.0 * fc * np.sinc(2.0 * fc * n)
    h *= np.hamming(numtaps)
    h /= h.sum()
    return h


def _compress_band_envelope(band, threshold, ratio, attack_coef,
                            release_coef, env):
    """
    Per-sample envelope follower + gain computer for one band of one block.

    `env` is the envelope state carried across blocks (and returned updated),
    so the detector is seamless no matter how the audio is chunked.
    """
    inv_ratio_term = 1.0 - (1.0 / ratio)
    gains = [1.0] * len(band)
    for i, sample in enumerate(band):
        level = sample if sample >= 0 else -sample
        coef = attack_coef if level > env else release_coef
        env = coef * env + (1.0 - coef) * level
        if env > 1e-10:
            over_db = 20.0 * math.log10(env) - threshold
            if over_db > 0.0:
                gains[i] = db_to_float(-over_db * inv_ratio_term)
    return gains, env


@register_pydub_effect
def multiband_compress(seg, low_crossover=200.0, high_crossover=2000.0,
                       thresholds=(-20.0, -20.0, -20.0),
                       ratios=(4.0, 4.0, 4.0),
                       attacks=(5.0, 5.0, 5.0),
                       releases=(50.0, 50.0, 50.0),
                       block_ms=10.0, numtaps=None):
    """
    Three-band dynamic range compressor.

    Splits the signal into low / mid / high bands with linear-phase FIR
    crossover filters, compresses each band independently, and recombines
    them delay-aligned, so with compression disabled (ratio 1:1 on every
    band) the output is (nearly) sample-identical to the input.

    Keyword Arguments:

        low_crossover - default: 200.0
            Crossover frequency in Hz between the low and mid bands. Must be
            > 0 and < high_crossover.

        high_crossover - default: 2000.0
            Crossover frequency in Hz between the mid and high bands. Must be
            > low_crossover and < frame_rate / 2.

        thresholds - default: (-20.0, -20.0, -20.0)
            Threshold in dBFS for the (low, mid, high) bands. Must be <= 0.

        ratios - default: (4.0, 4.0, 4.0)
            Compression ratio for the (low, mid, high) bands. 1.0 disables
            compression for that band. Must be >= 1.0.

        attacks - default: (5.0, 5.0, 5.0)
            Attack time in milliseconds for the (low, mid, high) bands.

        releases - default: (50.0, 50.0, 50.0)
            Release time in milliseconds for the (low, mid, high) bands.

        block_ms - default: 10.0
            Internal processing block size in milliseconds. Filter and
            envelope state is carried across blocks, so the result does not
            depend on this value; it only tunes memory/latency trade-offs.

        numtaps - default: None (chosen from the crossover frequencies)
            Length of the crossover FIR filters (forced odd). Longer filters
            give steeper band separation but require longer input.

    The returned AudioSegment has the same frame rate, channel count, sample
    width and duration as the input. Requires numpy.
    """
    np = _import_numpy()

    if seg.sample_width not in (1, 2, 4):
        raise ValueError(
            "multiband_compress only supports 8, 16 and 32 bit audio "
            "(sample_width 1, 2 or 4), got sample_width=%r"
            % (seg.sample_width,))

    nyquist = seg.frame_rate / 2.0
    for name, freq in (("low_crossover", low_crossover),
                       ("high_crossover", high_crossover)):
        if not isinstance(freq, (int, float)) or not math.isfinite(freq):
            raise ValueError("%s must be a finite number, got %r"
                             % (name, freq))
    if not 0.0 < low_crossover < high_crossover < nyquist:
        raise ValueError(
            "crossover frequencies must satisfy "
            "0 < low_crossover < high_crossover < frame_rate/2 "
            "(got low=%r, high=%r, frame_rate=%r)"
            % (low_crossover, high_crossover, seg.frame_rate))

    thresholds = _validate_band_params("thresholds", thresholds,
                                       _check_threshold)
    ratios = _validate_band_params("ratios", ratios, _check_ratio)
    attacks = _validate_band_params("attacks", attacks, _check_positive_time)
    releases = _validate_band_params("releases", releases,
                                     _check_positive_time)

    if not isinstance(block_ms, (int, float)) or not math.isfinite(block_ms) \
            or block_ms <= 0:
        raise ValueError("block_ms must be a finite number > 0, got %r"
                         % (block_ms,))

    # silence in -> silence out (also avoids log-of-zero in the detector)
    if seg.max == 0:
        return seg._spawn(seg.get_array_of_samples())

    frame_rate = seg.frame_rate

    if numtaps is None:
        transition = min(low_crossover,
                         high_crossover - low_crossover,
                         nyquist - high_crossover)
        # ~3.3 * frame_rate / transition covers the Hamming main lobe
        numtaps = int(math.ceil(3.3 * frame_rate / transition))
        numtaps = max(63, min(numtaps, 8191))
    numtaps = int(numtaps)
    if numtaps < 3:
        raise ValueError("numtaps must be >= 3, got %r" % (numtaps,))
    if numtaps % 2 == 0:
        numtaps += 1

    if seg.frame_count() < numtaps:
        raise InvalidDuration(
            "segment is too short for multiband compression: %d frames "
            "available, but the crossover filters require at least %d "
            "frames (use a longer segment, higher crossover frequencies, "
            "or fewer numtaps)" % (int(seg.frame_count()), numtaps))

    lp_low = _design_lowpass_taps(np, low_crossover, frame_rate, numtaps)
    lp_mid = _design_lowpass_taps(np, high_crossover, frame_rate, numtaps)
    delay = (numtaps - 1) // 2  # group delay of each crossover filter

    block_len = max(1, int(round(frame_rate * block_ms / 1000.0)))

    band_params = []
    for threshold, ratio, attack, release in zip(thresholds, ratios,
                                                 attacks, releases):
        attack_coef = math.exp(-1.0 / (attack * frame_rate / 1000.0))
        release_coef = math.exp(-1.0 / (release * frame_rate / 1000.0))
        band_params.append((threshold, ratio, attack_coef, release_coef))

    max_amp = float(seg.max_possible_amplitude)
    min_val, max_val = get_min_max_value(seg.sample_width * 8)
    clipped = [False]

    def process_channel(mono_seg):
        x = np.frombuffer(mono_seg._data,
                          dtype=np.dtype(mono_seg.array_type)
                          ).astype(np.float64)
        x /= max_amp

        # pad with 2*delay zeros to flush the crossover filter pipeline so
        # every input sample gets a fully-processed output sample
        x = np.concatenate([x, np.zeros(2 * delay)])

        filt_hist = numtaps - 1
        x_hist = np.zeros(filt_hist)
        rest_hist = np.zeros(filt_hist)
        low_delay_hist = np.zeros(delay)
        envelopes = [0.0, 0.0, 0.0]

        out = np.empty(len(x))
        pos = 0
        while pos < len(x):
            block = x[pos:pos + block_len]
            size = len(block)

            # stage 1: low band + complementary remainder, both aligned
            # to input delayed by `delay`
            x_ext = np.concatenate([x_hist, block])
            low = np.convolve(x_ext, lp_low, mode='valid')
            x_hist = x_ext[-filt_hist:]
            rest = x_ext[delay:delay + size] - low

            # stage 2: split the remainder into mid + high, aligned to
            # input delayed by 2*delay
            rest_ext = np.concatenate([rest_hist, rest])
            mid = np.convolve(rest_ext, lp_mid, mode='valid')
            rest_hist = rest_ext[-filt_hist:]
            high = rest_ext[delay:delay + size] - mid

            # delay-align the low band with the mid/high bands
            low_ext = np.concatenate([low_delay_hist, low])
            low_aligned = low_ext[:size]
            low_delay_hist = low_ext[size:]

            mixed = np.zeros(size)
            for band_i, band in enumerate((low_aligned, mid, high)):
                gains, envelopes[band_i] = _compress_band_envelope(
                    band, *band_params[band_i], envelopes[band_i])
                mixed += band * np.asarray(gains)

            out[pos:pos + size] = mixed
            pos += size

        # drop the 2*delay of pipeline latency so output lines up with input
        out = out[2 * delay:]

        int_samples = out * max_amp
        if np.any(int_samples > max_val) or np.any(int_samples < min_val):
            clipped[0] = True
            int_samples = np.clip(int_samples, min_val, max_val)
        return np.rint(int_samples).astype(np.dtype(mono_seg.array_type))

    mono_channels = seg.split_to_mono()
    processed = [process_channel(ch) for ch in mono_channels]

    if len(processed) == 1:
        out_data = processed[0]
    else:
        out_data = np.empty(len(processed[0]) * len(processed),
                            dtype=processed[0].dtype)
        for ch_i, ch_data in enumerate(processed):
            out_data[ch_i::len(processed)] = ch_data

    if clipped[0]:
        warnings.warn(
            "multiband_compress: output exceeded the maximum sample value "
            "and was clipped; consider lower thresholds/ratios or reducing "
            "the input gain", UserWarning)

    return seg._spawn(out_data.tobytes())


# Invert the phase of the signal.

@register_pydub_effect

def invert_phase(seg, channels=(1, 1)):
    """
    channels- specifies which channel (left or right) to reverse the phase of.
    Note that mono AudioSegments will become stereo.
    """
    if channels == (1, 1):
        inverted = audioop.mul(seg._data, seg.sample_width, -1.0)  
        return seg._spawn(data=inverted)
    
    else:
        if seg.channels == 2:
            left, right = seg.split_to_mono()
        else:
            raise Exception("Can't implicitly convert an AudioSegment with " + str(seg.channels) + " channels to stereo.")
            
        if channels == (1, 0):    
            left = left.invert_phase()
        else:
            right = right.invert_phase()
        
        return seg.from_mono_audiosegments(left, right)
        


# High and low pass filters based on implementation found on Stack Overflow:
#   http://stackoverflow.com/questions/13882038/implementing-simple-high-and-low-pass-filters-in-c

@register_pydub_effect
def low_pass_filter(seg, cutoff):
    """
        cutoff - Frequency (in Hz) where higher frequency signal will begin to
            be reduced by 6dB per octave (doubling in frequency) above this point
    """
    RC = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = dt / (RC + dt)
    
    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)
    
    frame_count = int(seg.frame_count())

    last_val = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filteredArray[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            last_val[j] = last_val[j] + (alpha * (original[offset] - last_val[j]))
            filteredArray[offset] = int(last_val[j])

    return seg._spawn(data=filteredArray)


@register_pydub_effect
def high_pass_filter(seg, cutoff):
    """
        cutoff - Frequency (in Hz) where lower frequency signal will begin to
            be reduced by 6dB per octave (doubling in frequency) below this point
    """
    RC = 1.0 / (cutoff * 2 * math.pi)
    dt = 1.0 / seg.frame_rate

    alpha = RC / (RC + dt)

    minval, maxval = get_min_max_value(seg.sample_width * 8)
    
    original = seg.get_array_of_samples()
    filteredArray = array.array(seg.array_type, original)
    
    frame_count = int(seg.frame_count())

    last_val = [0] * seg.channels
    for i in range(seg.channels):
        last_val[i] = filteredArray[i] = original[i]

    for i in range(1, frame_count):
        for j in range(seg.channels):
            offset = (i * seg.channels) + j
            offset_minus_1 = ((i-1) * seg.channels) + j

            last_val[j] = alpha * (last_val[j] + original[offset] - original[offset_minus_1])
            filteredArray[offset] = int(min(max(last_val[j], minval), maxval))

    return seg._spawn(data=filteredArray)
    
    
@register_pydub_effect
def pan(seg, pan_amount):
    """
    pan_amount should be between -1.0 (100% left) and +1.0 (100% right)
    
    When pan_amount == 0.0 the left/right balance is not changed.
    
    Panning does not alter the *perceived* loundness, but since loudness
    is decreasing on one side, the other side needs to get louder to
    compensate. When panned hard left, the left channel will be 3dB louder.
    """
    if not -1.0 <= pan_amount <= 1.0:
        raise ValueError("pan_amount should be between -1.0 (100% left) and +1.0 (100% right)")
    
    max_boost_db = ratio_to_db(2.0)
    boost_db = abs(pan_amount) * max_boost_db
    
    boost_factor = db_to_float(boost_db)
    reduce_factor = db_to_float(max_boost_db) - boost_factor
    
    reduce_db = ratio_to_db(reduce_factor)
    
    # Cut boost in half (max boost== 3dB) - in reality 2 speakers
    #   do not sum to a full 6 dB.
    boost_db = boost_db / 2.0
    
    if pan_amount < 0:
        return seg.apply_gain_stereo(boost_db, reduce_db)
    else:
        return seg.apply_gain_stereo(reduce_db, boost_db)
        
    
@register_pydub_effect
def apply_gain_stereo(seg, left_gain=0.0, right_gain=0.0):
    """
    left_gain - amount of gain to apply to the left channel (in dB)
    right_gain - amount of gain to apply to the right channel (in dB)
    
    note: mono audio segments will be converted to stereo
    """
    if seg.channels == 1:
        left = right = seg
    elif seg.channels == 2:
        left, right = seg.split_to_mono()
    
    l_mult_factor = db_to_float(left_gain)
    r_mult_factor = db_to_float(right_gain)
    
    left_data = audioop.mul(left._data, left.sample_width, l_mult_factor)
    left_data = audioop.tostereo(left_data, left.sample_width, 1, 0)
    
    right_data = audioop.mul(right._data, right.sample_width, r_mult_factor)
    right_data = audioop.tostereo(right_data, right.sample_width, 0, 1)
    
    output = audioop.add(left_data, right_data, seg.sample_width)
    
    return seg._spawn(data=output,
                overrides={'channels': 2,
                           'frame_width': 2 * seg.sample_width})
