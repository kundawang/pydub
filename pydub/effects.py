import sys
import math
import array
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


def _find_clipped_runs(chan, clip_level):
    """
    Return a list of (start, end) index pairs (end exclusive) of maximal
    runs of samples whose absolute value reaches ``clip_level``.
    """
    runs = []
    n = len(chan)
    i = 0
    while i < n:
        sample = chan[i]
        if sample >= clip_level or sample <= -clip_level:
            start = i
            while i < n and (chan[i] >= clip_level or chan[i] <= -clip_level):
                i += 1
            runs.append((start, i))
        else:
            i += 1
    return runs


def _interpolate_run(chan, start, end, min_val, max_val):
    """
    Estimate the ``end - start`` clipped samples of one channel using a
    cubic Hermite spline anchored on the unclipped samples on either side
    of the run (with tangents from finite differences of the surrounding
    unclipped samples). Results are clamped to the sample format's range
    so repaired peaks can never exceed full scale.
    """
    run = end - start
    interval = run + 1  # distance (in samples) between the two anchors
    n = len(chan)

    p0 = chan[start - 1]
    p1 = chan[end]

    # tangents: per-sample slope at each anchor, scaled to the interval
    if start >= 2:
        m0 = (chan[start - 1] - chan[start - 2]) * interval
    else:
        m0 = p1 - p0
    if end + 1 < n:
        m1 = (chan[end + 1] - chan[end]) * interval
    else:
        m1 = p1 - p0

    repaired = []
    for i in range(1, run + 1):
        t = i / interval
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        value = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        value = int(round(value))
        if value > max_val:
            value = max_val
        elif value < min_val:
            value = min_val
        repaired.append(value)
    return repaired


@register_pydub_effect
def declip(seg, threshold=0.999, max_clip_ms=15.0):
    """
    Repair clipped (flat-topped / "square waved") samples caused by
    recording past full scale.

    Each channel is scanned independently for contiguous runs of samples
    at (or beyond) ``threshold`` of full scale. Runs that are short enough
    (at most ``max_clip_ms`` milliseconds) and do not touch either end of
    the segment are replaced with values estimated by cubic Hermite
    interpolation between the unclipped samples surrounding the run.

    threshold - fraction of full scale (0, 1] at or above which samples
        are considered clipped (default 0.999)
    max_clip_ms - maximum length, in milliseconds, of a clipped run that
        will be repaired; longer runs are left untouched (default 15.0)

    Returns a tuple ``(repaired_segment, repairs)`` where ``repairs`` is a
    list of dicts, one per repaired run::

        {"channel": 0, "start_sample": 120, "end_sample": 128,
         "start_ms": 2.72, "end_ms": 2.90}

    (``end_sample``/``end_ms`` are exclusive). Unclipped samples are left
    byte-identical, and the duration, frame rate, sample width and channel
    count of the segment are unchanged.

    Clipped runs that are longer than ``max_clip_ms``, or that touch the
    beginning or end of the segment (e.g. a recording that starts or ends
    mid-clip, or a constant full-scale signal), cannot be interpolated
    reliably and are returned unmodified and unreported.
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("threshold must be a number in the range (0, 1]")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in the range (0, 1]")

    try:
        max_clip_ms = float(max_clip_ms)
    except (TypeError, ValueError):
        raise ValueError("max_clip_ms must be a positive number")
    if not max_clip_ms > 0.0:
        raise ValueError("max_clip_ms must be a positive number")

    min_val, max_val = get_min_max_value(seg.sample_width * 8)
    clip_level = threshold * max_val

    max_run = max(1, int(round(seg.frame_rate * max_clip_ms / 1000.0)))

    samples = seg.get_array_of_samples()
    out = array.array(samples.typecode, samples)

    n_channels = seg.channels
    n = len(samples) // n_channels

    repairs = []
    for channel_i in xrange(n_channels):
        chan = samples[channel_i::n_channels]
        for start, end in _find_clipped_runs(chan, clip_level):
            run = end - start
            # runs touching either end, or too long to estimate, are
            # left untouched
            if start == 0 or end == n or run > max_run:
                continue
            repaired = _interpolate_run(chan, start, end, min_val, max_val)
            for offset, value in enumerate(repaired):
                out[(start + offset) * n_channels + channel_i] = value
            repairs.append({
                "channel": channel_i,
                "start_sample": start,
                "end_sample": end,
                "start_ms": start * 1000.0 / seg.frame_rate,
                "end_ms": end * 1000.0 / seg.frame_rate,
            })

    return seg._spawn(out), repairs
