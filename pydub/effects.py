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


# Automatic mains hum removal.
#
# detect_hum() finds stable narrow-band peaks at the mains candidates
# (50/60 Hz) and their low-order harmonics, and remove_hum() applies
# adjustable-strength notch filters at the detected frequencies. The
# HumFilter class does the actual filtering and keeps its state per
# channel, so a long recording processed in consecutive chunks gives
# exactly the same result as processing it in one pass.


def _goertzel_power(samples, sample_rate, freq):
    """
    Power of the spectral component at ``freq`` (Hz) in ``samples`` (an
    iterable of floats), computed with the Goertzel algorithm. For a pure
    sine of amplitude A the result is approximately (N * A / 2) ** 2.
    """
    n = len(samples)
    if n == 0:
        return 0.0
    omega = 2.0 * math.pi * freq / sample_rate
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return max(power, 0.0)


@register_pydub_effect
def detect_hum(seg, candidates=(50.0, 60.0), max_harmonic=5,
               min_prominence_db=8.0, min_level_dbfs=-50.0,
               max_analysis_ms=2000, min_analysis_ms=200):
    """
    Detect mains hum and return a sorted list of the frequencies (in Hz)
    that should be notched out: the winning candidate fundamental plus
    whichever of its low-order harmonics are also present as stable
    narrow-band peaks. Returns an empty list when no hum is found (or
    when detection is not possible, e.g. the segment is too short, is
    silent, or its frame rate is too low to represent the candidates).

    candidates - mains frequencies to test (default: 50 and 60 Hz). At
        most one candidate family is returned: when more than one
        candidate qualifies, the one with the stronger peaks wins.

    max_harmonic - highest harmonic of each candidate to consider
        (default: 5, i.e. up to 250/300 Hz).

    min_prominence_db - how far a peak must stand above the spectrum at
        neighboring frequencies (in each half of the analysis window) to
        count as a stable narrow-band peak.

    min_level_dbfs - peaks quieter than this are ignored, so silence and
        very quiet noise floors never trigger detection.

    max_analysis_ms / min_analysis_ms - bounds on how much audio is
        analyzed. Segments shorter than min_analysis_ms cannot be
        analyzed reliably and always result in an empty list.
    """
    if max_harmonic < 1:
        raise ValueError("max_harmonic should be at least 1")
    if len(seg) < min_analysis_ms:
        return []

    frame_rate = seg.frame_rate
    nyquist = frame_rate / 2.0
    candidates = [float(f) for f in candidates if 0.0 < f < nyquist * 0.95]
    if not candidates:
        return []

    mono = seg.set_channels(1) if seg.channels > 1 else seg
    if len(mono) > max_analysis_ms:
        mono = mono[:max_analysis_ms]

    max_amp = float(mono.max_possible_amplitude)
    samples = [sample / max_amp for sample in mono.get_array_of_samples()]

    # A stable hum peaks in both halves of the analysis window; transient
    # tones (beeps, speech formants sweeping by) generally do not.
    half = len(samples) // 2
    windows = (samples[:half], samples[half:])
    if not windows[0] or not windows[1]:
        return []

    window_seconds = half / float(frame_rate)
    neighbor_offset = max(2.0, 2.0 / window_seconds)

    def peak_stats(window, freq):
        n = len(window)
        peak = _goertzel_power(window, frame_rate, freq)
        amplitude = 2.0 * math.sqrt(peak) / n if peak > 0.0 else 0.0
        level_dbfs = (20.0 * math.log10(amplitude)
                      if amplitude > 0.0 else float('-inf'))
        floor = 0.0
        for delta in (-neighbor_offset, neighbor_offset):
            neighbor = freq + delta
            if 0.0 < neighbor < nyquist * 0.98:
                floor = max(floor, _goertzel_power(window, frame_rate, neighbor))
        if floor <= 0.0:
            prominence_db = float('inf') if peak > 0.0 else 0.0
        elif peak <= 0.0:
            prominence_db = 0.0
        else:
            prominence_db = 10.0 * math.log10(peak / floor)
        return level_dbfs, prominence_db

    def stable_peak(freq):
        # (is_stable, level_dbfs) - prominent and loud enough in *both*
        # halves of the analysis window
        levels = []
        for window in windows:
            level_dbfs, prominence_db = peak_stats(window, freq)
            if level_dbfs < min_level_dbfs or prominence_db < min_prominence_db:
                return False, float('-inf')
            levels.append(level_dbfs)
        return True, min(levels)

    best = None  # (score, fundamental, [stable frequencies])
    for fundamental in candidates:
        harmonics = []
        for k in xrange(1, max_harmonic + 1):
            harmonic = fundamental * k
            if harmonic >= nyquist * 0.95:
                break
            harmonics.append(harmonic)
        if not harmonics:
            continue

        # the fundamental itself must be a stable peak - otherwise this
        # is not mains hum, whatever the harmonics look like
        fundamental_stable, fundamental_level = stable_peak(harmonics[0])
        if not fundamental_stable:
            continue

        # score by hum loudness so that when more than one candidate
        # qualifies (e.g. tones near both 50 and 60 Hz) the stronger -
        # and therefore more audible - hum family wins
        score = fundamental_level
        stable = [harmonics[0]]
        for harmonic in harmonics[1:]:
            harmonic_stable, harmonic_level = stable_peak(harmonic)
            if harmonic_stable:
                stable.append(harmonic)
                score += harmonic_level

        if best is None or score > best[0]:
            best = (score, fundamental, stable)

    if best is None:
        return []
    return sorted(best[2])


class HumFilter(object):
    """
    Stateful narrow-band (notch) filter removing mains hum frequencies.

    Filter state is kept per channel, so processing a long recording in
    consecutive chunks gives exactly the same result as processing it in
    one pass:

        hum_filter = HumFilter([50.0, 150.0], frame_rate=44100, channels=2,
                               sample_width=2)
        cleaned_chunks = [hum_filter.process(chunk)
                          for chunk in make_chunks(seg, 1000)]

    strength - 0.0 leaves the audio untouched, 1.0 applies the full
        notch. Values in between blend the filtered signal with the dry
        one. Values outside [0.0, 1.0] raise ValueError.

    q - quality factor of the notch filters; higher means narrower
        (default: 30.0, a ~2 Hz wide notch at 50/60 Hz).
    """

    def __init__(self, frequencies, frame_rate, channels=1, sample_width=2,
                 strength=1.0, q=30.0):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength should be between 0.0 and 1.0")
        if q <= 0.0:
            raise ValueError("q should be greater than 0")
        if frame_rate <= 0:
            raise ValueError("frame_rate should be greater than 0")

        nyquist = frame_rate / 2.0
        self.frame_rate = frame_rate
        self.channels = channels
        self.sample_width = sample_width
        self.strength = float(strength)
        self.q = float(q)
        self.frequencies = []
        self._filters = []
        for freq in frequencies:
            if not 0.0 < freq < nyquist:
                raise ValueError(
                    "hum frequency %r Hz is outside the representable "
                    "range (0, %s)" % (freq, nyquist))
            omega = 2.0 * math.pi * freq / frame_rate
            cos_omega = math.cos(omega)
            alpha = math.sin(omega) / (2.0 * self.q)
            a0 = 1.0 + alpha
            # RBJ audio EQ cookbook notch (band-reject) biquad
            self._filters.append((
                1.0 / a0,
                -2.0 * cos_omega / a0,
                1.0 / a0,
                -2.0 * cos_omega / a0,
                (1.0 - alpha) / a0,
            ))
            self.frequencies.append(float(freq))

        # one [x1, x2, y1, y2] state per filter stage, per channel
        self._states = [[[0.0, 0.0, 0.0, 0.0] for _ in self._filters]
                        for _ in range(channels)]
        self._minval, self._maxval = get_min_max_value(sample_width * 8)

    def process(self, seg):
        """
        Filter one AudioSegment and return the result. The segment must
        match the frame rate, channel count and sample width the filter
        was created with. May be called repeatedly with consecutive
        chunks of a longer recording.
        """
        if seg.frame_rate != self.frame_rate:
            raise ValueError("segment frame rate %r does not match the "
                             "filter's %r" % (seg.frame_rate, self.frame_rate))
        if seg.channels != self.channels:
            raise ValueError("segment has %r channels, the filter expects "
                             "%r" % (seg.channels, self.channels))
        if seg.sample_width != self.sample_width:
            raise ValueError("segment sample width %r does not match the "
                             "filter's %r" % (seg.sample_width,
                                              self.sample_width))

        original = seg.get_array_of_samples()
        if not self._filters or self.strength == 0.0 or len(seg) == 0:
            return seg._spawn(data=array.array(seg.array_type, original))

        out = array.array(seg.array_type, original)
        strength = self.strength
        dry_mix = 1.0 - strength
        filters = self._filters
        states = self._states
        channels = self.channels
        minval = self._minval
        maxval = self._maxval
        n_stages = len(filters)

        for i in xrange(int(seg.frame_count())):
            base = i * channels
            for j in range(channels):
                index = base + j
                x = original[index]
                y = float(x)
                channel_states = states[j]
                for stage in range(n_stages):
                    b0, b1, b2, a1, a2 = filters[stage]
                    st = channel_states[stage]
                    y_new = (b0 * y + b1 * st[0] + b2 * st[1]
                             - a1 * st[2] - a2 * st[3])
                    st[1] = st[0]
                    st[0] = y
                    st[3] = st[2]
                    st[2] = y_new
                    y = y_new
                value = int(round(dry_mix * x + strength * y))
                if value < minval:
                    value = minval
                elif value > maxval:
                    value = maxval
                out[index] = value

        return seg._spawn(data=out)


@register_pydub_effect
def remove_hum(seg, strength=1.0, candidates=(50.0, 60.0), max_harmonic=5,
               q=30.0, **detect_kwargs):
    """
    Detect mains hum (see detect_hum()) and notch it out.

    strength - how strongly the detected hum frequencies are suppressed:
        0.0 leaves the audio untouched, 1.0 (default) applies the full
        notch. Values outside [0.0, 1.0] raise ValueError.

    candidates, max_harmonic and any additional keyword arguments are
    passed on to detect_hum(); q controls the width of the notch filters
    (see HumFilter).

    When no hum is detected (or strength is 0.0) the audio is returned
    with its PCM data unchanged.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength should be between 0.0 (no change) and "
                         "1.0 (full suppression)")
    if strength == 0.0 or len(seg) == 0:
        return seg._spawn(data=seg._data)

    frequencies = detect_hum(seg, candidates=candidates,
                             max_harmonic=max_harmonic, **detect_kwargs)
    if not frequencies:
        return seg._spawn(data=seg._data)

    hum_filter = HumFilter(frequencies, frame_rate=seg.frame_rate,
                           channels=seg.channels,
                           sample_width=seg.sample_width,
                           strength=strength, q=q)
    return hum_filter.process(seg)
