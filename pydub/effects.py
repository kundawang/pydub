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


def _require_numpy():
    try:
        import numpy
    except ImportError:
        raise ImportError(
            "pydub.effects.convolve requires numpy. "
            "Install it with: pip install numpy"
        )
    return numpy


def _float_to_samples(np, data, array_type, sample_width):
    """
    Convert float values expressed in integer PCM units back to packed
    integer PCM. Values outside the representable range are saturated
    (clipped). pydub stores PCM little-endian, as used by the WAV container.
    """
    max_amplitude = (1 << (sample_width * 8 - 1)) - 1
    min_amplitude = -(max_amplitude + 1)
    rounded = np.rint(data)
    clipped = np.clip(rounded, min_amplitude, max_amplitude)
    int_dtype = '<i%d' % sample_width
    return array.array(array_type, clipped.astype(int_dtype).tobytes())


def _fft_convolve(np, signal, impulse):
    """
    Full linear convolution of two 1-D arrays, computed with a single FFT of
    size n + m - 1. Both the one-shot and the block-based code paths call
    this exact function, which guarantees the two paths agree sample for
    sample (before integer quantization).
    """
    n = signal.shape[0] + impulse.shape[0] - 1
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    spectrum = np.fft.rfft(signal, n) * np.fft.rfft(impulse, n)
    return np.fft.irfft(spectrum, n)


def _block_fft_convolve(np, signal, impulse, block_frames):
    """
    Linear convolution of two 1-D arrays processed in fixed-size blocks,
    using overlap-add. Each block is convolved with the full impulse via
    `_fft_convolve`, so the assembled output is identical to the one-shot
    convolution apart from floating point round-off.
    """
    m = impulse.shape[0]
    if m == 0:
        return np.zeros(signal.shape[0], dtype=np.float64)

    block_frames = max(1, int(block_frames))
    output = np.zeros(signal.shape[0] + m - 1, dtype=np.float64)
    for start in xrange(0, signal.shape[0], block_frames):
        block = signal[start:start + block_frames]
        conv = _fft_convolve(np, block, impulse)
        output[start:start + conv.shape[0]] += conv
    return output


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


@register_pydub_effect
def convolve(seg, impulse_response, dry=1.0, wet=1.0,
             include_tail=False, chunk_duration=10000):
    """
    Convolve the audio segment with an impulse response, e.g. apply the
    reverb of a room recorded as an impulse response (IR).

    Arguments:

        impulse_response (AudioSegment):
            The impulse response to convolve with. A mono impulse response is
            applied to every channel of the segment; otherwise the number of
            channels must match that of the segment. Its frame rate and
            sample width are converted to match the segment before
            processing, so the output keeps the segment's frame rate and
            sample width. The IR samples are interpreted relative to the
            full-scale level of the sample width: an IR whose first sample is
            at full scale and whose other samples are zero is therefore an
            exact identity, and an IR tap at half full scale applies a 0.5
            gain (no automatic gain normalization is performed).

        dry (float):
            Gain of the unprocessed (dry) signal, between 0.0 and 1.0. The
            default of 1.0 keeps the original signal at full level.

        wet (float):
            Gain of the convolved (wet) signal, between 0.0 and 1.0. The
            default of 1.0 mixes the full convolved signal in.

        include_tail (bool):
            When True (the default is False), the output keeps the reverb
            tail and is (segment + impulse response - 1 frame) long. When
            False, the output has exactly the same length as the input
            segment, i.e. the part of the convolution ringing past the end
            of the input is discarded.

        chunk_duration (int):
            Block size in milliseconds used for the block-based (overlap-add)
            convolution of long signals. Signals that fit in one block are
            convolved in a single FFT. The block-based result and the
            one-shot result are computed with the same FFT routine and agree
            to within integer quantization error.

    Behavior and rules:

        * An empty impulse response (zero frames) raises ValueError.
        * An impulse response with a channel count that is neither 1 nor the
          segment's channel count raises ValueError.
        * dry and wet must be finite numbers in the range 0.0 - 1.0; anything
          else (including NaN and infinity) raises ValueError. Negative gains
          or gains above 1.0 are rejected rather than silently clipping.
        * The output keeps the segment's frame rate, sample width and channel
          count. Internally the math is done in 64-bit float; the convolution
          is quantized to integer PCM units before the dry/wet mix, so the
          one-shot and block-based paths produce identical samples and a
          full-scale unit impulse is an exact identity. On return the mixed
          samples are rounded and saturated to the segment's integer format,
          so an overloaded mix clips instead of wrapping.
        * Convolving an all-silent segment yields an all-silent result.
        * An empty (zero-length) segment is returned unchanged.

    Requires numpy.
    """
    np = _require_numpy()

    if not isinstance(impulse_response, seg.__class__):
        raise TypeError(
            "impulse_response must be an AudioSegment instance, got {0}".format(
                type(impulse_response).__name__))

    for name, value in (("dry", dry), ("wet", wet)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "{0} must be a number between 0.0 and 1.0".format(name))
        if math.isnan(value) or math.isinf(value) \
                or not 0.0 <= value <= 1.0:
            raise ValueError(
                "{0} must be a finite number between 0.0 and 1.0, got {1}".format(
                    name, value))

    if int(impulse_response.frame_count()) == 0:
        raise ValueError(
            "impulse_response must contain at least one frame")

    if impulse_response.channels not in (1, seg.channels):
        raise ValueError(
            "impulse_response has {0} channel(s); it must be mono (1 "
            "channel, applied to every channel) or match the segment's "
            "channel count ({1})".format(
                impulse_response.channels, seg.channels))

    if int(seg.frame_count()) == 0:
        return seg

    # Match the IR's frame rate and sample width to the segment so that the
    # convolution indices and the output format are well defined.
    ir = (impulse_response
          .set_frame_rate(seg.frame_rate)
          .set_sample_width(seg.sample_width))

    input_frames = int(seg.frame_count())
    ir_frames = int(ir.frame_count())
    output_frames = input_frames + ir_frames - 1 if include_tail \
        else input_frames

    # deinterleave by hand: split_to_mono()'s spawned segments can keep the
    # parent's channels/frame_width metadata, but the raw samples are simply
    # interleaved frame by frame, so derive the real channel count from the
    # data length and slice directly.
    # The signal stays in integer PCM units while the impulse response is
    # normalized to the full-scale level (2 ** (bits - 1)), so an IR whose
    # peak reaches full scale has unity gain: a full-scale unit impulse
    # ([full_scale, 0, 0, ...]) is an exact identity, and an IR tap of half
    # full scale is a 0.5 gain. The convolution is evaluated in float64.
    full_scale = float(1 << (seg.sample_width * 8 - 1))
    input_raw = np.asarray(seg.get_array_of_samples(), dtype=np.float64)
    ir_raw = np.asarray(ir.get_array_of_samples(), dtype=np.float64) / full_scale
    input_channel_count = input_raw.shape[0] // input_frames
    ir_channel_count = ir_raw.shape[0] // ir_frames
    input_channels = [
        input_raw[channel_i::input_channel_count]
        for channel_i in xrange(input_channel_count)
    ]
    ir_channels = [
        ir_raw[channel_i::ir_channel_count]
        for channel_i in xrange(ir_channel_count)
    ]

    block_frames = max(1, int(seg.frame_count(ms=chunk_duration)))

    output_channels = []
    for channel_i in xrange(input_channel_count):
        signal = input_channels[channel_i]
        # A mono IR is reused for every channel; otherwise channel i of the
        # segment is convolved with channel i of the IR.
        impulse = ir_channels[0] if ir_channel_count == 1 \
            else ir_channels[channel_i]

        if input_frames <= block_frames:
            convolved = _fft_convolve(np, signal, impulse)
        else:
            convolved = _block_fft_convolve(
                np, signal, impulse, block_frames)

        # Quantize the convolution itself to integer PCM units before the
        # dry/wet mix: FFT round-off is removed here, a unit impulse becomes
        # an exact identity, and the one-shot and block paths stay identical.
        # Clipping is deferred to the final mix so the wet signal alone is
        # allowed to exceed full scale.
        convolved = np.rint(convolved)

        # pad the dry signal with silence when the reverb tail is kept
        dry_signal = signal
        if output_frames > input_frames:
            dry_signal = np.concatenate(
                (signal, np.zeros(output_frames - input_frames)))

        mixed = dry * dry_signal[:output_frames] \
            + wet * convolved[:output_frames]

        output_channels.append(
            _float_to_samples(np, mixed, seg.array_type, seg.sample_width))

    if input_channel_count == 1:
        out_data = output_channels[0]
    else:
        interleaved = array.array(
            seg.array_type,
            b'\0' * (output_frames * input_channel_count * seg.sample_width))
        for channel_i, channel_data in enumerate(output_channels):
            interleaved[channel_i::input_channel_count] = channel_data
        out_data = interleaved

    return seg._spawn(data=out_data)
