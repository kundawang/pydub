"""
Various functions for finding/manipulating silence in AudioSegments
"""
import itertools
import math

from .exceptions import InvalidDuration, InvalidThreshold
from .utils import db_to_float, register_pydub_effect


def detect_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Returns a list of all silent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_nonsilent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    seg_len = len(audio_segment)

    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_silence_len:
        return []

    # convert silence threshold to a float value (so we can compare it to rms)
    silence_thresh = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude

    # find silence and add start and end indicies to the to_cut list
    silence_starts = []

    # check successive (1 sec by default) chunk of sound for silence
    # try a chunk at every "seek step" (or every chunk for a seek step == 1)
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in slice_starts:
        audio_slice = audio_segment[i:i + min_silence_len]
        if audio_slice.rms <= silence_thresh:
            silence_starts.append(i)

    # short circuit when there is no silence
    if not silence_starts:
        return []

    # combine the silence we detected into ranges (start ms - end ms)
    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start,
                                  prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start,
                          prev_i + min_silence_len])

    return silent_ranges


def detect_nonsilent(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Returns a list of all nonsilent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_silent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step)
    len_seg = len(audio_segment)

    # if there is no silence, the whole thing is nonsilent
    if not silent_ranges:
        return [[0, len_seg]]

    # short circuit when the whole audio segment is silent
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges


def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1):
    """
    Returns list of audio segments from splitting audio_segment on silent sections

    audio_segment - original pydub.AudioSegment() object

    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms

    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS

    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms

    seek_step - step size for interating over the segment in ms
    """

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [ start - keep_silence, end + keep_silence ]
        for (start,end)
            in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return [
        audio_segment[ max(start,0) : min(end,len(audio_segment)) ]
        for start,end in output_ranges
    ]


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    """
    Returns the millisecond/index that the leading silence ends.

    audio_segment - the segment to find silence in
    silence_threshold - the upper bound for how quiet is silent in dFBS
    chunk_size - chunk size for interating over the segment in ms
    """
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    # if there is no end it should return the length of the segment
    return min(trim_ms, len(sound))


def _validate_compress_silence_params(audio_segment, min_silence_len, silence_thresh,
                                      seek_step, keep_silence, crossfade):
    # the detection threshold is a dBFS value, so it must be a real number
    # and cannot be positive (0 dBFS is the maximum possible amplitude)
    if isinstance(silence_thresh, bool) or not isinstance(silence_thresh, (int, float)):
        raise InvalidThreshold(
            "silence_thresh must be a number in dBFS, got {0}".format(
                type(silence_thresh).__name__))
    try:
        not_a_number = math.isnan(silence_thresh)
        positive_infinity = math.isinf(silence_thresh) and silence_thresh > 0
    except (TypeError, OverflowError):
        raise InvalidThreshold("silence_thresh is not a valid dBFS value")
    if not_a_number or positive_infinity or silence_thresh > 0:
        raise InvalidThreshold(
            "silence_thresh must be a real dBFS value <= 0, got {0}".format(
                silence_thresh))

    if isinstance(min_silence_len, bool) or min_silence_len <= 0:
        raise InvalidDuration(
            "min_silence_len must be greater than zero, got {0}".format(min_silence_len))

    if isinstance(seek_step, bool) or seek_step <= 0:
        raise InvalidDuration(
            "seek_step must be greater than zero, got {0}".format(seek_step))

    if isinstance(keep_silence, bool) or keep_silence < 0:
        raise InvalidDuration(
            "keep_silence must be zero or positive, got {0}".format(keep_silence))

    if isinstance(crossfade, bool) or crossfade < 0:
        raise InvalidDuration(
            "crossfade must be zero or positive, got {0}".format(crossfade))


def compress_silence(audio_segment, min_silence_len=1000, silence_thresh=-16,
                     keep_silence=100, crossfade=10, seek_step=1):
    """
    Compress (shorten) the silent pauses in an AudioSegment while preserving
    a natural amount of breathing room around the speech.

    Only silences that are at least ``min_silence_len`` milliseconds long and
    lie *between* two non-silent sections are compressed. Silence at the very
    beginning or end of the segment is never removed, and silences shorter
    than ``min_silence_len`` are left untouched. Each compressed pause keeps
    ``keep_silence`` milliseconds of its original silence on either side.

    audio_segment - original pydub.AudioSegment() object

    min_silence_len - (in ms) minimum length of a silence to be compressed.
        default: 1000ms

    silence_thresh - (in dBFS) anything quieter than this will be considered
        silence. Must be a real number <= 0. default: -16dBFS

    keep_silence - (in ms) amount of original silence preserved on each side
        of every compressed pause. Pauses shorter than 2 * keep_silence are
        kept in full instead of being split. default: 100ms

    crossfade - (in ms) crossfade applied at each splice to avoid clicks. It
        is clamped to half the length of the shortest piece being joined, so
        a crossfade longer than the available audio never raises. Use 0 for a
        hard, sample-exact splice. default: 10ms

    seek_step - step size for searching the segment in ms. default: 1ms

    Returns a tuple ``(compressed_segment, mapping)`` where ``mapping`` is a
    list of ``(src_start, src_end, dst_start, dst_end)`` tuples (all in ms)
    covering the whole original timeline in order. Kept sections map to the
    audio they became; removed sections map to an empty destination
    (``dst_start == dst_end``). Adjacent kept sections overlap in the
    destination by the (possibly clamped) crossfade duration.
    """
    _validate_compress_silence_params(
        audio_segment, min_silence_len, silence_thresh, seek_step,
        keep_silence, crossfade)

    seg_len = len(audio_segment)

    # an empty or fully silent segment has nothing compressible: return the
    # data unchanged with an identity mapping
    nonsilent_ranges = detect_nonsilent(
        audio_segment, min_silence_len, silence_thresh, seek_step)
    if not nonsilent_ranges or len(nonsilent_ranges) < 2:
        return audio_segment, [(0, seg_len, 0, seg_len)]

    # Silence detection works in milliseconds and can return fractional
    # values, while audio data is made of whole frames. Do every cut in frame
    # space so no sample is lost (or padded) and the mapping matches the real
    # output sample-for-sample.
    frame_rate = audio_segment.frame_rate

    def ms_to_frames(ms_value):
        return int(audio_segment.frame_count(ms=ms_value))

    def frames_to_ms(frames_value):
        return int(round(1000.0 * frames_value / frame_rate))

    keep_frames = ms_to_frames(keep_silence)

    # the frame ranges of silence *between* non-silent sections that are long
    # enough to be compressed; leading/trailing silence is never cut
    removed_frame_ranges = []
    prev_nonsilent_end = ms_to_frames(nonsilent_ranges[0][1])
    for range_start, range_end in nonsilent_ranges[1:]:
        gap_start = prev_nonsilent_end
        gap_end = ms_to_frames(range_start)
        gap_frames = gap_end - gap_start
        if gap_frames > 2 * keep_frames:
            removed_frame_ranges.append(
                (gap_start + keep_frames, gap_end - keep_frames))
        prev_nonsilent_end = ms_to_frames(range_end)

    if not removed_frame_ranges:
        return audio_segment, [(0, seg_len, 0, seg_len)]

    total_frames = int(audio_segment.frame_count())

    # the complement of the removed ranges is the audio that survives,
    # expressed as frame sections of the original in their original order
    kept_frame_ranges = []
    cursor = 0
    for cut_start, cut_end in removed_frame_ranges:
        if cut_start > cursor:
            kept_frame_ranges.append((cursor, cut_start))
        cursor = cut_end
    if cursor < total_frames:
        kept_frame_ranges.append((cursor, total_frames))

    pieces = [audio_segment.get_sample_slice(start, end)
              for start, end in kept_frame_ranges]

    # join the surviving pieces, clamping the crossfade so it never exceeds
    # half the length of either piece participating in a splice
    crossfade_frames = ms_to_frames(crossfade)
    output = pieces[0]
    splice_fades_frames = []
    for piece in pieces[1:]:
        fade_frames = min(crossfade_frames,
                          int(output.frame_count()) // 2,
                          int(piece.frame_count()) // 2)
        splice_fades_frames.append(fade_frames)
        if fade_frames == 0:
            output = output.append(piece, crossfade=0)
        else:
            # mirror AudioSegment.append() but in frame space so the splice
            # has exactly fade_frames frames and matches the mapping
            output_frames = int(output.frame_count())
            head = output.get_sample_slice(0, output_frames - fade_frames)
            tail_out = output.get_sample_slice(output_frames - fade_frames)
            tail_in = piece.get_sample_slice(0, fade_frames)
            tail_ms = frames_to_ms(fade_frames)
            cross = tail_out.fade(
                to_gain=-120, start=0, end=tail_ms).overlay(
                tail_in.fade(from_gain=-120, start=0, end=tail_ms))
            rest = piece.get_sample_slice(fade_frames)
            output = head.append(cross, crossfade=0).append(rest, crossfade=0)

    # build the source-time -> new-time mapping (reported in milliseconds);
    # each splice makes two kept sections overlap by the crossfade duration
    mapping = []
    dst_cursor_frames = 0
    piece_end_frames = []
    for index, ((src_start, src_end), piece) in enumerate(
            zip(kept_frame_ranges, pieces)):
        dst_start = dst_cursor_frames
        dst_end = dst_start + int(piece.frame_count())
        mapping.append((frames_to_ms(src_start), frames_to_ms(src_end),
                        frames_to_ms(dst_start), frames_to_ms(dst_end)))
        piece_end_frames.append(dst_end)
        if index < len(pieces) - 1:
            dst_cursor_frames = dst_end - splice_fades_frames[index]

    # removed sections map to an empty destination located at the splice point
    for cut_index, (cut_start, cut_end) in enumerate(removed_frame_ranges):
        # the splice sits at the end of the piece preceding the cut
        splice_point = frames_to_ms(piece_end_frames[cut_index])
        mapping.append((frames_to_ms(cut_start), frames_to_ms(cut_end),
                        splice_point, splice_point))

    mapping.sort(key=lambda item: (item[0], item[1]))

    return output, mapping


@register_pydub_effect
def compress_silence_effect(audio_segment, min_silence_len=1000,
                            silence_thresh=-16, keep_silence=100,
                            crossfade=10, seek_step=1):
    """
    AudioSegment effect form of compress_silence(). Returns only the
    compressed AudioSegment; use pydub.silence.compress_silence() directly
    when the source-time to new-time mapping is needed.
    """
    compressed, _ = compress_silence(
        audio_segment, min_silence_len, silence_thresh, keep_silence,
        crossfade, seek_step)
    return compressed


