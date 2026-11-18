from functools import partial
import array
import os
import sys
import unittest
from tempfile import (
    NamedTemporaryFile,
    mkdtemp,
    gettempdir
)
import tempfile
import struct

from pydub import AudioSegment
from pydub.audio_segment import extract_wav_headers
from pydub.utils import (
    db_to_float,
    ratio_to_db,
    make_chunks,
    mediainfo,
    get_encoder_name,
    get_supported_decoders,
    get_supported_encoders,
)
from pydub.exceptions import (
    InvalidTag,
    InvalidID3TagVersion,
    InvalidDuration,
    CouldntDecodeError,
    MissingAudioParameter,
)
from pydub.silence import (
    detect_silence,
    split_on_silence,
)
from pydub.generators import (
    Sine,
    Square,
    Pulse,
    Triangle,
    Sawtooth,
    WhiteNoise,
)

data_dir = os.path.join(os.path.dirname(__file__), 'data')


class UtilityTests(unittest.TestCase):

    def test_db_float_conversions(self):
        self.assertEqual(db_to_float(20), 10)
        self.assertEqual(db_to_float(10, using_amplitude=False), 10)
        self.assertEqual(db_to_float(0), 1)
        self.assertEqual(ratio_to_db(1), 0)
        self.assertEqual(ratio_to_db(10), 20)
        self.assertEqual(ratio_to_db(10, using_amplitude=False), 10)
        self.assertEqual(3, db_to_float(ratio_to_db(3)))
        self.assertEqual(12, ratio_to_db(db_to_float(12)))
        self.assertEqual(3, db_to_float(ratio_to_db(3, using_amplitude=False), using_amplitude=False))
        self.assertEqual(12, ratio_to_db(db_to_float(12, using_amplitude=False), using_amplitude=False))


if sys.version_info >= (3, 6):
    class PathLikeObjectTests(unittest.TestCase):

        class MyPathLike:
            def __init__(self, path):
                self.path = path

            def __fspath__(self):
                return self.path

        def setUp(self):
            self.mp3_path_str = os.path.join(data_dir, 'test1.mp3')

            from pathlib import Path
            self.mp3_pathlib_path = Path(self.mp3_path_str)

            self.mp3_path_like_str = self.MyPathLike(self.mp3_path_str)
            self.mp3_path_like_bytes = self.MyPathLike(bytes(self.mp3_path_str, sys.getdefaultencoding()))

        def test_audio_segment_from_pathlib_path(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_pathlib_path)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_audio_segment_from_path_like_str(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_path_like_str)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_audio_segment_from_path_like_bytes(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_path_like_bytes)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_non_existant_pathlib_path(self):
            from pathlib import Path
            path = Path('this/path/should/not/exist/do/not/make/this/exist')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = Path('')
            # On Unicies this will raise a IsADirectoryError, on Windows this
            # will result in a PermissionError. Both of these are subclasses of
            # OSError. We aren't so much worried about the specific exception
            # here, just that reading a file from an empty path is an error.
            with self.assertRaises(OSError):
                _ = AudioSegment.from_file(path)

        def test_non_existant_path_like_str(self):
            path = self.MyPathLike('this/path/should/not/exist/do/not/make/this/exist')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = self.MyPathLike('')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

        def test_non_existant_path_like_bytes(self):
            path = self.MyPathLike(bytes('this/path/should/not/exist/do/not/make/this/exist', sys.getdefaultencoding()))
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = self.MyPathLike(bytes('', sys.getdefaultencoding()))
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

        def assertWithinRange(self, val, lower_bound, upper_bound):
            self.assertTrue(lower_bound <= val <= upper_bound,
                            "%s is not in the acceptable range: %s - %s" %
                            (val, lower_bound, upper_bound))

        def assertWithinTolerance(self, val, expected, tolerance=None,
                                  percentage=None):
            if percentage is not None:
                tolerance = expected * percentage
            lower_bound = expected - tolerance
            upper_bound = expected + tolerance
            self.assertWithinRange(val, lower_bound, upper_bound)

        def test_export_pathlib_path(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            from pathlib import Path
            path = Path(tempfile.gettempdir()) / 'pydub-test-export-8ajds.mp3'
            try:
                seg1.export(path, format='mp3')
                seg2 = AudioSegment.from_file(path, format='mp3')

                self.assertTrue(len(seg1) > 0)
                self.assertWithinTolerance(len(seg1),
                                           len(seg2),
                                           percentage=0.01)
            finally:
                os.unlink(path)


class FileAccessTests(unittest.TestCase):

    def setUp(self):
        self.mp3_path = os.path.join(data_dir, 'test1.mp3')

    def test_audio_segment_from_mp3(self):
        seg1 = AudioSegment.from_mp3(os.path.join(data_dir, 'test1.mp3'))

        mp3_file = open(os.path.join(data_dir, 'test1.mp3'), 'rb')
        seg2 = AudioSegment.from_mp3(mp3_file)

        self.assertEqual(len(seg1), len(seg2))
        self.assertTrue(seg1._data == seg2._data)
        self.assertTrue(len(seg1) > 0)


test1wav = test4wav = test1 = test2 = test3 = testparty = testdcoffset = None


class AudioSegmentTests(unittest.TestCase):

    def setUp(self):
        global test1, test2, test3, testparty, testdcoffset
        if not test1:
            a = os.path.join(data_dir, 'test1.mp3')
            test1 = AudioSegment.from_mp3(os.path.join(data_dir, 'test1.mp3'))
            test2 = AudioSegment.from_mp3(os.path.join(data_dir, 'test2.mp3'))
            test3 = AudioSegment.from_mp3(os.path.join(data_dir, 'test3.mp3'))
            testdcoffset = AudioSegment.from_mp3(
                os.path.join(data_dir, 'test-dc_offset.wav'))
            testparty = AudioSegment.from_mp3(
                os.path.join(data_dir, 'party.mp3'))

        self.seg1 = test1
        self.seg2 = test2
        self.seg3 = test3
        self.mp3_seg_party = testparty
        self.seg_dc_offset = testdcoffset

        self.ogg_file_path = os.path.join(data_dir, 'bach.ogg')
        self.mp4_file_path = os.path.join(data_dir, 'creative_common.mp4')
        self.mp3_file_path = os.path.join(data_dir, 'party.mp3')
        self.webm_file_path = os.path.join(data_dir, 'test5.webm')

        self.jpg_cover_path = os.path.join(data_dir, 'cover.jpg')
        self.png_cover_path = os.path.join(data_dir, 'cover.png')

    def assertWithinRange(self, val, lower_bound, upper_bound):
        self.assertTrue(lower_bound <= val <= upper_bound,
                        "%s is not in the acceptable range: %s - %s" %
                        (val, lower_bound, upper_bound))

    def assertWithinTolerance(self, val, expected, tolerance=None,
                              percentage=None):
        if percentage is not None:
            tolerance = expected * percentage
        lower_bound = expected - tolerance
        upper_bound = expected + tolerance
        self.assertWithinRange(val, lower_bound, upper_bound)

    def test_direct_instantiation_with_bytes(self):
        seg = AudioSegment(
            b'RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x00}\x00\x00\x00\xf4\x01\x00\x04\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00')
        self.assertEqual(seg.frame_count(), 1)
        self.assertEqual(seg.channels, 2)
        self.assertEqual(seg.sample_width, 2)
        self.assertEqual(seg.frame_rate, 32000)

    def test_24_bit_audio(self):
        path24 = os.path.join(data_dir, 'test1-24bit.wav')
        seg24 = AudioSegment._from_safe_wav(path24)
        # The data length lies at bytes 40-44
        with open(path24, 'rb') as f:
            raw24 = f.read()
        len24 = struct.unpack("<L", raw24[40:44])[0]

        # should have been converted to 32 bit
        self.assertEqual(seg24.sample_width, 4)
        # the data length should have grown by exactly 4:3 (24 bits turn into 32 bits)
        self.assertEqual(len(seg24.raw_data) * 3, len24 * 4)

    def test_8_bit_audio(self):
        original_path = os.path.join(data_dir,'test1.wav')
        original_segment = AudioSegment.from_file(original_path)
        target_rms = original_segment.rms//2**8

        path_with_8bits = os.path.join(data_dir,'test1-8bit.wav')

        def check_8bit_segment(segment):
            self.assertWithinTolerance(segment.rms,target_rms,tolerance=0)

        # check reading directly
        check_8bit_segment(AudioSegment.from_file(path_with_8bits))

        # check using ffmpeg on it
        with open(path_with_8bits,'rb') as file_8bit:
            check_8bit_segment(AudioSegment.from_file(file_8bit))

        # check conversion from higher-width sample
        check_8bit_segment(AudioSegment.from_file(original_path).set_sample_width(1))

        # check audio export
        with NamedTemporaryFile('w+b', suffix='.wav') as tmp_file:
            original_segment.set_sample_width(1).export(tmp_file,format='wav')
            tmp_file.seek(0)
            check_8bit_segment(AudioSegment.from_file(tmp_file))

    def test_192khz_audio(self):
        test_files = [('test-192khz-16bit.wav', 16),
                      ('test-192khz-24bit.wav', 32),
                      ('test-192khz-32bit.flac', 32),
                      ('test-192khz-32bit.wav', 32),
                      ('test-192khz-64bit.wav', 64)]
        base_file, bit_depth = test_files[0]
        path = os.path.join(data_dir, base_file)
        base = AudioSegment.from_file(path)

        headers = extract_wav_headers(open(path, 'rb').read())
        data16_size = headers[-1].size
        self.assertEqual(len(base.raw_data), data16_size)
        self.assertEqual(base.frame_rate, 192000)
        self.assertEqual(base.sample_width, bit_depth / 8)

        for test_file, bit_depth in test_files[1:]:
            path = os.path.join(data_dir, test_file)
            seg = AudioSegment.from_file(path)
            self.assertEqual(seg.sample_width, bit_depth / 8)
            self.assertEqual(seg.frame_rate, 192000)
            self.assertEqual(len(seg.raw_data), len(base.raw_data) *
                             seg.sample_width / base.sample_width)
            self.assertEqual(seg.frame_rate, 192000)

    def test_concat(self):
        catted_audio = self.seg1 + self.seg2

        expected = len(self.seg1) + len(self.seg2)
        self.assertWithinTolerance(len(catted_audio), expected, 1)

    def test_append(self):
        merged1 = self.seg3.append(self.seg1, crossfade=100)
        merged2 = self.seg3.append(self.seg2, crossfade=100)

        self.assertEqual(len(merged1), len(self.seg1) + len(self.seg3) - 100)
        self.assertEqual(len(merged2), len(self.seg2) + len(self.seg3) - 100)

    def test_too_long_crossfade(self):
        s1 = self.seg1[:1000]
        s2 = self.seg2[:500]
        self.assertRaises(ValueError, lambda: s1.append(s2, crossfade=len(s1) + 10))

    def test_sum(self):
        def gen():
            yield self.seg1
            yield self.seg2
            yield self.seg3

        try:
            summed = sum(gen())
        except TypeError as e:
            if "unsupported operand" in str(e):
                self.fail("Could not sum() audio segments.")
            else:
                raise

    def test_volume_with_add_sub(self):
        quieter = self.seg1 - 6
        self.assertAlmostEqual(ratio_to_db(quieter.rms, self.seg1.rms),
                               -6,
                               places=2)

        louder = quieter + 2.5
        self.assertAlmostEqual(ratio_to_db(louder.rms, quieter.rms),
                               2.5,
                               places=2)

    def test_repeat_with_multiply(self):
        seg = self.seg1 * 3
        expected = len(self.seg1) * 3
        expected = (expected - 2, expected + 2)
        self.assertTrue(expected[0] < len(seg) < expected[1])

    def test_overlay(self):
        seg_mult = self.seg1[:5000] * self.seg2[:3000]
        seg_over = self.seg1[:5000].overlay(self.seg2[:3000], loop=True)

        self.assertEqual(len(seg_mult), len(seg_over))
        self.assertTrue(seg_mult._data == seg_over._data)

        self.assertEqual(len(seg_mult), 5000)
        self.assertEqual(len(seg_over), 5000)

    def test_overlay_times(self):
        # infinite
        seg_mult = self.seg1[:5000] * self.seg2[:3000]
        seg_over = self.seg1[:5000].overlay(self.seg2[:3000], times=99999999)
        self.assertEqual(len(seg_mult), len(seg_over))
        self.assertEqual(len(seg_over), 5000)
        self.assertTrue(seg_mult._data == seg_over._data)

        # no times, no-op
        piece = self.seg2[:1000]
        seg_manual = self.seg1[:4000]
        seg_over = self.seg1[:4000].overlay(piece, times=0)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 1 loop
        seg_manual = self.seg1[:4000].overlay(piece, position=500)
        seg_over = self.seg1[:4000].overlay(piece, times=1)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 2 loops
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500)
        seg_over = self.seg1[:4000].overlay(piece, times=2)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 3 loops
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500)
        seg_over = self.seg1[:4000].overlay(piece, times=3)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 4 loops (last will pass end)
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500) \
            .overlay(piece, position=3500)
        seg_over = self.seg1[:4000].overlay(piece, times=4)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 5 loops (last won't happen b/c past end)
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500) \
            .overlay(piece, position=3500).overlay(piece, position=3500)
        seg_over = self.seg1[:4000].overlay(piece, times=5)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # ~infinite, same (as 4 and 5 really)
        seg_over = self.seg1[:4000].overlay(piece, times=999999999)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

    def test_overlay_with_gain_change(self):
        # Use overlay silence with volume change
        seg_one = self.seg1[:5000]
        seg_silent = AudioSegment.silent(duration=2000)
        seg_over = seg_one.overlay(seg_silent, gain_during_overlay=-7)

        # Manually lower first segment
        seg_one_lower = seg_one - 7
        seg_manual = seg_one_lower[:2000] + seg_one[2000:]

        self.assertEqual(len(seg_over), len(seg_manual))
        self.assertAlmostEqual(seg_over.dBFS, seg_manual.dBFS)
        self.assertEqual(len(seg_manual), 5000)
        self.assertEqual(len(seg_over), 5000)

    def test_slicing(self):
        empty = self.seg1[:0]
        second_long_slice = self.seg1[:1000]
        remainder = self.seg1[1000:]

        self.assertEqual(len(empty), 0)
        self.assertEqual(len(second_long_slice), 1000)
        self.assertEqual(len(remainder), len(self.seg1) - 1000)

        last_5_seconds = self.seg1[-5000:]
        before = self.seg1[:-5000]

        self.assertEqual(len(last_5_seconds), 5000)
        self.assertEqual(len(before), len(self.seg1) - 5000)

        past_end = second_long_slice[:1500]
        self.assertTrue(second_long_slice._data == past_end._data)

    def test_slicing_by_step(self):
        audio = self.seg1[:10000]
        chunks = audio[:0]

        for chunk in audio[::1000]:
            self.assertTrue(isinstance(chunk, AudioSegment))
            self.assertEqual(len(chunk), 1000)
            chunks += chunk

        self.assertEqual(len(audio), len(chunks))

    def test_indexing(self):
        short = self.seg1[:100]

        rebuilt1 = self.seg1[:0]
        for part in short:
            rebuilt1 += part

        rebuilt2 = sum([part for part in short])

        self.assertTrue(short._data == rebuilt1._data)
        self.assertTrue(short._data == rebuilt2._data)

    def test_set_channels(self):
        mono = self.seg1.set_channels(1)
        stereo = mono.set_channels(2)

        self.assertEqual(len(self.seg1), len(mono))
        self.assertEqual(len(self.seg1), len(stereo))

        mono = self.seg2.set_channels(1)
        mono = mono.set_frame_rate(22050)

        self.assertEqual(len(mono), len(self.seg2))

        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_file:
            if sys.platform == 'win32':
                tmp_file.close()

            mono.export(tmp_file.name, 'mp3')
            monomp3 = AudioSegment.from_mp3(tmp_file.name)

            self.assertWithinTolerance(
                len(monomp3),
                len(self.seg2),
                tolerance=105
            )

            if sys.platform == 'win32':
                os.remove(tmp_file.name)

        merged = mono.append(stereo, crossfade=100)
        self.assertWithinTolerance(len(merged),
                                   len(self.seg1) + len(self.seg2) - 100,
                                   tolerance=1)

    def test_split_to_mono(self):
        seg = self.seg1
        mono_segments = seg.split_to_mono()
        seg_lchannel = mono_segments[0]
        seg_rchannel = mono_segments[1]

        self.assertEqual(len(seg_lchannel), len(seg))
        self.assertEqual(len(seg_rchannel), len(seg))

        self.assertEqual(seg_lchannel.frame_rate, seg.frame_rate)
        self.assertEqual(seg_rchannel.frame_rate, seg.frame_rate)

        self.assertEqual(seg_lchannel.frame_count(), seg.frame_count())
        self.assertEqual(seg_rchannel.frame_count(), seg.frame_count())

    def test_apply_gain_stereo(self):
        seg = self.seg1

        orig_l, orig_r = seg.split_to_mono()
        orig_dbfs_l = orig_l.dBFS
        orig_dbfs_r = orig_r.dBFS

        # for readability: infinity
        inf = float("inf")

        def assertAlmostEqual(v1, v2, **kwargs):
            if v1 in (inf, -inf):
                self.assertEqual(v1, v2)
            else:
                self.assertAlmostEqual(v1, v2, **kwargs)

        def check_stereo_gain(left_dbfs_change, right_dbfs_change):
            panned = seg.apply_gain_stereo(left_dbfs_change, right_dbfs_change)
            self.assertEqual(panned.channels, 2)

            l, r = panned.split_to_mono()
            assertAlmostEqual(l.dBFS, orig_dbfs_l + left_dbfs_change, places=2)
            assertAlmostEqual(r.dBFS, orig_dbfs_r + right_dbfs_change, places=2)

        # hard left
        check_stereo_gain(0.0, -inf)
        check_stereo_gain(0.0, -6.0)
        check_stereo_gain(0.0, 0.0)
        check_stereo_gain(-6.0, 0.0)
        check_stereo_gain(-inf, 0.0)

    def test_pan(self):
        seg = self.seg1

        orig_l, orig_r = seg.split_to_mono()
        orig_dbfs_l = orig_l.dBFS
        orig_dbfs_r = orig_r.dBFS

        # for readability: infinity
        inf = float("inf")

        def assertAlmostEqual(v1, v2, **kwargs):
            if v1 in (inf, -inf):
                self.assertEqual(v1, v2)
            else:
                self.assertAlmostEqual(v1, v2, **kwargs)

        def check_pan(pan, left_dbfs_change, right_dbfs_change):
            panned = seg.pan(pan)

            l, r = panned.split_to_mono()
            assertAlmostEqual(l.dBFS, orig_dbfs_l + left_dbfs_change, places=1)
            assertAlmostEqual(r.dBFS, orig_dbfs_r + right_dbfs_change, places=1)

        check_pan(-1.0, 3.0, -inf)
        check_pan(-0.5, 1.5, -4.65)
        check_pan(0.0, 0.0, 0.0)
        check_pan(0.5, -4.65, 1.5)
        check_pan(1.0, -inf, 3.0)

    def test_export_as_mp3(self):
        seg = self.seg1
        exported_mp3 = seg.export()
        seg_exported_mp3 = AudioSegment.from_mp3(exported_mp3)

        self.assertWithinTolerance(len(seg_exported_mp3),
                                   len(seg),
                                   percentage=0.01)

    def test_export_as_wav(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav')
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)

    def test_export_as_wav_with_codec(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav', codec='pcm_s32le')
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)
        self.assertEqual(seg_exported_wav.sample_width, 4)

    def test_export_as_wav_with_parameters(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav', parameters=['-ar', '16000', '-ac', '1'])
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)
        self.assertEqual(seg_exported_wav.frame_rate, 16000)
        self.assertEqual(seg_exported_wav.channels, 1)

    def test_export_as_raw(self):
        seg = self.seg1
        exported_raw = seg.export(format='raw')
        seg_exported_raw = AudioSegment.from_raw(exported_raw, sample_width=seg.sample_width, frame_rate=seg.frame_rate,
                                                 channels=seg.channels)

        self.assertWithinTolerance(len(seg_exported_raw),
                                   len(seg),
                                   percentage=0.01)

    def test_export_as_raw_with_codec(self):
        seg = self.seg1
        with self.assertRaises(AttributeError):
            seg.export(format='raw', codec='pcm_s32le')

    def test_export_as_raw_with_parameters(self):
        seg = self.seg1
        with self.assertRaises(AttributeError):
            seg.export(format='raw', parameters=['-ar', '16000', '-ac', '1'])

    def test_export_as_ogg(self):
        seg = self.seg1
        exported_ogg = seg.export(format='ogg')
        seg_exported_ogg = AudioSegment.from_ogg(exported_ogg)

        self.assertWithinTolerance(len(seg_exported_ogg),
                                   len(seg),
                                   percentage=0.01)

    def test_export_forced_codec(self):
        seg = self.seg1 + self.seg2

        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_file:
            if sys.platform == 'win32':
                tmp_file.close()

            seg.export(tmp_file.name, 'ogg', codec='libvorbis')
            exported = AudioSegment.from_ogg(tmp_file.name)
            self.assertWithinTolerance(len(exported),
                                       len(seg),
                                       percentage=0.01)
            if sys.platform == 'win32':
                os.remove(tmp_file.name)

    def test_fades(self):
        seg = self.seg1[:10000]

        # 1 ms difference in the position of the end of the fade out
        inf_end = seg.fade(start=0, end=float('inf'), to_gain=-120)
        negative_end = seg.fade(start=0, end=-1, to_gain=-120)

        self.assertWithinTolerance(inf_end.rms, negative_end.rms,
                                   percentage=0.001)
        self.assertTrue(negative_end.rms <= inf_end.rms)
        self.assertTrue(inf_end.rms < seg.rms)

        self.assertEqual(len(inf_end), len(seg))

        self.assertTrue(-6 < ratio_to_db(inf_end.rms, seg.rms) < -5)

        # use a slice out of the middle to make sure there is audio
        seg = self.seg2[2000:8000]
        fade_out = seg.fade_out(1000)
        fade_in = seg.fade_in(1000)

        self.assertTrue(0 < fade_out.rms < seg.rms)
        self.assertTrue(0 < fade_in.rms < seg.rms)

        self.assertEqual(len(fade_out), len(seg))
        self.assertEqual(len(fade_in), len(seg))

        db_at_beginning = ratio_to_db(fade_in[:1000].rms, seg[:1000].rms)
        db_at_end = ratio_to_db(fade_in[-1000:].rms, seg[-1000:].rms)
        self.assertTrue(db_at_beginning < db_at_end)

        db_at_beginning = ratio_to_db(fade_out[:1000].rms, seg[:1000].rms)
        db_at_end = ratio_to_db(fade_out[-1000:].rms, seg[-1000:].rms)
        self.assertTrue(db_at_end < db_at_beginning)

    def test_reverse(self):
        seg = self.seg1
        rseg = seg.reverse()

        # the reversed audio should be exactly equal in playback duration
        self.assertEqual(len(seg), len(rseg))

        r2seg = rseg.reverse()

        # if you reverse it twice you should get an identical AudioSegment
        self.assertEqual(seg, r2seg)

    def test_normalize(self):
        seg = self.seg1
        normalized = seg.normalize(0.0)

        self.assertEqual(len(normalized), len(seg))
        self.assertTrue(normalized.rms > seg.rms)
        self.assertWithinTolerance(
            normalized.max,
            normalized.max_possible_amplitude,
            percentage=0.0001
        )

    def test_for_accidental_shortening(self):
        seg = self.mp3_seg_party
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            if sys.platform == 'win32':
                tmp_mp3_file.close()

            fd = seg.export(tmp_mp3_file.name)
            fd.close()

            for i in range(3):
                fd = AudioSegment.from_mp3(tmp_mp3_file.name).export(tmp_mp3_file.name, "mp3")
                fd.close()

            tmp_seg = AudioSegment.from_mp3(tmp_mp3_file.name)
            self.assertAlmostEqual(len(tmp_seg), len(seg), places=1)

            if sys.platform == 'win32':
                os.remove(tmp_mp3_file.name)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_formats(self):
        seg_m4a = AudioSegment.from_file(
            os.path.join(data_dir, 'format_test.m4a'), "m4a")
        self.assertTrue(len(seg_m4a))

    def test_equal_and_not_equal(self):
        wav_file = self.seg1.export(format='wav')
        wav = AudioSegment.from_wav(wav_file)
        self.assertTrue(self.seg1 == wav)
        self.assertFalse(self.seg1 != wav)

    def test_duration(self):
        self.assertEqual(int(self.seg1.duration_seconds), 10)

        wav_file = self.seg1.export(format='wav')
        wav = AudioSegment.from_wav(wav_file)
        self.assertEqual(wav.duration_seconds, self.seg1.duration_seconds)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_autodetect_format(self):
        aac_path = os.path.join(data_dir, 'wrong_extension.aac')
        fn = partial(AudioSegment.from_file, aac_path, "aac")
        self.assertRaises(CouldntDecodeError, fn)

        # Trying to auto detect input file format
        aac_file = AudioSegment.from_file(
            os.path.join(data_dir, 'wrong_extension.aac'))
        self.assertEqual(int(aac_file.duration_seconds), 9)

    def test_export_ogg_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(self.ogg_file_path).export(tmp_mp3_file,
                                                              format="mp3")

    def test_export_mp3_as_ogg(self):
        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_ogg_file:
            AudioSegment.from_file(self.mp3_file_path).export(tmp_ogg_file,
                                                              format="ogg")

    def test_export_webm_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(
                self.webm_file_path,
                codec="opus"
            ).export(tmp_mp3_file, format="mp3")

    def test_export_mp3_as_webm(self):
        with NamedTemporaryFile('w+b', suffix='.webm') as tmp_webm_file:
            AudioSegment.from_file(self.mp3_file_path).export(tmp_webm_file,
                                                              format="webm")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_ogg(self):
        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_ogg_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_ogg_file,
                                                              format="ogg")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file,
                                                              format="mp3")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_wav(self):
        with NamedTemporaryFile('w+b', suffix='.wav') as tmp_wav_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_wav_file,
                                                              format="mp3")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            tags_dict = {
                'title': "The Title You Want",
                'artist': "Artist's name",
                'album': "Name of the Album"
            }
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file,
                                                              format="mp3",
                                                              tags=tags_dict)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags_raises_exception_when_tags_are_not_a_dictionary(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            json = '{"title": "The Title You Want", "album": "Name of the Album", "artist": "Artist\'s name"}'
            func = partial(
                AudioSegment.from_file(self.mp4_file_path).export, tmp_mp3_file,
                format="mp3", tags=json)
            self.assertRaises(InvalidTag, func)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags_raises_exception_when_id3version_is_wrong(self):
        tags = {'artist': 'Artist', 'title': 'Title'}
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            func = partial(
                AudioSegment.from_file(self.mp4_file_path).export,
                tmp_mp3_file,
                format="mp3",
                tags=tags,
                id3v2_version='BAD VERSION'
            )
            self.assertRaises(InvalidID3TagVersion, func)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp3_with_tags(self):
        tags = {'artist': 'Mozart', 'title': 'The Magic Flute'}

        delete = sys.platform != 'win32'

        with NamedTemporaryFile('w+b', suffix='.mp3', delete=delete) as tmp_mp3_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file, format="mp3", tags=tags)

            if sys.platform == 'win32':
                tmp_mp3_file.close()

            info = mediainfo(filepath=tmp_mp3_file.name)
            info_tags = info["TAG"]

            self.assertEqual(info_tags["artist"], "Mozart")
            self.assertEqual(info_tags["title"], "The Magic Flute")

            if sys.platform == 'win32':
                os.remove(tmp_mp3_file.name)

    def test_mp3_with_jpg_cover_img(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            outf = self.seg1.export(tmp_mp3_file, format="mp3", cover=self.jpg_cover_path)
            testseg = AudioSegment.from_file(outf, format="mp3")

            # should be within a 150ms and 1.5dB (not perfectly equal due to codecs)
            self.assertWithinTolerance(len(self.seg1), len(testseg), 150)
            self.assertWithinTolerance(self.seg1.dBFS, testseg.dBFS, 1.5)

    def test_mp3_with_png_cover_img(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            outf = self.seg1.export(tmp_mp3_file, format="mp3", cover=self.png_cover_path)
            testseg = AudioSegment.from_file(outf, format="mp3")

            # should be within a 150ms and 1.5dB (not perfectly equal due to codecs)
            self.assertWithinTolerance(len(self.seg1), len(testseg), 150)
            self.assertWithinTolerance(self.seg1.dBFS, testseg.dBFS, 1.5)

    def test_fade_raises_exception_when_duration_start_end_are_none(self):
        seg = self.seg1
        func = partial(seg.fade, start=1, end=1, duration=1)
        self.assertRaises(TypeError, func)

    def test_silent(self):
        seg = AudioSegment.silent(len(self.seg1))
        self.assertEqual(len(self.seg1), len(seg))
        self.assertEqual(seg.rms, 0)
        self.assertEqual(seg.frame_width, 2)

        seg_8bit = seg.set_sample_width(1)
        self.assertEqual(seg_8bit.sample_width, 1)
        self.assertEqual(seg_8bit.frame_width, 1)
        self.assertEqual(seg_8bit.rms, 0)

        seg *= self.seg1
        self.assertEqual(seg.rms, self.seg1.rms)
        self.assertEqual(len(seg), len(self.seg1))
        self.assertEqual(seg.frame_width, self.seg1.frame_width)
        self.assertEqual(seg.frame_rate, self.seg1.frame_rate)

    def test_from_mono_audiosegments(self):
        monoseg1 = self.seg1.set_channels(1)
        monoseg2 = monoseg1.reverse()
        stereo_sound = AudioSegment.from_mono_audiosegments(monoseg1, monoseg2)

        self.assertEqual(stereo_sound.channels, 2)
        self.assertEqual(stereo_sound.dBFS, monoseg1.dBFS)
        self.assertEqual(len(stereo_sound), len(monoseg1))

    def test_fade_raises_exception_when_duration_is_negative(self):
        seg = self.seg1
        func = partial(seg.fade,
                       to_gain=1,
                       from_gain=1,
                       start=None,
                       end=None,
                       duration=-1)
        self.assertRaises(InvalidDuration, func)

    def test_make_chunks(self):
        seg = self.seg1
        chunks = make_chunks(seg, 100)
        seg2 = chunks[0]
        for chunk in chunks[1:]:
            seg2 += chunk
        self.assertEqual(len(seg), len(seg2))

    def test_empty(self):
        self.assertEqual(len(self.seg1), len(self.seg1 + AudioSegment.empty()))
        self.assertEqual(len(self.seg2), len(self.seg2 + AudioSegment.empty()))
        self.assertEqual(len(self.seg3), len(self.seg3 + AudioSegment.empty()))

    def test_speedup(self):
        speedup_seg = self.seg1.speedup(2.0)

        self.assertWithinTolerance(
            len(self.seg1) / 2, len(speedup_seg), percentage=0.02)

    def test_dBFS(self):
        seg_8bit = self.seg1.set_sample_width(1)
        self.assertWithinTolerance(seg_8bit.dBFS, -18.06, tolerance=1.5)
        self.assertWithinTolerance(self.seg1.dBFS, -17.76, tolerance=1.5)
        self.assertWithinTolerance(self.seg2.dBFS, -20.78, tolerance=1.5)
        self.assertWithinTolerance(self.seg3.dBFS, -12.94, tolerance=1.5)

    def test_compress(self):
        compressed = self.seg1.compress_dynamic_range()
        self.assertWithinTolerance(self.seg1.dBFS - compressed.dBFS,
                                   10.0,
                                   tolerance=10.0)

        # Highest peak should be lower
        self.assertTrue(compressed.max < self.seg1.max)

        # average volume should be reduced
        self.assertTrue(compressed.rms < self.seg1.rms)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_exporting_to_ogg_uses_default_codec_when_codec_param_is_none(self):
        delete = sys.platform != 'win32'

        with NamedTemporaryFile('w+b', suffix='.ogg', delete=delete) as tmp_ogg_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_ogg_file, format="ogg")

            if sys.platform == 'win32':
                tmp_ogg_file.close()

            info = mediainfo(filepath=tmp_ogg_file.name)

            if sys.platform == 'win32':
                os.remove(tmp_ogg_file.name)

        self.assertEqual(info["codec_name"], "vorbis")
        self.assertEqual(info["format_name"], "ogg")

    def test_zero_length_segment(self):
        self.assertEqual(0, len(self.seg1[0:0]))

    def test_invert(self):
        s_mono = Sine(100).to_audio_segment()
        s = s_mono.set_channels(2)

        try:
            s_mono.invert_phase(channels=(1, 0))
        except Exception:
            pass
        else:
            raise Exception("AudioSegment.invert_phase() didn't catch a bad input (mono)")

        s_inv = s.invert_phase()
        self.assertFalse(s == s_inv)
        self.assertTrue(s.rms == s_inv.rms)
        self.assertTrue(s == s_inv.invert_phase())

        s_inv_right = s.invert_phase(channels=(0, 1))
        left, right = s_inv_right.split_to_mono()

        self.assertFalse(s_mono == s_inv_right)
        self.assertFalse(s_inv == s_inv_right)
        self.assertTrue(left == s_mono)
        self.assertFalse(right == s_mono)

        s_inv_left = s.invert_phase(channels=(1, 0))
        left, right = s_inv_left.split_to_mono()

        self.assertFalse(s_mono == s_inv_left)
        self.assertFalse(s_inv == s_inv_left)
        self.assertFalse(left == s_mono)
        self.assertTrue(right == s_mono)

    def test_max_dBFS(self):
        sine_0_dbfs = Sine(1000).to_audio_segment()
        sine_minus_3_dbfs = Sine(1000).to_audio_segment(volume=-3.0)
        self.assertAlmostEqual(-0.0, sine_0_dbfs.max_dBFS, 2)
        self.assertAlmostEqual(-3.0, sine_minus_3_dbfs.max_dBFS, 2)

    def test_array_type(self):
        self.assertEqual(self.seg1.array_type, "h")
        self.assertEqual(self.seg2.array_type, "h")
        self.assertEqual(self.seg3.array_type, "h")
        self.assertEqual(self.mp3_seg_party.array_type, "h")

        silence = AudioSegment.silent(50)
        self.assertEqual(silence.array_type, "h")
        self.assertEqual(silence.set_sample_width(1).array_type, "b")
        self.assertEqual(silence.set_sample_width(4).array_type, "i")

    def test_sample_array(self):
        samples = Sine(450).to_audio_segment().get_array_of_samples()
        self.assertEqual(
            list(samples[:8]),
            [0, 2099, 4190, 6263, 8311, 10325, 12296, 14217]
        )

    def test_get_dc_offset(self):
        seg = self.seg_dc_offset
        self.assertWithinTolerance(seg.get_dc_offset(), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg.get_dc_offset(1), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg.get_dc_offset(2), 0.1, tolerance=0.01)

    def test_remove_dc_offset(self):
        seg = self.seg_dc_offset

        seg1 = seg.remove_dc_offset()
        self.assertWithinTolerance(seg1.get_dc_offset(1), 0.0, tolerance=0.0001)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.0, tolerance=0.0001)

        seg1 = seg.remove_dc_offset(1)
        self.assertWithinTolerance(seg1.get_dc_offset(1), 0.0, tolerance=0.0001)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.1, tolerance=0.01)

        seg1 = seg.remove_dc_offset(2)
        self.assertWithinTolerance(seg1.get_dc_offset(1), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.0, tolerance=0.0001)

        seg1 = seg.remove_dc_offset(channel=1, offset=(-0.06))
        self.assertWithinTolerance(seg1.get_dc_offset(1), -0.1, tolerance=0.01)

    def test_from_file_clean_fail(self):
        delete = sys.platform != 'win32'
        orig_tmpdir = gettempdir()
        new_tmpdir = mkdtemp()
        tempfile.tempdir = new_tmpdir

        with NamedTemporaryFile('w+b', suffix='.wav', delete=delete) as tmp_wav_file:
            tmp_wav_file.write("not really a wav".encode('utf-8'))
            tmp_wav_file.flush()
            self.assertRaises(CouldntDecodeError, AudioSegment.from_file, tmp_wav_file.name)
            files = os.listdir(tempfile.tempdir)
            self.assertEqual(files, [os.path.basename(tmp_wav_file.name)])

        if sys.platform == 'win32':
            os.remove(tmp_wav_file.name)

        tempfile.tempdir = orig_tmpdir
        os.rmdir(new_tmpdir)


class SilenceTests(unittest.TestCase):

    def setUp(self):
        global test1wav, test4wav
        if not test1wav:
            test1wav = AudioSegment.from_wav(os.path.join(data_dir, 'test1.wav'))
        if not test4wav:
            test4wav = AudioSegment.from_wav(os.path.join(data_dir, 'test4.wav'))

        self.seg1 = test1wav
        self.seg4 = test4wav

    def test_split_on_silence_complete_silence(self):
        seg = AudioSegment.silent(5000)
        self.assertEqual( split_on_silence(seg), [] )

    def test_split_on_silence_test1(self):
        self.assertEqual(
            len(split_on_silence(self.seg1, min_silence_len=500, silence_thresh=-20)),
            3
        )
    def test_split_on_silence_no_silence(self):
        splits = split_on_silence(self.seg1, min_silence_len=5000, silence_thresh=-200, keep_silence=True)
        lens = [len(split) for split in splits]
        self.assertEqual( lens, [len(self.seg1)] )

    def test_detect_completely_silent_segment(self):
        seg = AudioSegment.silent(5000)
        silent_ranges = detect_silence(seg, min_silence_len=1000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 5000]])

    def test_detect_tight_silent_segment(self):
        seg = AudioSegment.silent(1000)
        silent_ranges = detect_silence(seg, min_silence_len=1000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 1000]])

    def test_detect_too_long_silence(self):
        seg = AudioSegment.silent(3000)
        silent_ranges = detect_silence(seg, min_silence_len=5000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [])

    def test_detect_silence_seg1(self):
        silent_ranges = detect_silence(self.seg1, min_silence_len=500, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 775], [3141, 4033], [5516, 6051]])

    def test_detect_silence_seg1_with_seek_split(self):
        silent_ranges = detect_silence(self.seg1, min_silence_len=500, silence_thresh=-20,
                                       seek_step=10)
        self.assertEqual(silent_ranges, [[0, 770], [3150, 4030], [5520, 6050]])

    def test_realistic_audio(self):
        silent_ranges = detect_silence(self.seg4, min_silence_len=1000, silence_thresh=self.seg4.dBFS)

        prev_end = -1
        for start, end in silent_ranges:
            self.assertTrue(start > prev_end)
            prev_end = end


class GeneratorTests(unittest.TestCase):

    def test_with_smoke(self):
        Sine(440).to_audio_segment()
        Square(440).to_audio_segment()
        Triangle(440).to_audio_segment()

        Pulse(440, duty_cycle=0.75).to_audio_segment()
        Sawtooth(440, duty_cycle=0.75).to_audio_segment()

        WhiteNoise().to_audio_segment()

    def test_loudness(self):
        sine_dbfs = Sine(440).to_audio_segment().dBFS
        square_dbfs = Square(440).to_audio_segment().dBFS
        white_noise_dbfs = WhiteNoise().to_audio_segment().dBFS

        self.assertAlmostEqual(sine_dbfs, -3.0, places=1)
        self.assertAlmostEqual(square_dbfs, 0.0, places=1)
        self.assertAlmostEqual(white_noise_dbfs, -5, places=0)

    def test_duration(self):
        one_sec = Sine(440).to_audio_segment(duration=1000)
        five_sec = Sine(440).to_audio_segment(duration=5000)
        half_sec = Sine(440).to_audio_segment(duration=500)

        self.assertAlmostEqual(len(one_sec), 1000)
        self.assertAlmostEqual(len(five_sec), 5000)
        self.assertAlmostEqual(len(half_sec), 500)


class NoConverterTests(unittest.TestCase):

    def setUp(self):
        self.wave_file = os.path.join(data_dir, 'test1.wav')
        self.wave24_file = os.path.join(data_dir, 'test1-24bit.wav')
        self.wave_empty = os.path.join(data_dir, 'test1_empty.wav')
        self.mp3_file = os.path.join(data_dir, 'test1.mp3')
        self.raw_file = os.path.join(data_dir, 'test1.raw')
        AudioSegment.converter = "definitely-not-a-path-to-anything-asdjklqwop"

    def tearDown(self):
        AudioSegment.converter = get_encoder_name()

    def test_opening_wav_file(self):
        seg = AudioSegment.from_wav(self.wave_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file, "wav")
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file, format="wav")
        self.assertTrue(len(seg) > 1000)

    def test_opening_wav24_file(self):
        seg = AudioSegment.from_wav(self.wave24_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file, "wav")
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file, format="wav")
        self.assertTrue(len(seg) > 1000)

    def test_opening_raw_file(self):
        seg = AudioSegment.from_raw(self.raw_file, sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.raw_file, "raw", sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.raw_file, format="raw", sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

    def test_opening_raw_file_with_missing_args_fails(self):
        func = partial(AudioSegment.from_raw, self.raw_file)
        self.assertRaises(KeyError, func)

    def test_opening_mp3_file_fails(self):
        func = partial(AudioSegment.from_mp3, self.mp3_file)
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file)
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file, "mp3")
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file, format="mp3")
        self.assertRaises(OSError, func)

    def test_init_AudioSegment_data_buffer(self):
        seg = AudioSegment(data="\0" * 34, sample_width=2, frame_rate=4, channels=1)

        self.assertEqual(seg.duration_seconds, 4.25)

        self.assertEqual(seg.sample_width, 2)

        self.assertEqual(seg.frame_rate, 4)

    def test_init_AudioSegment_data_buffer_with_missing_args_fails(self):
        func = partial(AudioSegment, data="\0" * 16, sample_width=2, frame_rate=2)
        self.assertRaises(MissingAudioParameter, func)

        func = partial(AudioSegment, data="\0" * 16, sample_width=2, channels=1)
        self.assertRaises(MissingAudioParameter, func)

        func = partial(AudioSegment, data="\0" * 16, frame_rate=2, channels=1)
        self.assertRaises(MissingAudioParameter, func)

    def test_init_AudioSegment_data_buffer_with_bad_values_fails(self):
        func = partial(AudioSegment, data="\0" * 14, sample_width=4, frame_rate=2, channels=1)
        self.assertRaises(ValueError, func)

    def test_exporting(self):
        seg = AudioSegment.from_wav(self.wave_file)
        exported = AudioSegment.from_wav(seg.export(format="wav"))

        self.assertEqual(len(exported), len(seg))

    def test_opening_empty_wav_file(self):
        seg = AudioSegment.from_wav(self.wave_empty)
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty)
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty, "wav")
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty, format="wav")
        self.assertTrue(len(seg) == 0)


class FilterTests(unittest.TestCase):

    def setUp(self):
        global test1wav
        if not test1wav:
            test1wav = AudioSegment.from_wav(os.path.join(data_dir, 'test1.wav'))

        self.seg1 = test1wav

    def test_highpass_works_on_multichannel_segments(self):
        self.assertEqual(self.seg1.channels, 2)
        less_bass = self.seg1.high_pass_filter(800)
        self.assertTrue(less_bass.dBFS < self.seg1.dBFS)

    def test_highpass_filter_reduces_loudness(self):
        s = Square(200).to_audio_segment()
        less_bass = s.high_pass_filter(400)
        self.assertTrue(less_bass.dBFS < s.dBFS)

    def test_highpass_filter_cutoff_frequency(self):
        # A Sine wave should not be affected by a HPF 3 octaves lower
        s = Sine(800).to_audio_segment()
        less_bass = s.high_pass_filter(100)
        self.assertAlmostEqual(less_bass.dBFS, s.dBFS, places=0)

    def test_lowpass_filter_reduces_loudness(self):
        s = Square(200).to_audio_segment()
        less_treble = s.low_pass_filter(400)
        self.assertTrue(less_treble.dBFS < s.dBFS)

    def test_lowpass_filter_cutoff_frequency(self):
        # A Sine wave should not be affected by a LPF 3 octaves Higher
        s = Sine(100).to_audio_segment()
        less_treble = s.low_pass_filter(800)
        self.assertAlmostEqual(less_treble.dBFS, s.dBFS, places=0)


class PartialAudioSegmentLoadTests(unittest.TestCase):

    def setUp(self):
        self.wave_path_str = os.path.join(data_dir, 'test1.wav')
        self.mp3_path_str = os.path.join(data_dir, 'test1.mp3')
        self.raw_path_str = os.path.join(data_dir, 'test1.raw')

    def tearDown(self):
        AudioSegment.converter = get_encoder_name()

    def test_partial_load_duration_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[:1000]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[1000:]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_duration_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[:1000]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[1000:]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_duration_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[:1000]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[1000:]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)


try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


@unittest.skipUnless(_HAS_NUMPY, "convolve effect requires numpy")
class ConvolveTests(unittest.TestCase):
    """
    Tests for AudioSegment.convolve(), the convolution reverb effect.

    The reference signals are built from explicit integer samples so the
    expected outputs can be calculated by hand rather than by another
    convolution implementation.
    """

    frame_rate = 8000
    sample_width = 2
    values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

    def setUp(self):
        self.seg = AudioSegment(
            array.array('h', self.values),
            sample_width=self.sample_width,
            frame_rate=self.frame_rate,
            channels=1)
        # full-scale unit impulse: [+full_scale, 0, 0] -> unity gain
        self.identity_ir = AudioSegment(
            array.array('h', [32767, 0, 0]),
            sample_width=self.sample_width,
            frame_rate=self.frame_rate,
            channels=1)
        # two-tap response: 0.5 gain taps two frames apart
        self.two_tap_ir = AudioSegment(
            array.array('h', [16384, 0, 16384]),
            sample_width=self.sample_width,
            frame_rate=self.frame_rate,
            channels=1)
        self.two_tap_expected = [
            int(round(
                0.5 * (self.values[k] if k < len(self.values) else 0)
                + 0.5 * (self.values[k - 2] if k - 2 >= 0 else 0)))
            for k in range(len(self.values) + 3 - 1)
        ]

    def _mono(self, values, frame_rate=None, sample_width=None):
        typecode = {1: 'b', 2: 'h', 4: 'i'}[
            sample_width or self.sample_width]
        return AudioSegment(
            array.array(typecode, values),
            sample_width=sample_width or self.sample_width,
            frame_rate=frame_rate or self.frame_rate,
            channels=1)

    # --- unit impulse -----------------------------------------------------

    def test_unit_impulse_wet_only_is_identity(self):
        out = self.seg.convolve(self.identity_ir, dry=0, wet=1)
        self.assertEqual(out.get_array_of_samples().tolist(), self.values)

    def test_unit_impulse_dry_only_is_identity(self):
        out = self.seg.convolve(self.identity_ir, dry=1, wet=0)
        self.assertEqual(out.get_array_of_samples().tolist(), self.values)

    def test_unit_impulse_default_doubles_signal(self):
        # dry=1 and wet=1 with an identity impulse yields 2x the signal
        out = self.seg.convolve(self.identity_ir)
        self.assertEqual(out.get_array_of_samples().tolist(),
                         [2 * v for v in self.values])

    def test_unit_impulse_tail_extends_with_silence(self):
        out = self.seg.convolve(
            self.identity_ir, dry=0, wet=1, include_tail=True)
        self.assertEqual(int(out.frame_count()),
                         int(self.seg.frame_count()) + 2)
        self.assertEqual(out.get_array_of_samples().tolist(),
                         self.values + [0, 0])

    def test_no_tail_keeps_input_length(self):
        out = self.seg.convolve(
            self.two_tap_ir, dry=0, wet=1, include_tail=False)
        self.assertEqual(int(out.frame_count()), int(self.seg.frame_count()))
        self.assertEqual(len(out), len(self.seg))

    # --- two-tap impulse response: numeric correctness --------------------

    def test_two_tap_response_numeric_values_with_tail(self):
        out = self.seg.convolve(
            self.two_tap_ir, dry=0, wet=1, include_tail=True)
        self.assertEqual(out.get_array_of_samples().tolist(),
                         self.two_tap_expected)

    def test_two_tap_response_duration(self):
        out = self.seg.convolve(
            self.two_tap_ir, dry=0, wet=1, include_tail=True)
        expected_frames = len(self.values) + 3 - 1
        self.assertEqual(int(out.frame_count()), expected_frames)
        self.assertEqual(len(out),
                         round(1000 * expected_frames / self.frame_rate))

    def test_dry_wet_mix_is_linear_combination(self):
        seg = self._mono([100, 200, 0, 0])
        ir = self._mono([16384, 16384])  # [0.5, 0.5]
        # wet convolution (first 4 frames): [50, 150, 100, 0]
        # mix = 0.25 * dry + 0.5 * wet
        out = seg.convolve(ir, dry=0.25, wet=0.5)
        self.assertEqual(out.get_array_of_samples().tolist(),
                         [50, 125, 50, 0])

    def test_zero_dry_and_wet_is_silence(self):
        out = self.seg.convolve(
            self.two_tap_ir, dry=0, wet=0, include_tail=True)
        self.assertEqual(int(out.frame_count()),
                         len(self.values) + 3 - 1)
        self.assertEqual(
            max(abs(s) for s in out.get_array_of_samples()), 0)

    # --- block-based vs one-shot convolution ------------------------------

    def test_block_processing_matches_one_shot(self):
        rng = np.random.default_rng(42)
        signal = (rng.standard_normal(5000) * 20000).astype(np.int16).tolist()
        ir_values = (rng.standard_normal(137) * 5000).astype(np.int16).tolist()
        long_seg = self._mono(signal)
        ir = self._mono(ir_values)

        one_shot = long_seg.convolve(
            ir, dry=0.5, wet=0.7, include_tail=True)
        blocked = long_seg.convolve(
            ir, dry=0.5, wet=0.7, include_tail=True, chunk_duration=73)

        self.assertEqual(one_shot.frame_count(), blocked.frame_count())
        a = np.asarray(one_shot.get_array_of_samples(), dtype=np.int64)
        b = np.asarray(blocked.get_array_of_samples(), dtype=np.int64)
        # same FFT kernel + pre-mix quantization => at most one LSB apart
        self.assertLessEqual(int(np.max(np.abs(a - b))), 1)

    def test_block_processing_with_ir_longer_than_block(self):
        rng = np.random.default_rng(7)
        signal = (rng.standard_normal(4000) * 15000).astype(np.int16).tolist()
        ir_values = (rng.standard_normal(300) * 3000).astype(np.int16).tolist()
        long_seg = self._mono(signal)
        ir = self._mono(ir_values)

        one_shot = long_seg.convolve(ir, include_tail=True)
        blocked = long_seg.convolve(
            ir, include_tail=True, chunk_duration=50)
        a = np.asarray(one_shot.get_array_of_samples(), dtype=np.int64)
        b = np.asarray(blocked.get_array_of_samples(), dtype=np.int64)
        self.assertEqual(a.shape, b.shape)
        self.assertLessEqual(int(np.max(np.abs(a - b))), 1)

    def test_single_sample_chunk_does_not_crash(self):
        out = self.seg.convolve(
            self.two_tap_ir, dry=0, wet=1, include_tail=True,
            chunk_duration=1)
        self.assertEqual(out.get_array_of_samples().tolist(),
                         self.two_tap_expected)

    # --- channels ----------------------------------------------------------

    def test_mono_ir_applied_to_every_stereo_channel(self):
        stereo = AudioSegment.from_mono_audiosegments(
            self.seg, self.seg)
        out = stereo.convolve(self.identity_ir, dry=0, wet=1)
        self.assertEqual(out.channels, 2)
        self.assertEqual(int(out.frame_count()), int(self.seg.frame_count()))
        samples = out.get_array_of_samples().tolist()
        self.assertEqual(samples[0::2], self.values)
        self.assertEqual(samples[1::2], self.values)

    def test_stereo_ir_convolves_matching_channels(self):
        stereo = AudioSegment.from_mono_audiosegments(
            self.seg, self.seg)
        stereo_ir = AudioSegment.from_mono_audiosegments(
            self.identity_ir, self.two_tap_ir)
        out = stereo.convolve(
            stereo_ir, dry=0, wet=1, include_tail=True)
        samples = out.get_array_of_samples().tolist()
        self.assertEqual(samples[0::2], self.values + [0, 0])
        self.assertEqual(samples[1::2], self.two_tap_expected)

    def test_mono_ir_on_stereo_built_with_set_channels(self):
        stereo = self.seg.set_channels(2)
        out = stereo.convolve(self.identity_ir, dry=0, wet=1)
        samples = out.get_array_of_samples().tolist()
        self.assertEqual(samples[0::2], self.values)
        self.assertEqual(samples[1::2], self.values)

    def test_mismatched_channel_count_raises(self):
        stereo = AudioSegment.from_mono_audiosegments(
            self.seg, self.seg)
        three_channel_ir = AudioSegment.from_mono_audiosegments(
            self.identity_ir, self.identity_ir, self.identity_ir)
        with self.assertRaises(ValueError):
            stereo.convolve(three_channel_ir)

    # --- error conditions --------------------------------------------------

    def test_empty_impulse_response_raises(self):
        empty = AudioSegment(
            b'', metadata={"channels": 1, "sample_width": 2,
                           "frame_rate": self.frame_rate,
                           "frame_width": 2})
        with self.assertRaises(ValueError):
            self.seg.convolve(empty)

    def test_non_audio_segment_impulse_response_raises(self):
        with self.assertRaises(TypeError):
            self.seg.convolve([1, 0, 0])

    def test_invalid_dry_wet_ratios_raise(self):
        for bad in (-0.1, 1.1, float('nan'), float('inf'), 'nope'):
            with self.assertRaises(ValueError):
                self.seg.convolve(self.identity_ir, dry=bad)
            with self.assertRaises(ValueError):
                self.seg.convolve(self.identity_ir, wet=bad)

    def test_boundary_ratios_are_accepted(self):
        # exactly 0.0 and 1.0 must be accepted
        out = self.seg.convolve(self.identity_ir, dry=0.0, wet=1.0)
        self.assertEqual(len(out), len(self.seg))
        out = self.seg.convolve(self.identity_ir, dry=1.0, wet=0.0)
        self.assertEqual(len(out), len(self.seg))

    # --- silence, clipping, empty segment ---------------------------------

    def test_all_silent_input_stays_silent(self):
        silent = AudioSegment.silent(
            duration=10, frame_rate=self.frame_rate).set_sample_width(2)
        out = silent.convolve(self.two_tap_ir, include_tail=True)
        self.assertEqual(
            max(abs(s) for s in out.get_array_of_samples()), 0)
        self.assertEqual(int(out.frame_count()),
                         int(silent.frame_count()) + 2)

    def test_silent_ir_produces_silence(self):
        silent_ir = self._mono([0, 0, 0, 0])
        out = self.seg.convolve(silent_ir, dry=0, wet=1, include_tail=True)
        self.assertEqual(
            max(abs(s) for s in out.get_array_of_samples()), 0)

    def test_overload_saturates_without_wrapping(self):
        max_positive = (1 << 15) - 1
        min_negative = -(1 << 15)
        loud = self._mono([max_positive] * 4)
        out = loud.convolve(self.identity_ir, dry=1, wet=1)
        self.assertEqual(out.get_array_of_samples().tolist(),
                         [max_positive] * 4)

        loud_negative = self._mono([min_negative] * 4)
        out = loud_negative.convolve(self.identity_ir, dry=1, wet=1)
        self.assertEqual(out.get_array_of_samples().tolist(),
                         [min_negative] * 4)

    def test_empty_segment_returned_with_zero_frames(self):
        empty = self.seg[0:0]
        out = empty.convolve(self.identity_ir)
        self.assertEqual(int(out.frame_count()), 0)
        self.assertEqual(out.channels, empty.channels)

    # --- frame rate, sample width and export rules ------------------------

    def test_output_keeps_frame_rate_sample_width_channels(self):
        out = self.seg.convolve(self.identity_ir, dry=0, wet=1)
        self.assertEqual(out.frame_rate, self.seg.frame_rate)
        self.assertEqual(out.sample_width, self.seg.sample_width)
        self.assertEqual(out.channels, self.seg.channels)

    def test_impulse_response_frame_rate_is_matched(self):
        ir = self._mono([32767, 0, 0], frame_rate=4000)
        out = self.seg.convolve(ir, dry=0, wet=1)
        self.assertEqual(out.frame_rate, self.frame_rate)
        self.assertEqual(out.sample_width, self.sample_width)

    def test_8bit_and_32bit_outputs_keep_sample_width(self):
        for sample_width, typecode, full_scale in ((1, 'b', 127),
                                                   (4, 'i', 2147483647)):
            seg = self.seg.set_sample_width(sample_width)
            ir = AudioSegment(
                array.array(typecode, [full_scale, 0, 0]),
                sample_width=sample_width,
                frame_rate=self.frame_rate, channels=1)
            out = seg.convolve(ir, dry=0, wet=1)
            self.assertEqual(out.sample_width, sample_width)
            # a full-scale unit impulse reproduces the input exactly
            self.assertEqual(out.get_array_of_samples().tolist(),
                             seg.get_array_of_samples().tolist())

    def test_result_survives_wav_export_round_trip(self):
        out = self.seg.convolve(
            self.two_tap_ir, dry=0.3, wet=0.7, include_tail=True)
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            path = tmp.name
        try:
            out.export(path, format='wav')
            reloaded = AudioSegment.from_wav(path)
            self.assertEqual(reloaded.frame_rate, out.frame_rate)
            self.assertEqual(reloaded.channels, out.channels)
            self.assertEqual(reloaded.sample_width, out.sample_width)
            self.assertEqual(int(reloaded.frame_count()),
                             int(out.frame_count()))
            self.assertEqual(reloaded.get_array_of_samples().tolist(),
                             out.get_array_of_samples().tolist())
        finally:
            os.unlink(path)

    def test_stereo_result_survives_wav_export_round_trip(self):
        stereo = AudioSegment.from_mono_audiosegments(
            self.seg, self.seg)
        out = stereo.convolve(
            self.two_tap_ir, dry=0.5, wet=0.5, include_tail=True)
        with NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            path = tmp.name
        try:
            out.export(path, format='wav')
            reloaded = AudioSegment.from_wav(path)
            self.assertEqual(reloaded.channels, 2)
            self.assertEqual(reloaded.get_array_of_samples().tolist(),
                             out.get_array_of_samples().tolist())
        finally:
            os.unlink(path)


if __name__ == "__main__":
    import sys

    if sys.version_info >= (3, 1):
        unittest.main(warnings="ignore")
    else:
        unittest.main()
