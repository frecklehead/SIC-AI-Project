"""Comparison tests for the 3:1 resampler replacement. Not run here."""
import unittest

import numpy as np

from resampler import Polyphase3Resampler, StreamResampler


def compare_streams(sr_in, sr_out, samples, sizes=(128, 480, 960, 2048)):
    old = StreamResampler(sr_in, sr_out)
    new = Polyphase3Resampler(sr_in, sr_out)
    old_parts, new_parts = [], []
    cursor = 0
    index = 0
    while cursor < samples.size:
        end = min(samples.size, cursor + sizes[index % len(sizes)])
        old_parts.append(old.process(samples[cursor:end]).copy())
        new_parts.append(new.process(samples[cursor:end]).copy())
        cursor, index = end, index + 1
    ref = np.concatenate(old_parts)
    actual = np.concatenate(new_parts)
    n = min(ref.size, actual.size)
    ref, actual = ref[:n], actual[:n]
    max_abs = float(np.max(np.abs(ref - actual))) if n else 0.0
    ref_spectrum = np.fft.rfft(ref.astype(np.float64))
    error_spectrum = np.fft.rfft((ref - actual).astype(np.float64))
    snr_db = 20 * np.log10((np.linalg.norm(ref_spectrum) + 1e-20) /
                           (np.linalg.norm(error_spectrum) + 1e-20))
    return max_abs, float(snr_db), ref.size, actual.size


class Polyphase3ParityTests(unittest.TestCase):
    def test_single_call_equals_requested_block_sizes(self):
        rng = np.random.default_rng(41)
        for sr_in, sr_out in ((48000, 16000), (16000, 48000)):
            signal = rng.standard_normal(1217).astype(np.float32)
            one_call = Polyphase3Resampler(sr_in, sr_out).process(signal).copy()
            for block_size in (1, 160, 480, 1000, 4096, signal.size):
                resampler = Polyphase3Resampler(sr_in, sr_out)
                pieces = [resampler.process(signal[i:i + block_size]).copy()
                          for i in range(0, signal.size, block_size)]
                chunked = np.concatenate(pieces)
                self.assertEqual(chunked.size, one_call.size)
                self.assertLessEqual(float(np.max(np.abs(chunked - one_call))), 1e-6)

    def test_empty_and_non_multiple_of_three_lengths(self):
        for rates in ((48000, 16000), (16000, 48000)):
            resampler = Polyphase3Resampler(*rates)
            self.assertEqual(resampler.process(np.empty(0, np.float32)).size, 0)
            output = resampler.process(np.ones(101, np.float32))
            self.assertGreater(output.size, 0)

    def test_rejects_wrong_dtype_and_shape(self):
        resampler = Polyphase3Resampler(48000, 16000)
        with self.assertRaises(TypeError):
            resampler.process(np.ones(10, np.int16))
        with self.assertRaises(ValueError):
            resampler.process(np.ones((2, 5), np.float32))

    def test_large_upsample_is_split_internally(self):
        rng = np.random.default_rng(44)
        signal = rng.standard_normal(Polyphase3Resampler._CAPACITY + 257).astype(np.float32)
        whole = Polyphase3Resampler(16000, 48000).process(signal)
        max_abs, snr_db, ref_n, actual_n = compare_streams(
            16000, 48000, signal, sizes=(4096,))
        self.assertEqual(whole.size, ref_n)
        self.assertEqual(ref_n, actual_n)
        self.assertLess(max_abs, 2e-5)
        self.assertGreaterEqual(snr_db, 60.0)

    def test_fifteen_second_downsample_length_includes_only_filter_delay(self):
        resampler = Polyphase3Resampler(48000, 16000)
        signal = np.zeros(15 * 48000, np.float32)
        output = resampler.process(signal)
        expected = 15 * 16000
        delay_out = int(np.ceil(resampler.half / 3))
        self.assertLessEqual(output.size, expected)
        self.assertLessEqual(expected - output.size, delay_out)
        self.assertGreaterEqual(output.size, expected - delay_out)

    def test_downsample_matches_old_stream_resampler(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(48000 * 3).astype(np.float32)
        max_abs, snr_db, ref_n, actual_n = compare_streams(48000, 16000, signal)
        self.assertEqual(actual_n, ref_n)
        self.assertLess(max_abs, 2e-5)
        self.assertGreaterEqual(snr_db, 60.0)

    def test_upsample_matches_old_stream_resampler(self):
        rng = np.random.default_rng(43)
        signal = rng.standard_normal(16000).astype(np.float32)
        max_abs, snr_db, ref_n, actual_n = compare_streams(16000, 48000, signal)
        self.assertEqual(actual_n, ref_n)
        self.assertLess(max_abs, 2e-5)
        self.assertGreaterEqual(snr_db, 60.0)


if __name__ == "__main__":
    unittest.main()
