"""Headless tests for MovingObjectDetection.

Run with:  python -m unittest test_moving_object_detection -v
Uses synthetic frames and a fake capture, so no camera is needed.
"""

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location(
    "moving_object_detection",
    Path(__file__).resolve().parent / "MovingObjectDetection.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # importing must not open a camera or loop


def blank(value=100):
    return np.full((480, 640, 3), value, np.uint8)


def with_rect(x=200, y=150):
    frame = blank()
    cv2.rectangle(frame, (x, y), (x + 120, y + 120), (255, 255, 255), -1)
    return frame


class FakeCam:
    def __init__(self, frames, drop_at=None, opened=True):
        self.frames, self.drop_at = frames, drop_at
        self.i, self.released, self._opened = 0, False, opened

    def isOpened(self):
        return self._opened

    def read(self):
        if self.drop_at is not None and self.i >= self.drop_at:
            return False, None
        frame = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class DetectorTests(unittest.TestCase):
    def test_static_scene_is_normal(self):
        det = mod.MotionDetector()
        for _ in range(10):
            self.assertEqual(det.process(blank())[0], mod.STATUS_NORMAL)

    def test_none_frame_rejected(self):
        with self.assertRaises(ValueError):
            mod.MotionDetector().process(None)

    def test_moving_object_detected(self):
        det = mod.MotionDetector()
        det.process(blank())
        hits = sum(det.process(with_rect(x=60 + 15 * i))[0] == mod.STATUS_MOVING
                   for i in range(15))
        self.assertGreaterEqual(hits, 10)

    def test_ghost_background_adapts(self):
        det = mod.MotionDetector()
        det.process(with_rect())  # object present in the first frame, then gone
        seq = [det.process(blank())[0] for _ in range(150)]
        self.assertEqual(seq[0], mod.STATUS_MOVING)
        self.assertNotIn(mod.STATUS_MOVING, seq[-20:])

    def test_small_blob_ignored(self):
        det = mod.MotionDetector()
        det.process(blank())
        small = blank()
        cv2.rectangle(small, (10, 10), (25, 25), (255, 255, 255), -1)
        self.assertEqual(det.process(small)[0], mod.STATUS_NORMAL)


class MainLoopTests(unittest.TestCase):
    def run_main(self, cam):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main([], video_capture=lambda src: cam, display=False)
        return rc, out.getvalue(), err.getvalue()

    def test_unopened_source_fails_cleanly(self):
        cam = FakeCam([blank()], opened=False)
        rc, _, err = self.run_main(cam)
        self.assertEqual(rc, 1)
        self.assertIn("could not open", err)
        self.assertTrue(cam.released)

    def test_first_read_failure_fails_cleanly(self):
        cam = FakeCam([blank()], drop_at=0)
        rc, _, err = self.run_main(cam)
        self.assertEqual(rc, 1)
        self.assertIn("no frames", err)
        self.assertTrue(cam.released)

    def test_eof_after_frames_is_clean_and_prints_on_change_only(self):
        cam = FakeCam([blank()], drop_at=40)
        rc, out, _ = self.run_main(cam)
        self.assertEqual(rc, 0)
        self.assertTrue(cam.released)
        self.assertEqual(out.splitlines(), [mod.STATUS_NORMAL])


class ParseSourceTests(unittest.TestCase):
    def test_numeric_is_camera_index(self):
        self.assertEqual(mod.parse_source("0"), 0)

    def test_path_stays_a_path(self):
        self.assertEqual(mod.parse_source("clip.mp4"), "clip.mp4")


if __name__ == "__main__":
    unittest.main()
