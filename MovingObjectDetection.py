"""Moving Object Detection.

Real-time motion detection from a camera or a video file, built on
OpenCV background differencing. Importing this module has no side
effects: the camera is only opened when ``main()`` runs (i.e. when the
file is executed as a script).
"""

import argparse
import sys
import time

import cv2
import imutils

STATUS_NORMAL = "Normal"
STATUS_MOVING = "Moving Object detected"

DEFAULT_WIDTH = 500
DEFAULT_MIN_AREA = 500
DEFAULT_THRESHOLD = 25
DEFAULT_LEARNING_RATE = 0.05
BLUR_KERNEL = (21, 21)


def preprocess(frame, width=DEFAULT_WIDTH):
    """Resize a BGR frame and return (resized, grayscale + blurred)."""
    resized = imutils.resize(frame, width=width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    return resized, blurred


def find_moving_boxes(background, blurred, min_area=DEFAULT_MIN_AREA,
                      threshold=DEFAULT_THRESHOLD):
    """Return bounding boxes whose change vs `background` exceeds the area."""
    diff = cv2.absdiff(background, blurred)
    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        boxes.append(cv2.boundingRect(contour))
    return boxes


class MotionDetector:
    """Stateful detector with a slowly adapting background.

    The background is a running average of the blurred frames, so a
    change that stops moving (an object left in view, a lighting shift,
    or an object that was present in the very first frame and then
    removed) fades from the background instead of being reported as
    motion forever, while an object that keeps moving keeps producing
    a difference at its new position.
    """

    def __init__(self, min_area=DEFAULT_MIN_AREA, threshold=DEFAULT_THRESHOLD,
                 learning_rate=DEFAULT_LEARNING_RATE, width=DEFAULT_WIDTH):
        self.min_area = min_area
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.width = width
        self.background = None  # float32 running average of blurred frames

    def process(self, frame):
        """Process one BGR frame; return (status, boxes, resized_frame)."""
        if frame is None:
            raise ValueError("frame must not be None")
        resized, blurred = preprocess(frame, width=self.width)
        if self.background is None:
            self.background = blurred.astype("float32")
            return STATUS_NORMAL, [], resized
        background_u8 = cv2.convertScaleAbs(self.background)
        boxes = find_moving_boxes(background_u8, blurred,
                                  min_area=self.min_area,
                                  threshold=self.threshold)
        cv2.accumulateWeighted(blurred, self.background, self.learning_rate)
        status = STATUS_MOVING if boxes else STATUS_NORMAL
        return status, boxes, resized


def annotate(frame, boxes, status):
    """Draw bounding boxes and the status text onto `frame` in place."""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    return frame


def parse_source(value):
    """A numeric source is a camera index; anything else is a file path."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def main(argv=None, video_capture=None, display=True):
    """Run the detection loop. Returns a process exit code (0 = clean)."""
    parser = argparse.ArgumentParser(description="Moving object detection")
    parser.add_argument("--source", default="0",
                        help="camera index (default 0) or path to a video file")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA,
                        help="minimum contour area treated as motion")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help="frame width to resize to before processing")
    parser.add_argument("--learning-rate", type=float,
                        default=DEFAULT_LEARNING_RATE,
                        help="background adaptation rate per frame")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a window (e.g. headless runs)")
    args = parser.parse_args(argv)
    if display and args.no_display:
        display = False

    source = parse_source(args.source)
    capture_factory = video_capture or cv2.VideoCapture
    cam = capture_factory(source)
    try:
        if not cam.isOpened():
            print(f"error: could not open video source {args.source!r}",
                  file=sys.stderr)
            return 1
        if isinstance(source, int):
            time.sleep(1)  # camera warm-up, as in the original script
        detector = MotionDetector(min_area=args.min_area, width=args.width,
                                  learning_rate=args.learning_rate)
        frames = 0
        last_status = None
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                if frames == 0:
                    print(f"error: no frames could be read from video "
                          f"source {args.source!r}", file=sys.stderr)
                    return 1
                break  # end of a video file / stream: finish cleanly
            status, boxes, resized = detector.process(frame)
            frames += 1
            if status != last_status:
                print(status)
                last_status = status
            if display:
                cv2.imshow("cameraFeed", annotate(resized, boxes, status))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        return 0
    finally:
        cam.release()
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
