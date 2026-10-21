"""Moving Object Detection.

Real-time motion detection from a camera, built on OpenCV background
differencing. Importing this module has no side effects: the camera is
only opened when ``main()`` runs (i.e. when the file is executed as a
script).
"""

import time

import cv2
import imutils

STATUS_NORMAL = "Normal"
STATUS_MOVING = "Moving Object detected"

DEFAULT_WIDTH = 500
DEFAULT_MIN_AREA = 500
DEFAULT_THRESHOLD = 25
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


def annotate(frame, boxes, status):
    """Draw bounding boxes and the status text onto `frame` in place."""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    return frame


def main():
    cam = cv2.VideoCapture(0)
    try:
        time.sleep(1)
        first_frame = None
        while True:
            _, img = cam.read()
            text = STATUS_NORMAL
            img, blurred = preprocess(img)
            if first_frame is None:
                first_frame = blurred
                continue
            boxes = find_moving_boxes(first_frame, blurred)
            if boxes:
                text = STATUS_MOVING
            print(text)
            cv2.imshow("cameraFeed", annotate(img, boxes, text))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cam.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
