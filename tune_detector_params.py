import os
import argparse
import sys

import cv2
from typing import List
import numpy as np

from logo_detector.yolov4_tiny import YOLOv4Tiny


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True,
                        help="Path to weights")
    parser.add_argument("-c", "--cfg", required=True, help="Path to config")
    parser.add_argument("-t", "--txt", required=True, help="Path to txt")
    parser.add_argument("-i", "--input", required=True, help="Path to image")
    arguments = parser.parse_args()
    return vars(arguments)


def draw_bbs(boxes: List[list], image: np.ndarray) -> None:
    for box in boxes:
        cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4
        )
        text = "{}_{:.4f}".format(box[-1], box[-2])
        cv2.putText(
            image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 0), thickness=2
        )

    return


def on_change(conf: int):
    conf_thresh = float(conf / 100)
    preds = model.predict_with_custom_thresholds(
        images=[image],
        conf=conf_thresh
    )
    preds = preds[0]
    img = image.copy()
    draw_bbs(preds, img)
    cv2.imshow(window_name, img)


def nothing(x):
    pass


if __name__ == "__main__":
    args = parse_arguments()
    try:
        model = YOLOv4Tiny(
            weights=args["weights"],
            cfg=args["cfg"],
            txt=args["txt"]
        )
    except Exception as e:
        print(f"Failed to initialize the model. Error: {e}")
        raise e

    window_name = "window"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("confidence", window_name, 0, 100, nothing)
    cv2.createTrackbar("nms", window_name, 0, 100, nothing)
    for image_name in os.listdir(args["input"]):
        image_path = os.path.join(args["input"], image_name)
        image = cv2.imread(image_path)
        assert image is not None
        img = image.copy()
        while True:
            cv2.imshow(window_name, img)
            if cv2.waitKey() == ord("q"):
                break
            elif cv2.waitKey() == ord("e"):
                sys.exit()

            # Get current trackbar positions
            conf = cv2.getTrackbarPos("confidence", window_name)
            nms = cv2.getTrackbarPos("nms", window_name)
            conf_thresh = float(conf / 100)
            nms_thresh = 1 - float(nms / 100)
            assert 0.0 <= conf_thresh <= 1.0 and 0.0 <= nms_thresh <= 1.0

            # Get new clean image and process it with new conf and nms values
            img = image.copy()
            preds = model.predict_with_custom_thresholds(
                images=[img],
                conf=conf_thresh,
                nms=nms_thresh
            )
            preds = preds[0]  # Outputs for a batch of images
            if preds:
                draw_bbs(preds, img)
