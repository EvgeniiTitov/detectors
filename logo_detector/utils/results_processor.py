import os

import cv2
from typing import List, Tuple, Set
import numpy as np


def draw_bb_for_batch_remember_detected_classes(
        images: List[np.ndarray],
        boxes: List[list],
        colour: Tuple[int] = (193, 17, 146)
) -> Set[str]:
    """
    Draws bounding boxes on a batch of images
    :param images:
    :param boxes: list containing nested lists = detections for batch
    :param colour: colour of bounding boxes
    :return:
    """
    assert len(images) == len(boxes), "Number of images != detections"
    detected_classes = set()
    for image, bbs in zip(images, boxes):
        for bb in bbs:
            cv2.rectangle(
                image, (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])), colour, 4
            )
            text = "{}_{:.4f}".format(bb[-1], bb[-2])
            cv2.putText(
                image, text, (int(bb[0]) + 5, int(bb[1]) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, colour, thickness=2
            )
            detected_classes.add(bb[-1])

    return detected_classes


def save_batch_on_disk(
        images: List[np.ndarray],
        video_writer: cv2.VideoWriter = None
) -> None:
    """"""
    if video_writer:
        for image in images:
            video_writer.write(image)
    else:
        print("[ERROR]: Video writer is not initialized!")


def create_log_file(payload: dict, save_path: str, filename: str) -> bool:
    if not filename.endswith(".txt"):
        filename = os.path.splitext(filename)[0] + ".txt"
    path_to_file = os.path.join(save_path, filename)
    try:
        with open(path_to_file, mode="w") as file:
            file.write(f"Detection results for: {filename}\n")
            file.write("\nsec: detections\n")
            for k, v in payload.items():
                line_to_write = f" {k}: " \
                                f"{' '.join([classname for classname in v])}\n"
                file.write(line_to_write)
    except Exception as e:
        print(f"Failed to write results to the log file. Error: {e}")
        return False

    return True
