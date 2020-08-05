import os
import cv2
import argparse
import numpy as np
import sys
from typing import List, Tuple


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to a folder with image-txt pairs for visualisation")

    return parser.parse_args()


def read_txt_content(path_to_txt: str) -> list:
    boxes = list()
    with open(path_to_txt, "r") as text_file:
        for line in text_file:
            items = line.split()
            boxes.append(items)

    return boxes


def draw_bb(image: np.ndarray, bbs: list) -> None:
    image_height, image_width = image.shape[:2]
    for bb in bbs:
        centre_x = int(float(bb[1]) * image_width)
        centre_y = int(float(bb[2]) * image_height)
        bb_width = int(float(bb[3]) * image_width)
        bb_height = int(float(bb[4]) * image_height)

        left = centre_x - (bb_width // 2)
        top = centre_y - (bb_height // 2)
        right = left + bb_width
        bottom = top + bb_height
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    return


def collect_img_txt_pairs(path_to_folder: str) -> Tuple[list, list]:
    path_to_images, path_to_txts = list(), list()
    for filename in os.listdir(path_to_folder):
        if filename.endswith("txt"):
            path_to_txts.append(os.path.join(path_to_folder, filename))
        elif any(filename.endswith(ext) for ext in [".jpg", ".JPG", ".png", ".PNG", "jpeg", "JPEG"]):
            path_to_images.append(os.path.join(path_to_folder, filename))
        else:
            continue

    assert len(path_to_images) == len(path_to_txts), "Number of images and txts do not match"
    assert set(os.path.splitext(e)[0] for e in path_to_images) == \
                            set(os.path.splitext(e)[0] for e in path_to_txts), "Labels do not match images"

    return path_to_images, path_to_txts


def visualise_images(path_to_images: List[str], path_to_txts: List[str]) -> None:
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    while path_to_images and path_to_txts:
        path_to_image = path_to_images.pop(0)
        path_to_txt = path_to_txts.pop(0)

        try:
            image = cv2.imread(path_to_image)
        except Exception as e:
            print(f"Failed to open image: {path_to_image}. Error: {e}. Skipped")
            continue
        try:
            bbs = read_txt_content(path_to_txt)
        except Exception as e:
            print(f"Failed while reading txt content: {path_to_txt}. Error: {e}. Skipped")
            continue

        draw_bb(image, bbs)
        cv2.putText(image, os.path.basename(path_to_image), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        cv2.imshow("", image)
        cv2.waitKey(0)

    assert not path_to_images and not path_to_txts, "Something went wrong. Didnt process all pairs"
    return


def main():
    args = read_args()
    assert os.path.isdir(args.folder), "Wrong input provided. Folder expected"
    path_to_images, path_to_txts = collect_img_txt_pairs(args.folder)
    visualise_images(path_to_images, path_to_txts)


if __name__ == "__main__":
    main()
