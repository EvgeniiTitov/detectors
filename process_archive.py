from typing import List
from logo_detector.yolov4_tiny import YOLOv4Tiny
from logo_detector.utils.read_targz_archive import extract_archive_content
from logo_detector.utils.json_generator import JSONOutputGenerator
import os
import cv2
import argparse


TEMP_STORAGE_PATH = r"D:\Desktop\system_output\single_view_output\archive_content"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=str, help="Path to archive to process")
    parser.add_argument("--save_path", required=True, type=str, help="Path where JSON's will be saved")
    arguments = parser.parse_args()

    return arguments


def get_paths_to_images_within_archive(folder: str, paths: list) -> list:
    for filename in os.listdir(folder):
        path_to_file = os.path.join(folder, filename)
        if os.path.isdir(path_to_file):
            get_paths_to_images_within_archive(path_to_file, paths)
        elif os.path.isfile(path_to_file):
            if os.path.splitext(filename)[-1].lower() == ".png":
                paths.append(path_to_file)

    return paths


def process_images(model: YOLOv4Tiny, paths_to_images: List[str], json_gen: JSONOutputGenerator) -> None:
    detection_results = dict()
    for path_to_image in paths_to_images:
        image_name = os.path.basename(path_to_image)
        image = cv2.imread(path_to_image)
        if image is None:
            print("Failed to read image:", path_to_image)
            continue

        detection = model.predict([image])
        detection_results[image_name] = detection[0]

    formatted_output = json_gen.generate_output_json(detection_results)
    _ = json_gen.dump_json(payload=formatted_output, name="randomname")

    return


def main():
    args = parse_arguments()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    model = YOLOv4Tiny(device="cpu")
    json_gen = JSONOutputGenerator(save_path=args.save_path)
    extract_archive_content(path_to_archive=args.archive, save_path=TEMP_STORAGE_PATH)
    paths_to_images = list()
    paths_to_images = get_paths_to_images_within_archive(TEMP_STORAGE_PATH, paths_to_images)
    process_images(model, paths_to_images, json_gen)


if __name__ == "__main__":
    main()
