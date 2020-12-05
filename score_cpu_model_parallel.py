import os
import multiprocessing
import argparse
from typing import List
import sys

import numpy as np
import cv2

from logo_detector import YOLOv3Model, YOLOv4Tiny, YOLOv5, YOLOv5ONNX


'''
Since we're leaning towards smaller models to avoid VMs with TPUs, here 
is the script to score the model on CPU using multiprocessing
'''
every_Nth_frame = 4


def read_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=["v3", "v4", "v5", "v5ONNX"],
                        help="Model to score")
    parser.add_argument("-v", "--videos", type=str,
                        help="Path to videos to use for scoring")
    parser.add_argument("-s", "--save_path", type=str,
                        help="Path to save processed data")
    return vars(parser.parse_args())


def collect_videos_to_process(dirpath: str) -> List[str]:
    return [
        os.path.join(dirpath, e) for e in os.listdir(dirpath)
        if os.path.splitext(e)[-1].lower() in [".mp4", ".avi", ".flv"]
    ]


def draw_boxes(boxes: List[list], image: np.ndarray) -> None:
    for box in boxes:
        cls_, x1, y1, x2, y2, conf = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (203, 192, 255), 3)
        cv2.putText(image, f"{cls_}_{conf}", (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def score_model(arguments: dict) -> None:
    pid = os.getpid()
    video_path = arguments["video"]
    model_to_score = arguments["model"]
    save_path = arguments["save_path"]

    if not os.path.exists(video_path):
        print(f"[ERROR]: Process {pid} failed to find its video {video_path}")
        sys.exit(1)
    try:
        if model_to_score == "v3":
            print(f"[INFO]: Process {pid} loading v3 ...")
            model = YOLOv3Model(device="cpu")
        elif model_to_score == "v4":
            print(f"[INFO]: Process {pid} loading v4 ...")
            model = YOLOv4Tiny(device="cpu")
        elif model_to_score == "v5":
            print(f"[INFO]: Process {pid} loading v5 ...")
            model = YOLOv5(device="cpu")
        elif model_to_score == "v5ONNX":
            print(f"[INFO]: Process {pid} loading v5ONNX...")
            model = YOLOv5ONNX()
        else:
            raise Exception("Unknown model name provided")
    except Exception as e:
        print(f"[ERROR]: Process {pid} failed while loading the model. "
              f"Error: {e}")
        sys.exit(1)
    print(f"[INFO]: Process {pid} successfully loaded {model_to_score}!")

    try:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()
    except Exception as e:
        print(f"[ERRO]: Process {pid} failed to open the cap. Error {e}")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = int(total_frames / fps)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_name = os.path.splitext(os.path.basename(video_path))[0]
    try:
        video_writer = cv2.VideoWriter(
            os.path.join(save_path, out_name + "_out.avi"), fourcc, fps,
            (frame_width, frame_height), True
        )
    except Exception as e:
        print(f"[ERROR]: Process {pid} failed to init videowriter. Error: {e}")
        sys.exit(1)

    frames_passed, seconds_passed = 0, 0
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        frames_passed += 1
        if frames_passed == fps:
            seconds_passed += 1
            frames_passed = 0
        if not frames_passed % every_Nth_frame == 0:
            continue

        detections = model.predict(frame)
        if len(detections):
            print(f"Process {pid} detected:"
                  f" {' '.join([e[0] for e in detections])}")
            draw_boxes(detections, frame)
        video_writer.write(frame)
        if (
                seconds_passed % 10 == 0 and
                seconds_passed != 0 and
                frames_passed == 0
        ):
            print(f"Process {pid} processed {seconds_passed} /"
                  f" {total_seconds} seconds")


def main():
    args = read_args()
    if not os.path.exists(args["save_path"]):
        os.mkdir(args["save_path"])

    video_paths = set(collect_videos_to_process(args["videos"]))
    if not len(video_paths):
        raise Exception("No videos to process found in", args["videos"])
    arguments = {
        "video": str,
        "model": args["model"],
        "save_path": args["save_path"]
    }
    to_distribute = []
    for video_path in video_paths:
        payload = arguments.copy()
        payload["video"] = video_path
        to_distribute.append(payload)

    cores = multiprocessing.cpu_count()
    workers = cores * 1.2
    if len(to_distribute) < workers:
        workers = len(to_distribute)
    with multiprocessing.Pool(processes=workers) as p:
        p.map(score_model, to_distribute)


if __name__ == "__main__":
    main()
