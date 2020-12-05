import os
import argparse

from logo_detector.detector import Detector
from logo_detector.yolov5 import YOLOv5


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--videos", required=True,
                        help="Path to videos to process")
    parser.add_argument("-s", "--save_path",
                        default=r"D:\Desktop\system_output",
                        help="Path to where store processed data")
    parser.add_argument("--batch_size", type=int, default=40,
                        help="Batch size for video processing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")

    return vars(parser.parse_args())


def main() -> None:
    args = parse_arguments()
    if not os.path.exists(args["save_path"]):
        os.mkdir(args["save_path"])
    yolov5_onnx = YOLOv5(
        weights=r"logo_detector\yolov5\dependencies\run10\v5.pt",
        txt=r"logo_detector\yolov5\dependencies\run10\v5.txt",
        device=args["device"]
    )
    d = Detector(
        save_path=args["save_path"], batch_size=int(args["batch_size"]),
        model=yolov5_onnx
    )
    ids = d.process(args["videos"])
    for k, v in ids.items():
        print(k, v)
    d.stop()


if __name__ == "__main__":
    main()
