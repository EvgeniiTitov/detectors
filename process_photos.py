import os
import cv2
import argparse
import time

from logo_detector.yolov5 import YOLOv5
from logo_detector.yolov5 import YOLOv5ONNX


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str,
                        help="Path to photos to process")
    parser.add_argument("save_path", type=str,
                        help="Path where processed photos get saved")
    return vars(parser.parse_args())


def main():
    args = parse_arguments()
    if not os.path.exists(args["save_path"]):
        os.mkdir(args["save_path"])

    model = YOLOv5ONNX()
    total_time, images_processed = 0, 0
    for item in os.listdir(args["folder"]):
        path_to_image = os.path.join(args["folder"], item)
        image = cv2.imread(path_to_image)
        if image is None:
            print("Failed to open:", item)
            continue
        s_time = time.time()
        detections = model.predict(image)
        total_time += time.time() - s_time
        for pred in detections:
            print("PRED:", pred)
            cls_, left, top, right, bot, conf = pred
            cv2.rectangle(image, (int(left), int(top)),
                          (int(right), int(bot)), (0, 255, 0), 2)
            cv2.putText(image, f"{cls_}_{round(conf, 3)}",
                        (int(left), int(top) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.imwrite(
            os.path.join(args["save_path"], item), image
        )
        print("Processed:", item)
        images_processed += 1
    print(f"\n\nAVERAGE PROCESSING TIME PER IMAGE: "
          f"{round(total_time / images_processed, 4)} seconds")


if __name__ == "__main__":
    main()
