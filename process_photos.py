import os
import cv2
from logo_detector.yolov5 import YOLOv5


model = YOLOv5()
data_to_process = r"D:\SingleView\SpotIQ\tests\CHANNELS\cropped_frames"
save_path = r"D:\SingleView\SpotIQ\tests\RESULTS\CHANNELS\frames_results"

for item in os.listdir(data_to_process):
    path_to_image = os.path.join(data_to_process, item)
    image = cv2.imread(path_to_image)
    if image is None:
        print("Failed to open:", item)
        continue

    detections = model.predict([image])[0]
    for pred in detections:
        left, top, right, bot, conf, cls = pred
        cv2.rectangle(image, (int(left), int(top)),
                      (int(right), int(bot)), (0, 255, 0), 2)
        cv2.putText(image, f"{cls}_{round(conf, 3)}", (int(left), int(top) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imwrite(
        os.path.join(save_path, item), image
    )
    print("Processed: item")