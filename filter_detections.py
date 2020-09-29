import os

import cv2

from logo_detector.yolov5 import YOLOv5


def calculate_accuracies(detections: dict) -> list:
    summed_detections = list()
    for cls, obj_scores in detections.items():
        sum_ = round(sum(obj_scores), 4)
        summed_detections.append((cls, sum_))

    return summed_detections


if __name__ == "__main__":
    model = YOLOv5()
    print("[INFO]: Model loaded")
    test_dir = r"D:\SingleView\SpotIQ\tests\RETAIL\cropped_test_ads"
    save_path = r"D:\SingleView\SpotIQ\DELETE_ME"
    BATCH_SIZE = 15

    total, total_correct_classifications = 0, 0
    for element in os.listdir(test_dir):
        print(f"\n--------- Testing class: {element.upper()} ---------")
        path_to_folder = os.path.join(test_dir, element)
        for video in os.listdir(path_to_folder):
            path_to_video = os.path.join(path_to_folder, video)
            cap = cv2.VideoCapture(path_to_video)
            if not cap.isOpened():
                print("[ERROR]: Failed to open video:", path_to_video)
                continue
            print("Processing video:", path_to_video)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out_name = os.path.join(
                save_path, os.path.splitext(video)[0] + "_out.avi"
            )
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                out_name, fourcc, fps, (frame_width, frame_height), True
            )
            # PROCESS VIDEO
            batch = list()
            to_break = False
            detections = dict()
            while True:
                if len(batch) < BATCH_SIZE:
                    success, frame = cap.read()
                    if not success:
                        to_break = True
                    else:
                        batch.append(frame)
                        continue
                if len(batch):
                    predictions = model.predict(batch)
                    for preds, image in zip(predictions, batch):
                        for pred in preds:
                            left, top, right, bot, obj_score, cls = pred
                            cv2.rectangle(
                                image, (int(left), int(top)),
                                (int(right), int(bot)),
                                (0, 255, 0), 3
                            )
                            cv2.putText(image, f"{cls}_{round(obj_score, 4)}",
                                        (int(left), int(top) + 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2
                            )
                            if cls not in detections.keys():
                                detections[cls] = list()
                            detections[cls].append(obj_score)

                        cv2.imshow("", image)
                        writer.write(image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                if to_break:
                    break
                batch = list()

            # ESTIMATE MODEL PERFORMANCE
            if detections:
                weighted_detections = calculate_accuracies(detections)
                weighted_detections.sort(key=lambda x: x[-1], reverse=True)
                if len(weighted_detections) > 2:
                    top = weighted_detections[:2]
                    print(f"Truth class: {element}; Top 2 preds: {top}")
                else:
                    top = weighted_detections
                    print(f"Truth class: {element}; "
                          f"Top 2 preds: {weighted_detections}")
                if top[0][0] == element:
                    total_correct_classifications += 1

            total += 1
            cap.release()
            cv2.destroyAllWindows()

    print(f"\n\n==> Total: {total}, corrects: {total_correct_classifications}")
