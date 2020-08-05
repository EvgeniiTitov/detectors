from yolov4_tiny import YOLOv4Tiny
import utils.results_processor as res_procesor
import argparse
import cv2
import os


ALLOWED_EXTS = [".jpg", ".png", ".jpeg", ".mp4", ".avi", ".flv"]
VIDEO_EXTS = [".mp4", ".avi", ".flv"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, help="Path to data to process")
    parser.add_argument("-s", "--save_path", default=r"D:\Desktop\system_output",
                        help="Path to where store processed data")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for video processing")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    arguments = parser.parse_args()

    return arguments


def process_files(model: YOLOv4Tiny, path_to_data: str, save_path: str, batch_size) -> None:
    assert batch_size > 0, "Batch size cannot be negative"
    for file in os.listdir(path_to_data):
        if not os.path.splitext(file)[-1].lower() in ALLOWED_EXTS:
            continue
        print("Processing:", file)
        cap = cv2.VideoCapture(os.path.join(path_to_data, file))
        if not cap.isOpened():
            print("Failed to create a cap for:", path_to_data)
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = int(total_frames / fps)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_name = os.path.join(save_path, "out_" + file) if not os.path.splitext(file)[-1].lower() in VIDEO_EXTS \
                                                        else os.path.join(save_path, os.path.splitext(file)[0] + ".avi")
        video_writer = cv2.VideoWriter(out_name, fourcc, fps, (frame_width, frame_height), True)
        batch = list()
        to_break = False
        frames_passed, seconds_passed = 0, 0
        log = {}
        while True:
            if len(batch) < batch_size:
                has_frame, frame = cap.read()
                if has_frame:
                    batch.append(frame)
                    frames_passed += 1
                    if frames_passed == fps:
                        seconds_passed += 1
                        frames_passed = 0
                        print(f"File: {file}. Processed {seconds_passed} \ {total_seconds} seconds")
                    continue
                else:
                    to_break = True

            if len(batch) > 0:
                detections = model.predict(batch)
                detected_classes = res_procesor.draw_bb_for_batch_remember_detected_classes(
                    images=batch,
                    boxes=detections
                )
                # If any objects detected, remember them for the log file
                if detected_classes:
                    if seconds_passed not in log.keys():
                        log[seconds_passed] = set()
                    log[seconds_passed].update(detected_classes)
                res_procesor.save_batch_on_disk(images=batch, video_writer=video_writer)

            for frame in batch:
                cv2.imshow("", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if to_break:
                break
            batch = list()

        success = res_procesor.create_log_file(payload=log, save_path=save_path, filename=file)
        if not success:
            print("Failed to create log file for:", file)

        cap.release()
        video_writer.release()

    return


def main():
    args = parse_arguments()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    batch_size = args.batch_size
    model = YOLOv4Tiny(device=args.device)
    process_files(model, args.folder, args.save_path, batch_size)


if __name__ == "__main__":
    main()
