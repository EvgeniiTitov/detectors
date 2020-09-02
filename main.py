import os

import uuid
import queue
import argparse
import cv2

from logo_detector.yolov4_tiny import YOLOv4Tiny
import logo_detector.utils.results_processor as res_procesor
from logo_detector.workers import BatchCollector, NetRunner, Writer


ALLOWED_EXTS = [".jpg", ".png", ".jpeg", ".mp4", ".avi", ".flv"]
VIDEO_EXTS = [".mp4", ".avi", ".flv"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True,
                        help="Path to data to process")
    parser.add_argument("-s", "--save_path", default=r"D:\Desktop\system_output",
                        help="Path to where store processed data")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size for video processing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    arguments = parser.parse_args()

    return arguments


class Detector:
    def __init__(
            self,
            save_path: str,
            batch_size: int,
            device: str = "cuda"
    ):
        self.save_path = save_path
        self.progress = dict()

        self.files_to_process_Q = queue.Queue()
        self.reader_to_nn_Q = queue.Queue(maxsize=3)
        self.nn_to_writer_Q = queue.Queue(maxsize=4)

        try:
            model = YOLOv4Tiny(device=device)
        except Exception as e:
            print("Failed to initialize the model. Error:", e)
            raise e

        self.threads = list()
        self.batch_collector = BatchCollector(
            batch_size=batch_size,
            in_q=self.files_to_process_Q,
            out_q=self.reader_to_nn_Q,
            progress=self.progress
        )
        self.threads.append(self.batch_collector)

        self.net_runner = NetRunner(
            in_q=self.reader_to_nn_Q,
            out_q=self.nn_to_writer_Q,
            model=model
        )
        self.threads.append(self.net_runner)

        self.writer = Writer(
            save_path=save_path,
            in_q=self.nn_to_writer_Q,
            progress=self.progress,
            result_processor=res_procesor
        )
        self.threads.append(self.writer)

        self.start()

    def process(self, path_to_data: str) -> dict:
        assert os.path.isdir(path_to_data)
        file_ids = dict()
        for filename in os.listdir(path_to_data):
            ext = os.path.splitext(filename)[-1].lower()
            if ext not in ALLOWED_EXTS:
                print(f"Cannot process {filename}. Unsupported file extension")
                continue

            filetype = "video" if ext in VIDEO_EXTS else "photo"
            file_id = str(uuid.uuid4())
            self.progress[file_id] = {
                "status": "Awaiting processing",
                "filetype": filetype,
                "file_path": os.path.join(path_to_data, filename),
                "frame_width": None,
                "frame_height": None,
                "fps": None,
                "total_frames": None,
                "total_seconds": None,
                "processed_frames": 0
            }
            file_ids[filename] = file_id
            self.files_to_process_Q.put(file_id)

        self.files_to_process_Q.put("STOP")
        return file_ids

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.files_to_process_Q.put("STOP")
        for thread in self.threads:
            thread.join()


def process_files(
        model: YOLOv4Tiny,
        path_to_data: str,
        save_path: str,
        batch_size: int,
        show_frames: bool = False
) -> None:
    assert batch_size > 0, "Batch size cannot be negative"
    for file in os.listdir(path_to_data):
        if not os.path.splitext(file)[-1].lower() in ALLOWED_EXTS:
            continue

        print("\nProcessing:", file)
        cap = cv2.VideoCapture(os.path.join(path_to_data, file))
        if not cap.isOpened():
            print("Failed to create a cap for:", path_to_data)
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video resolution: {frame_width} {frame_height}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = int(total_frames / fps)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_name = os.path.join(save_path, "out_" + file) if not os.path.splitext(file)[-1].lower() in VIDEO_EXTS \
                                                        else os.path.join(save_path, os.path.splitext(file)[0] + ".avi")
        video_writer = cv2.VideoWriter(
            out_name, fourcc, fps, (frame_width, frame_height), True
        )

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

            if show_frames:
                for frame in batch:
                    cv2.imshow("", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if to_break:
                break
            batch = list()

        success = res_procesor.create_log_file(
            payload=log, save_path=save_path, filename=file
        )
        if not success:
            print("Failed to create log file for:", file)

        cap.release()
        video_writer.release()

    return


def main():
    args = parse_arguments()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    detector = Detector(
        save_path=args.save_path,
        batch_size=int(args.batch_size),
        device=args.device
    )
    ids = detector.process(args.folder)
    detector.stop()
    print("\nFile ids:")
    for k, v in ids.items():
        print(k, v)


if __name__ == "__main__":
    main()
