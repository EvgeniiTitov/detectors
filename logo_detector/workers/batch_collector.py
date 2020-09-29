import threading
import cv2
import os
import queue


class BatchCollector(threading.Thread):
    def __init__(
            self,
            batch_size: int,
            in_q: queue.Queue,
            out_q: queue.Queue,
            progress: dict,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.in_q = in_q
        self.out_q = out_q
        self.progress = progress
        print("[INFO]: Batch collector thread started")

    def run(self) -> None:
        while True:
            input_ = self.in_q.get()
            if input_ == "STOP":
                break

            file_id = input_
            file_path = self.progress[file_id]["file_path"]
            file_type = self.progress[file_id]["filetype"]
            if not os.path.exists(file_path):
                print(f"Failed to file {file_path} on disk")
                continue

            try:
                cap = cv2.VideoCapture(file_path)
            except Exception as e:
                print(f"Failed to open: {file_path}. Error: {e}")
                continue
            if not cap.isOpened():
                continue

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_seconds = total_frames // fps

            self.progress[file_id]["frame_width"] = frame_width
            self.progress[file_id]["frame_height"] = frame_height
            self.progress[file_id]["fps"] = fps
            self.progress[file_id]["total_frames"] = total_frames
            self.progress[file_id]["total_seconds"] = total_seconds
            self.progress[file_id]["status"] = "Processing"

            batch_frames = list()
            to_break = False
            while True:
                if file_type == "video" and len(batch_frames) < self.batch_size:
                    has_frame, frame = cap.read()
                    if not has_frame:
                        to_break = True
                    else:
                        batch_frames.append(frame)
                        continue
                elif file_type == "image":
                    has_frame, frame = cap.read()
                    if has_frame:
                        batch_frames.append(frame)
                    to_break = True

                if len(batch_frames):
                    self.out_q.put((file_id, batch_frames))

                if to_break:
                    break

                batch_frames = list()

            cap.release()
            self.out_q.put("END")

        self.out_q.put("STOP")
        print("Batch collector thread killed")
