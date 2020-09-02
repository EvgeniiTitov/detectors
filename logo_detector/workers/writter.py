import os
import threading
import queue

import cv2


class Writer(threading.Thread):
    def __init__(
            self,
            save_path: str,
            in_q: queue.Queue,
            progress: dict,
            result_processor,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_q = in_q
        self.result_processor = result_processor
        self.progress = progress
        self.save_path = save_path
        if not os.path.exists(save_path):
            try:
                os.mkdir(save_path)
            except Exception as e:
                print("Failed to create sav dir. Error:", e)
                raise e
        self.video_writter = None
        self.previous_id = None
        self.log = dict()
        self.total_seconds, self.processed_frames, self.fps = 0, 0, 0
        print("Writer started")

    def run(self) -> None:
        while True:
            input_ = self.in_q.get()

            if input_ == "STOP":
                break
            elif input_ == "END":
                self.refresh_video_writer()
                self.refresh_log()
                self.refresh_frame_counters()
                continue

            file_id, batch, detections = input_

            filetype = self.progress[file_id]["filetype"]
            filename = os.path.splitext(
                os.path.basename(self.progress[file_id]["file_path"])
            )[0]

            if self.previous_id != file_id:
                self.previous_id = file_id

            if filetype == "video" and not self.video_writter:
                out_path = os.path.join(self.save_path, filename + "_out.avi")

                self.fps = self.progress[file_id]["fps"]
                self.total_seconds = self.progress[file_id]["total_seconds"]

                w = self.progress[file_id]["frame_width"]
                h = self.progress[file_id]["frame_height"]
                self.create_video_writer(out_path, (w, h))

            if len(detections):
                self.result_processor.draw_bb_for_batch_remember_detected_classes(
                    images=batch,
                    boxes=detections
                )
                self.result_processor.save_batch_on_disk(
                    images=batch,
                    video_writer=self.video_writter
                )

            if filetype == "video":
                self.processed_frames += len(batch)
                print(f"Processed {int(self.processed_frames / self.fps)} /"
                      f" {self.total_seconds} seconds")

        print("Writer killed")

    def refresh_video_writer(self):
        self.video_writter = None

    def create_video_writer(self, store_path: str, dim: tuple):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        try:
            self.video_writter = cv2.VideoWriter(
                store_path, fourcc, self.fps, dim, True
            )
        except Exception as e:
            print(f"Failed while creating video writer. Error: {e}")
            raise e

    def refresh_log(self):
        self.log = dict()

    def refresh_frame_counters(self):
        self.total_seconds = 0
        self.processed_frames = 0
        self.fps = 0
