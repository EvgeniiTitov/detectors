import os
import threading
from threading import Thread
from typing import List
import time

import cv2
import numpy as np

from logo_detector.yolov5 import YOLOv5


class StreamsProcessor:
    # TODO: How not to skip frames? Like do not overwrite list values? Check if
    #       the previous frame is still there, dont put a new one there.

    def __init__(self, sources: List[cv2.VideoCapture]):
        self.frames = [None] * len(sources)
        self.n_sources = len(sources)
        for i, cap in enumerate(sources):
            assert cap.isOpened()
            t = Thread(target=self.read, args=(i, cap, 0.01), daemon=True)
            t.start()

    def read(self, i: int, cap: cv2.VideoCapture, sleep: int) -> None:
        c = 0
        while cap.isOpened():
            c += 1
            cap.grab()
            if c == 4:
                _, self.frames[i] = cap.retrieve()
                #print(f"Thread {threading.get_ident()} put a frame at i = {i}")
                c = 0
            time.sleep(sleep)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        self.n += 1
        # If q is pressed, or all videos processed
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration

        return self.frames.copy()


class StreamsWriter(Thread):
    # SUCKS
    def __init__(self, source_metas: dict, save_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.frames = [None] * len(source_metas.keys())
        for i, (video_name, [h, w, fps]) in enumerate(source_metas.items()):
            name_no_ext = os.path.splitext(video_name)[0]
            dest_path = os.path.join(save_path, name_no_ext + "_out.avi")
            writer = cv2.VideoWriter(dest_path, fourcc, fps, (w, h), True)
            t = Thread(target=self.write_frame, args=(i, writer), daemon=True)
            t.start()

    def save_frames(self, frames_to_save: List[np.ndarray]) -> None:
        for i, frame in enumerate(frames_to_save):
            if frame is not None:
                self.frames[i] = frame

    def write_frame(self, i: int, writer: cv2.VideoWriter) -> None:
        # Set index to None once saved
        while True:
            if cv2.waitKey(1) == ord("q"):
                break

            if self.frames[i] is None:
                time.sleep(0.01)
                continue
            if self.frames[i] == "DONE":
                break

            frame = self.frames[i].copy()
            if frame is not None:
                writer.write(frame)
        print(f"Writer thread {threading.get_ident()} stopped")

    def stop_writers(self) -> None:
        for i in range(len(self.frames)):
            self.frames[i] = "DONE"

if __name__ == "__main__":
    model = YOLOv5()
    videos = [
            r"D:\SingleView\SpotIQ\tests\1.flv",
            r"D:\SingleView\SpotIQ\tests\2.flv",
            r"D:\SingleView\SpotIQ\tests\3.flv",
            r"D:\SingleView\SpotIQ\tests\4.flv",
            r"D:\SingleView\SpotIQ\tests\5.flv",
            r"D:\SingleView\SpotIQ\tests\6.flv"
        ]
    SAVE_PATH = r""

    caps = list()
    caps_meta = dict()
    for video in videos:
        video_name = os.path.basename(video)
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print("Failed to open video:", video_name)
            continue
        caps.append(cap)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        caps_meta[video_name] = [frame_height, frame_width, fps]
    windows = [f"window{i}" for i in range(len(videos))]
    proc = StreamsProcessor(sources=caps)
    writer = StreamsWriter(source_metas=caps_meta, save_path=SAVE_PATH)

    for imgs in proc:
        batch, received_frames = list(), list()
        for img, window in zip(imgs, windows):
            if img is not None:
                batch.append(img)
                received_frames.append(window)
        # If no batch, nothing to show anyways
        if len(batch):
            predictions = model.predict(batch)
            for preds, img, window in zip(predictions, batch, received_frames):
                for pred in preds:
                    left, top, right, bot, obj_score, cls = pred
                    cv2.rectangle(
                        img=img, pt1=(int(left), int(top)),
                        pt2=(int(right), int(bot)),
                        color=(0, 255, 0), thickness=3
                    )
                    label = f"{cls}_{round(obj_score, 3)}"
                    cv2.putText(
                        img, label, (int(left), int(top) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                if len(preds):
                    print(f"Window: {window}. Detections:"
                          f" {' '.join([e[-1] for e in preds])}")
                cv2.imshow(window, cv2.resize(img, dsize=(640, 480)))

        # TODO: Do it sequentially
        writer.save_frames(imgs)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.stop_writers()
