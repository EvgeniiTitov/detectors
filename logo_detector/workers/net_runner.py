import threading
import queue


class NetRunner(threading.Thread):
    def __init__(
            self,
            in_q: queue.Queue,
            out_q: queue.Queue,
            model,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_q = in_q
        self.out_q = out_q
        self.model = model
        print("Net runner started")

    def run(self) -> None:
        while True:
            input_ = self.in_q.get()

            if input_ == "STOP":
                break
            elif input_ == "END":
                self.out_q.put("END")
                continue

            file_id, batch = input_
            detections = self.model.predict(images=batch)

            self.out_q.put((file_id, batch, detections))

        self.out_q.put("STOP")
        print("Net runner killed")