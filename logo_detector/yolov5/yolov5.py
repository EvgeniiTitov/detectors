import os
import time
import sys
search_path = r"C:\\Users\\Evgenii\\logo_detector\\logo_detector\\yolov5"
if not search_path in sys.path:
    sys.path.append(search_path)

import torch
from typing import List, Tuple
import numpy as np
import cv2

from .models.experimental import attempt_load
from logo_detector.abstract_model import AbstractModel


os.environ['KMP_DUPLICATE_LIB_OK']='True'


class YOLOv5(AbstractModel):
    path_to_dependencies = os.path.join(
        os.getcwd(), "logo_detector", "yolov5", "dependencies", "run9"
    )
    dependencies = "yolov5"
    conf_thresh = 0.2
    NMS_thresh = 0.1

    def __init__(
            self,
            device: str = "gpu",
            weights: str = None,
            txt: str = None,
            conf: float = None,
            nms: float = None,
            img_size: int = 800
    ):
        if weights:
            weights_path = weights
        else:
            weights_path = os.path.join(
                self.path_to_dependencies, self.dependencies + ".pt"
            )
        if txt:
            classes_path = txt
        else:
            classes_path = os.path.join(
                self.path_to_dependencies, self.dependencies + ".txt"
            )
        if conf:
            self.conf_thresh = conf
        if nms:
            self.NMS_thresh = nms
        self.model_name = "YOLOv5"

        if device != "cpu":
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        try:
            self.model = attempt_load(weights_path, map_location=self.device)
            self.model.eval()
            print("v5 model initialized")
        except Exception as e:
            print(f"Failed while loading the v5 model. Error: {e}")
            raise e

        self.image_size = img_size

        # TODO: Check half precision
        self.half = False

        try:
            self.classes = self.model.classes
            assert self.classes is not None and len(self.classes)
        except:
            self.classes = YOLOv5.load_class_names(classes_path)
        print("[INFO]: Detecting classes: ", " ".join(self.classes))

    def predict(self, images: List[np.ndarray]) -> List[list]:
        # Preprocess images
        batch, original_shapes = self.preprocess_images(images)
        # Run model
        preds = self.model(batch)[0]

        # Postprocess predictions
        preds = self.non_max_suppression(
            preds, self.conf_thresh, self.NMS_thresh
        )
        # Rescale boxes relatively to the original image shapes
        assert len(preds) == len(original_shapes) == len(batch)
        preds = self.rescale_batch_coords(preds, original_shapes, batch)

        return preds

    def preprocess_images(
            self,
            images: List[np.ndarray]
    ) -> Tuple[torch.Tensor, list]:
        batch = torch.zeros(
            (len(images), 3, self.image_size, self.image_size)
        )
        original_shapes = list()
        for i, image in enumerate(images):
            original_shapes.append(image.shape[:2])
            image = self.letterbox(
                image, new_shape=self.image_size, auto=False
            )[0]
            image = image[:, :, ::-1].transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(self.device)
            image = image.half() if self.half else image.float()
            image /= 255.0
            batch[i] = image
        batch = batch.to(self.device)
        return batch, original_shapes

    def letterbox(
            self,
            img,
            new_shape=(640, 640),
            color=(114, 114, 114),
            auto=True,
            scaleFill=False,
            scaleup=True
    ) -> tuple:
        # Resize image to a 32-pixel-multiple rectangle
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # wh padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            # width, height ratios
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return img, ratio, (dw, dh)

    def non_max_suppression(
            self,
            prediction,
            conf_thres=0.1,
            iou_thres=0.6,
            merge=False,
            classes=None,
            agnostic=False
    ):
        """
        Performs Non-Maximum Suppression (NMS) on inference results
        Returns:
             detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        nc = prediction[0].shape[1] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
        t = time.time()
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, j + 5, None], j[:, None].float()), 1
                )
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat(
                    (box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes:
                x = x[
                    (x[:, 5:6] == torch.tensor(classes, device=x.device)).any(
                        1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Sort by confidence
            # x = x[x[:, 4].argsort(descending=True)]

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:,
                                          4]  # boxes (offset by class), scores
            i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (
                    1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights,
                                        x[:, :4]).float() / weights.sum(1,
                                                                        keepdim=True)  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy
                except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                    print(x, i, x.shape, i.shape)
                    pass

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        # where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2):
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(
            box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        # iou = inter / (area1 + area2 - inter)
        return inter / (area1[:, None] + area2 - inter)

    def rescale_batch_coords(
            self,
            predictions: List[torch.Tensor],
            original_shapes: List[tuple],
            batch: torch.Tensor
    ) -> List[list]:
        out = list()
        for i, (preds, origin_shape) in enumerate(zip(predictions, original_shapes)):
            if preds is not None and len(preds):
                img = batch[i]
                preds[:, :4] = self.scale_coords(
                    img.shape[1:], preds[:, :4], origin_shape
                )
                preds_ = preds.tolist()
                for i in range(len(preds_)):
                    preds_[i][-1] = self.classes[int(preds_[i][-1])]
                out.append(preds_)
            else:
                out.append([])

        return out

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, img_shape) -> None:
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    @staticmethod
    def load_class_names(namesfile) -> List[str]:
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names
