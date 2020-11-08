import torchvision.transforms as transforms
import torch.nn.functional as F
from typing import List
from .tool.utils import *
from .tool.darknet2pytorch import Darknet
import os
import torch
import numpy as np
import cv2


WEIGHTS_VERSION = 4


class YOLOv4Tiny:
    path_to_dependencies = os.path.join(
        os.getcwd(), "logo_detector", "yolov4_tiny", "dependencies",
        f"weights_spotiq_v{WEIGHTS_VERSION}"
    )
    dependencies = "yolov4_tiny"
    conf_thresh = 0.1
    NMS_thresh = 0.1

    def __init__(
            self,
            device: str = "gpu",
            weights: str = None,
            cfg: str = None,
            txt: str = None,
            conf: float = None,
            nms: float = None
    ):
        if cfg:
            config_path = cfg
        else:
            config_path = os.path.join(
                self.path_to_dependencies, self.dependencies + ".cfg"
            )
        if weights:
            weights_path = weights
        else:
            weights_path = os.path.join(
                self.path_to_dependencies, self.dependencies + ".weights"
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
        self.model_name = "YOLOv4Tiny"

        if device != "cpu":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        # Initialize the model
        try:
            self.model: Darknet = Darknet(config_path)
        except Exception as e:
            print(f"Failed to initialize Darknet. Error: {e}")
            raise e
        print(f"Darknet initialized with net resolution of:"
              f" {self.model.height} {self.model.width}")

        # Load model's weights
        try:
            self.model.load_weights(weights_path)
            print("Loaded weights:", weights_path)
        except Exception as e:
            print(f"Failed to load model weights. Error: {e}")
            raise e

        # Load classes
        self.classes = load_class_names(classes_path)
        print(f"Total of {len(self.classes)} classes read:"
              f" {' '.join([str(e) for e in self.classes])}")

        # Move model to device and prepare for inference
        self.model.to(self.device).eval()
        print("YoloV4Tiny model successfully initialized\n")

    def predict(self, images: List[np.ndarray]) -> List[list]:
        """
        Receives a batch of images, preprocesses them, runs through the net,
        postprocesses detections and returns the results
        :param images:
        :return:
        """
        # Preprocess data: resize, normalization, batch etc
        images_ = self.preprocess_image_pipeline_v1(images)
        #images_ = self.preprocess_image_pipeline_v2(images)
        images_ = torch.autograd.Variable(images_)

        # Run data through the net
        with torch.no_grad():
            output = self.model(images_)

        del images_
        # Postprocess data: NMS, thresholding
        boxes = YOLOv4Tiny.postprocess_detection_results(
            output, self.NMS_thresh, self.conf_thresh
        )
        boxes = self.rescale_boxes(
            boxes, self.model.height, images[0].shape[:2]
        )

        return boxes

    def predict_with_custom_thresholds(
            self,
            images: List[np.ndarray],
            conf: float = None,
            nms: float = None
    ) -> List[list]:
        images_ = self.preprocess_image_pipeline_v1(images)
        images_ = torch.autograd.Variable(images_)
        with torch.no_grad():
            output = self.model(images_)
        del images_
        if conf:
            confidence = conf
        else:
            confidence = self.conf_thresh
        if nms:
            non_max_supr = nms
        else:
            non_max_supr = self.NMS_thresh
        print(f"Conf: {confidence}, NMS: {non_max_supr}")
        boxes = YOLOv4Tiny.postprocess_detection_results(
            output, non_max_supr, confidence
        )
        return self.rescale_boxes(
            boxes, self.model.height, images[0].shape[:2]
        )

    def preprocess_image_pipeline_v1(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocesses image(s) in accordance with the preprocessing steps taken
        during model training and collects them in batch
        :param image:
        :return:
        """
        # Preprocess images
        processed_images = list()
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image, dsize=(self.model.width, self.model.height)
            )
            image = torch.from_numpy(
                image.transpose(2, 0, 1)
            ).float().div(255).unsqueeze(0)
            processed_images.append(image)

        # Collect images in a batch
        try:
            batch_images = torch.cat(processed_images)
        except Exception as e:
            print("Failed to concat processed imgs into a tensor. Error: {e}")
            raise e

        # Move batch to the same device on which the model's sitting
        batch_images = batch_images.to(device=self.device)

        return batch_images

    def preprocess_image_pipeline_v2(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Another preprocessing approach that resizes image keeping its aspect
        ratio using padding
        :param images:
        :return:
        """
        processed_images = list()
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = YOLOv4Tiny.preprocess_image(
                image=image, image_size=self.model.height
            )
            processed_images.append(image)

        # Collect images in a batch
        try:
            batch_images = torch.cat(processed_images)
        except Exception as e:
            print("Failed to concat processed imgs into a tensor. Error: {e}")
            raise e

        # Move batch to the same device on which the model's sitting
        batch_images = batch_images.to(device=self.device)

        return batch_images

    @staticmethod
    def postprocess_detection_results(
            predictions: list,
            NMS_thresh: float,
            conf_thresh: float
    ) -> list:
        """
        Filters out poor detections made by the net by comparing them to the
        NMS and confidence thresholds
        :param predictions:
        :param NMS_thresh:
        :param conf_thresh:
        :return:
        """
        # [batch, num, 1, 4]
        box_array = predictions[0]
        # [batch, num, num_classes]
        confs = predictions[1]
        if type(box_array).__name__ != 'ndarray':
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        num_classes = confs.shape[2]
        # [batch, num, 4]
        box_array = box_array[:, :, 0]
        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]
            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = nms_cpu(ll_box_array, ll_max_conf, NMS_thresh)
                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3],
                                       ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

            bboxes_batch.append(bboxes)

        return bboxes_batch

    def rescale_boxes(
            self,
            boxes_for_batch: list,
            current_dim: int,
            original_shape: tuple
    ) -> list:
        """

        :param boxes_for_batch: list containing N (= batch size) lists (detections per image in the batch),
         which in turn contain lists representing detected objects if any: [ [[],[],...], [[], [], ...], ...]
        :param current_dim:
        :param original_shape:
        :return:
        """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x

        boxes_batch_rescaled = list()
        for boxes in boxes_for_batch:
            boxes_rescaled = list()
            for box in boxes:
                new_left = int(box[0] * orig_w)
                new_left = 2 if new_left <= 0 else new_left
                new_top = int(box[1] * orig_h)
                new_top = 2 if new_top <= 0 else new_top
                new_right = int(box[2] * orig_w)
                new_right = orig_w - 2 if new_right >= orig_w else new_right
                new_bot = int(box[3] * orig_h)
                new_bot = orig_h - 2 if new_bot >= orig_h else new_bot
                # new_left = int((box[0] - pad_x // 2) * (orig_w / unpad_w))
                # new_left = 2 if new_left == 0 else new_left
                # new_top = int((box[1] - pad_y // 2) * (orig_h / unpad_h))
                # new_top = 2 if new_top == 0 else new_top
                # new_right = int((box[2] - pad_x // 2) * (orig_w / unpad_w))
                # new_bot = int((box[3] - pad_y // 2) * (orig_h / unpad_h))
                obj_score = round(box[4], 4)
                conf = round(box[5], 4)
                index = self.classes[box[6]]
                boxes_rescaled.append(
                    [new_left, new_top, new_right, new_bot, obj_score, conf, index]
                )
            boxes_batch_rescaled.append(boxes_rescaled)

        return boxes_batch_rescaled

    @staticmethod
    def pad_to_square(img: torch.Tensor, pad_value: int) -> tuple:
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

        return img, pad

    @staticmethod
    def resize(image: torch.Tensor, size: int) -> torch.Tensor:
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest")
        return image

    @staticmethod
    def preprocess_image(image: np.ndarray, image_size: int = 608) -> torch.Tensor:
        image = transforms.ToTensor()(image)
        # Padd the image so that it is a square
        image, _ = YOLOv4Tiny.pad_to_square(image, 0)
        # Resize the image to the expected net's resolution
        image = YOLOv4Tiny.resize(image, image_size)

        return image
