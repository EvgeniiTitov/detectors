from yolov3.models import Darknet
from yolov3.utils.utils import load_classes
from torch.autograd import Variable
import torch
import os


class YOLOv3Model:
    path_to_dependencies = os.path.join(os.getcwd(), "yolov3", "dependencies")
    dependencies = "yolov3_v1"
    net_resolution = 608
    confidence_thresh = 0.8
    NMS_thresh = 0.3

    def __init__(self, device: str = "gpu"):
        print("\nInitializing YOLOv3 model")
        config_path = os.path.join(self.path_to_dependencies, self.dependencies + ".cfg")
        weights_path = os.path.join(self.path_to_dependencies, self.dependencies + ".weights")
        classes_path = os.path.join(self.path_to_dependencies, self.dependencies + ".txt")

        self.device = device
        if device != "cpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # Initialize the model
        try:
            self.model: Darknet = Darknet(config_path, self.net_resolution)
            self.model.to(self.device)
        except Exception as e:
            print(f"Failed to initialize darknet. Error: {e}")
            raise e
        print("\nDarknet initialized")

        # Load the weights
        if weights_path.endswith(".weights"):
            self.model.load_darknet_weights(weights_path=weights_path)
        else:
            self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        print("Weights loaded")

        # Load classes getting detected
        try:
            self.classes = load_classes(classes_path)
        except Exception as e:
            print("Failed while reading classes txt. Error: {e}")
            raise e
        print("Classes txt parsed")

    def predict(self, batch: torch.Tensor) -> list:
        """

        :param batch:
        :return:
        """
        if self.device == "cuda" and not batch.is_cuda:
            batch = batch.to(self.device)

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        batch = Variable(batch.type(Tensor))

        with torch.no_grad():
            predictions = self.model(batch)
            detections = self.non_max_suppression(predictions)

        return detections

    def non_max_suppression(self, prediction):
        """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])
        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            # Filter out confidence scores below threshold
            image_pred = image_pred[image_pred[:, 4] >= self.confidence_thresh]
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Object confidence times class confidence
            score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
            # Sort by it
            image_pred = image_pred[(-score).argsort()]
            class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
            detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
            # Perform non-maximum suppression
            keep_boxes = []
            while detections.size(0):
                large_overlap = self.bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > self.NMS_thresh
                label_match = detections[0, -1] == detections[:, -1]
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                invalid = large_overlap & label_match
                weights = detections[invalid, 4:5]
                # Merge overlapping bboxes by order of confidence
                detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
                keep_boxes += [detections[0]]
                detections = detections[~invalid]
            if keep_boxes:
                output[image_i] = torch.stack(keep_boxes)

        return output

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def xywh2xyxy(self, x):
        y = x.new(x.shape)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
