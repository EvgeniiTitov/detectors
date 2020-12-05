import os

import cv2
import onnxruntime as rt
from typing import List
import numpy as np
import torch


# TODO: Test batch processing
# TODO: Test cuda
# TODO: Rewrite without TORCH dependency (yikes)
# TODO: Check what to use in prod opencv vs PIL. Rewrite with PIL if needed
# TODO: Try CPP onnxruntime

NMS = 0.2
CONF = 0.2
ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326]
]

def load_class_names(namesfile) -> List[str]:
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def letterbox(
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

    if shape[::-1] != new_unpad:  # resize to new size keeping aspect ratio
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def preprocess_image(image: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    img = letterbox(image, new_shape=(img_w, img_h), auto=False)[0]
    img = img[:, :, ::-1] # bgr -> rgb
    img = img.transpose(2, 0, 1)
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # normalize
    return img


def w_non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # CREDITS - https://github.com/ultralytics/yolov5/issues/343
    # box_corner = prediction.new(prediction.shape)
    box_corner = torch.FloatTensor(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        if not image_pred.size(0):
            continue

        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1, keepdim=True
        )
        # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1
        )
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            _, conf_sort_index = torch.sort(
                detections_class[:, 4], descending=True
            )
            detections_class = detections_class[conf_sort_index]
            max_detections = []
            while detections_class.size(0):
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = w_bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections)
            )

    return output


def w_bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    def clip_coords(boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, \
              (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def draw_boxes(
        detections,
        image_source: np.ndarray,
        model_size: int,
        classes: list
) -> None:
    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]
    img_height, img_width = image_source.shape[:2]
    boxs[:, :] = scale_coords(
        (model_size, model_size), boxs[:, :], (img_height, img_width)
    ).round()
    for label, box, conf in zip(labels, boxs, confs):
        label = f"{classes[int(label)]}_{round(float(conf), 4)}"
        left, top, right, bot = box
        cv2.rectangle(image_source, (int(left), int(top)),
                      (int(right), int(bot)), (0, 255, 0), thickness=3)
        cv2.putText(image_source, label, (int(left), int(top) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    return


def main():
    # --- LOAD THE ONNX MODEL ---
    model_path = r"D:\SingleView\SpotIQ\training_results\v5\channels" \
                 r"\v5_1_800\best.onnx"
    test_dir = r"D:\SingleView\SpotIQ\tests\CHANNELS\cropped_ads"
    save_dir = r"D:\SingleView\SpotIQ\tests\RESULTS\CHANNELS\onnx_inference"

    try:
        session = rt.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to create inference session. Error: {e}")
        raise e
    batch_size = session.get_inputs()[0].shape[0]
    img_height = session.get_inputs()[0].shape[2]
    img_width = session.get_inputs()[0].shape[3]
    print(f"[INFO]: Inference image size: {img_height} {img_width}")

    # --- LOAD CLASSES ---
    cls_path = r"dependencies/run6/yolov5.txt"
    classes = load_class_names(cls_path)
    print("[INFO]: Inference classes:", " ".join(map(str, classes)))

    for element in os.listdir(test_dir):
        # --- OPEN AND PREPROCESS IMAGE ---
        path_to_image = os.path.join(test_dir, element)
        image = cv2.imread(path_to_image)
        assert image is not None
        img = preprocess_image(image, img_width, img_height)

        # --- RUN THE MODEL ---
        input_name = session.get_inputs()[0].name
        #output_name = session.get_outputs()[0].name
        outputs = session.run(None, {input_name: img})

        # --- POSTPROCESS RESULTS ---
        if len(outputs) == 4:
            batch_detections = []
        else:
            boxs = []
            a = torch.tensor(ANCHORS).float().view(3, -1, 2)
            anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)
            if len(outputs) == 4:
                outputs = [outputs[1], outputs[2], outputs[3]]
            for index, out in enumerate(outputs):
                out = torch.from_numpy(out)
                batch = out.shape[1]
                feature_w = out.shape[2]
                feature_h = out.shape[3]

                # Feature map corresponds to the original image zoom factor
                stride_w = int(img_width / feature_w)
                stride_h = int(img_height / feature_h)
                grid_x, grid_y = np.meshgrid(np.arange(feature_w),
                                             np.arange(feature_h))

                # cx, cy, w, h
                pred_boxes = torch.FloatTensor(out[..., :4].shape)
                pred_boxes[..., 0] = (torch.sigmoid(
                    out[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w  # cx
                pred_boxes[..., 1] = (torch.sigmoid(
                    out[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h  # cy
                pred_boxes[..., 2:4] = (torch.sigmoid(out[..., 2:4]) * 2) ** 2 * \
                                       anchor_grid[index]  # wh

                conf = torch.sigmoid(out[..., 4])
                pred_cls = torch.sigmoid(out[..., 5:])

                output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                                    conf.view(batch_size, -1, 1),
                                    pred_cls.view(batch_size, -1, len(classes))),
                                   -1)
                boxs.append(output)

            outputx = torch.cat(boxs, 1)
            batch_detections = w_non_max_suppression(
                outputx, len(classes), CONF, NMS
            )

        if batch_detections[0] is not None:
            draw_boxes(batch_detections[0], image, img_height, classes)

        # cv2.imshow("", image)
        # if cv2.waitKey(0) == ord("q"):
        #     break
        cv2.imwrite(os.path.join(save_dir, element), image)


if __name__ == "__main__":
    with torch.no_grad():
        main()
