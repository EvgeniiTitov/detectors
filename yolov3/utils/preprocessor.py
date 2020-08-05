import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np


class YOLOv3PrePostProcessor:

    @staticmethod
    def pad_to_square(img, pad_value):
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
    def resize(image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest")
        return image

    @staticmethod
    def preprocess_image(image: np.ndarray, image_size: int = 608) -> torch.Tensor:
        image = transforms.ToTensor()(image)
        # Padd the image so that it is a square
        image, _ = YOLOv3PrePostProcessor.pad_to_square(image, 0)
        # Resize the image to the expected net's resolution
        image = YOLOv3PrePostProcessor.resize(image, image_size)

        return image

    @staticmethod
    def rescale_boxes(boxes: list, current_dim: int, original_shape: tuple) -> list:
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x

        boxes_rescaled = list()
        # WHAT THE FUCK
        boxes = boxes[0].cpu().tolist()

        for box in boxes:
            new_left = int((box[0] - pad_x // 2) * (orig_w / unpad_w))
            new_left = 2 if new_left == 0 else new_left
            new_top = int((box[1] - pad_y // 2) * (orig_h / unpad_h))
            new_top = 2 if new_top == 0 else new_top
            new_right = int((box[2] - pad_x // 2) * (orig_w / unpad_w))
            new_bot = int((box[3] - pad_y // 2) * (orig_h / unpad_h))
            obj_score = round(box[4], 4)
            conf = round(box[5], 4)
            index = int(box[6])

            boxes_rescaled.append(
                [new_left, new_top, new_right, new_bot, obj_score, conf, index]
            )

        return boxes_rescaled
