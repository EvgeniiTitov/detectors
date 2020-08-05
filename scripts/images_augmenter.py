from typing import List, Tuple
import skimage
import cv2
import os
import argparse
import random
import numpy as np
import imutils


# TODO: Add colour augmentation


ALLOWED_EXTS = [".jpg", ".jpeg", ".png"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("transparent", help="Folder to images to augment RGBA")
    parser.add_argument("background", help="Folder to backgrounds")
    parser.add_argument("--save_path", default=r"", help="Folder where augmented images will be saved")
    parser.add_argument("--total_images", type=int, default=100, help="Number of augmentation images required")

    parser.add_argument("--rotation_limit", type=int, default=90, help="Max angle of rotation")
    parser.add_argument("--rotation_thresh", type=float, default=0.5,
                        help="Thresh to rotate rgba image on random angle")

    parser.add_argument("--deformation_limit", type=float, default=0.25,
                        help="Max deformation value. 0.2 -> 20% max deformation applied to either height or width")
    parser.add_argument("--deform_thresh", type=float, default=0.5,
                        help="Thresh to deform (stretch/squash) rgba image")

    parser.add_argument("--resize_range", nargs="+", default=[0.2, 0.6],
                        help="Min and max width of augmented image relatively to the background image's width")
    parser.add_argument("--noise_thresh", type=float, default=0.90,
                        help="Thresh for applying random noise on top of the augmented image")
    parser.add_argument("--blur_thresh", type=float, default=0.90,
                        help="Thesh for blurring the output augmented image")

    parser.add_argument("--transparency_range", nargs="+", default=[0.5, 1.0], help="Transparency range in percent")
    parser.add_argument("--transparency_thresh", type=float, default=0.5)
    arguments = parser.parse_args()

    return arguments


class Augmenter:
    NOISES = ['gaussian', 'speckle', 'salt', 'pepper']
    FILTER_SIZES = [(7, 7), (11, 11)]

    def __init__(
            self,
            augmentation: list,
            transparency_range: List[float],
            noise_thresh: float = 0.92,
            blur_thresh: float = 0.92,
            rotation_thresh: float = 0.5,
            deformation_thresh: float = 0.5,
            transparency_thresh: float = 0.5
    ):
        self.augmentation = augmentation
        self.transp_min, self.transp_max = transparency_range
        self.noise_threshold = noise_thresh
        self.blur_thresh = blur_thresh
        self.rotation_thresh = rotation_thresh
        self.deformation_thresh = deformation_thresh
        self.transp_thresh = transparency_thresh

    def __call__(self, image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
        background_image_size = background_image.shape[0:2]
        # Apply augmentation
        for transform in self.augmentation:
            if transform.name == "rotation" and random.random() < self.rotation_thresh:
                continue
            elif transform.name == "deformation" and random.random() < self.deformation_thresh:
                continue
            image = transform(image, background_image_size)

        # Combine the two images - allows the image go beyond the edges a little bit - free augmentation
        try:
            x = random.randint(0, background_image.shape[1] - int(image.shape[1] * .8))
            y = random.randint(0, background_image.shape[0] - int(image.shape[0] * .8))
            image = self.overlay(background_image, image, x, y)
        except Exception as e:
            print(f"Failed while generating location for the augmented image. Error: {e}")
            return

        # Apply noise
        if random.random() >= self.noise_threshold:
            image = self.apply_noise(image)
        elif random.random() >= self.blur_thresh:
            image = self.apply_blur(image)

        return image

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        filter_ = random.choice(self.FILTER_SIZES)

        return cv2.GaussianBlur(image, filter_, 0)

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        noise_type = random.choice(self.NOISES)
        image = skimage.util.random_noise(image, mode=noise_type)

        return image * 255.0

    def overlay(self, background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
        transparency_factor = 1.0
        if random.random() > self.transp_thresh:
            transparency_factor = float(random.randint(int(self.transp_min * 100), int(self.transp_max * 100)) / 100)
            assert 0 <= transparency_factor <= 1, "Wrong transparency factor. Expected [0.0, 1.0]"

        background_width = background.shape[1]
        background_height = background.shape[0]
        # Check if x, y coordinates are beyond the background image edges
        if x >= background_width or y >= background_height:
            return background
        # If overlay image edge(s) go beyond, cut the array accordingly to keep only the overlay image area that
        # happens to be within the background image area
        h, w = overlay.shape[0], overlay.shape[1]
        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]
        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        if overlay.shape[2] < 4:
            # Simulate the 4th alpha channel with values 255 - entirely transparent
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
                ],
                axis=2,
            )
        overlay_image = overlay[..., :3]  # rgb image
        mask = (overlay[..., 3:] / 255.0) * transparency_factor  # alpha channel
        background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

        return background


class Rotation:
    """
    Random logo rotation within the allowed range [-range:range]
    """
    def __init__(self, rotation_limit: int):
        self.name = "rotation"
        assert 0 <= rotation_limit <= 180, "Wrong rotation range value. Expected: [0, 180]"
        self.range = rotation_limit

    def __call__(self, image: np.ndarray, background_size: List[int]) -> np.ndarray:
        rotation_angle = random.randint(-self.range, self.range)
        #rotated_image = imutils.rotate(image, rotation_angle)
        rotated_image = imutils.rotate_bound(image, rotation_angle)

        return rotated_image


class Resize:
    """
    Resizes logo within the allowed range keeping its aspect ratio since its already been deformed
    """
    def __init__(self, resize_range: List[float]):
        self.name = "resize"
        assert len(resize_range) == 2, "Expected 2 values for the resize range"
        assert resize_range[0] < resize_range[1], "Wrong resize range values"
        assert all(0.0 < e < 1.0 for e in resize_range), "Wrong resize range values"
        self.min_allowed, self.max_allowed = resize_range

    def __call__(self, image: np.ndarray, background_size: List[int]) -> np.ndarray:
        background_height, background_width = background_size
        image_height, image_width = image.shape[:2]
        resize_factor = random.randint(int(self.min_allowed * 100), int(self.max_allowed * 100)) / 100

        # Take longer side and rescale it according to the randomly picked resize_factor, which is relative to the
        # background image side
        if image.shape[0] > image.shape[1]:
            new_image_height = background_height * resize_factor
            aspect_ratio_factor = new_image_height / image_height
            new_image_width = image_width * aspect_ratio_factor
        else:
            new_image_width = background_width * resize_factor  # size of rgba image relatively to the background image
            aspect_ratio_factor = new_image_width / image_width
            new_image_height = image_height * aspect_ratio_factor
        resized_image = cv2.resize(image, dsize=(int(new_image_width), int(new_image_height)))

        return resized_image


class Deformation:
    """
    Logo deformations such as squashing and stretching. Default range +/- 20%
    """
    def __init__(self, deformation_limit: float = 0.2):
        self.name = "deformation"
        assert 0 <= deformation_limit < 1.0, "Wrong deformation limit range. Expected 0 - 99%"
        self.limit = deformation_limit

    def __call__(self, image: np.ndarray, background_size: List[int]) -> np.ndarray:
        # Select random: deformation factor
        deformation_factor = random.randrange(int(100 - self.limit * 100), int(100 + self.limit * 100)) / 100.0
        # Pick axis (height or width) will will be applied by the deformation factor
        axis = random.randint(0, 1)
        # Deform the image
        image_height, image_width = image.shape[:2]
        new_size = (image_width, int(image_height * deformation_factor)) if axis == 0 else \
                                                            (int(image_width * deformation_factor), image_height)
        deformed_image = cv2.resize(image, dsize=new_size)

        return deformed_image


def get_paths_to_images(folder: str, img_type: str) -> List[str]:
    if img_type == "rgba":
        return [os.path.join(folder, file) for file in os.listdir(folder) if \
                                                os.path.splitext(file)[-1].lower() in ".png"]
    else:
        return [os.path.join(folder, file) for file in os.listdir(folder) if \
                                                os.path.splitext(file)[-1].lower() in ALLOWED_EXTS]


def confirm_alpha_channel_exists(paths: List[str]) -> Tuple[List[str], List[str]]:
    paths_to_confirmed_rgba_images = list()
    paths_to_not_rgba__images = list()
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("Failed to open:", path)
            continue
        if image.shape[-1] != 4:
            print(f"Image {path} is not RGBA")
            paths_to_not_rgba__images.append(path)
            continue
        paths_to_confirmed_rgba_images.append(path)

    return paths_to_confirmed_rgba_images, paths_to_not_rgba__images


def augment_images(
        rgba_image_paths: List[str],
        background_image_paths: List[str],
        augmenter: Augmenter,
        save_path: str,
        total_number: int
) -> None:
    """
    Keeps running until total_number of augmented images has been generated
    :param rgba_image_paths:
    :param background_image_paths:
    :param augmenter:
    :param save_path:
    :param total_number:
    :return:
    """
    total_augmented = 0
    while True:
        if total_augmented == total_number:
            break
        # Pick random rgba image and background image and read them
        rgba_path = random.choice(rgba_image_paths)
        background_path = random.choice(background_image_paths)
        print("\nPicked for augmentations:")
        print("RGBA:", os.path.basename(rgba_path))
        print("BACKGROUND:", os.path.basename(background_path))

        rgba_image = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
        if rgba_image is None:
            print(f"Failed to open RGBA image {rgba_path}")
            continue
        background_image = cv2.imread(background_path)
        if background_image is None:
            print(f"Failed to open background image {background_path}")
            continue

        augmented_image = augmenter(rgba_image, background_image)
        if augmented_image is not None:
            try:
                cv2.imwrite(os.path.join(save_path, str(total_augmented) + ".jpg"), augmented_image)
            except Exception as e:
                print(f"Failed to save augmented image. Error: {e}")
                continue
            total_augmented += 1

    return


def turn_to_rgba(image_paths: List[str]) -> List[str]:
    successfully_converted = list()
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue
        # Split image into channels, create a new array representing the alpha channel and concat them together
        b, g, r = cv2.split(image)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        image = cv2.merge((b, g, r, alpha))
        path, image_name = os.path.split(image_path)
        image_name = os.path.splitext(image_name)[0] + "_converted.png"
        save_path = os.path.join(path, image_name)
        try:
            cv2.imwrite(save_path, image)
        except Exception as e:
            print(f"Failed to save newly converted to RGBA image. Error: {e}")
            continue
        successfully_converted.append(save_path)

    return successfully_converted


def main():
    # Parse args
    args = parse_arguments()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    assert args.total_images > 0, "Wrong argument total images. Cannot be negative"
    assert 0 <= args.rotation_limit <= 180, "Wrong argument rotation limit. Expected [0, 180]"
    assert 0.0 <= args.deformation_limit < 1, "Wrong argument deformation limit. Expected [0, 1)"
    resize_range = [float(e) for e in args.resize_range]
    assert 0.0 <= args.noise_thresh <= 1.0, "Wrong noise threshold. Expected [0, 100]"
    assert 0.0 <= args.blur_thresh <= 1.0, "Wrong blur threshold. Expected [0, 100]"
    assert 0.0 <= args.rotation_thresh <= 1.0, "Wrong rotation threshold. Expected [0, 100]"
    assert 0.0 <= args.deform_thresh <= 1.0, "Wrong deformation threshold. Expected [0, 100]"
    transparency_range = [float(e) for e in args.transparency_range]
    assert all(0.0 <= e <= 1.0 for e in transparency_range), "Wrong value of transparency range. Expected [0, 1]"
    assert 0.0 <= args.transparency_thresh <= 1.0, "Wrong transprency thresh. Expected [0, 100]"

    # Get paths to RGBA images to augment. Check they are actually RGBA
    image_paths = get_paths_to_images(args.transparent, img_type="rgba")
    if not image_paths:
        print("No images to augment found in the provided folder")
        return

    confirmed_rgba, not_rgba = confirm_alpha_channel_exists(image_paths)
    if not_rgba:
        print("Detected some not RGBA images in the folder with images to augment. Attempting to transform them to RGBA")
        confirmed_rgba.extend(turn_to_rgba(not_rgba))

    if not confirmed_rgba:
        print("No RGBA images to augment")
        return

    # Get paths to background images onto which rbga images will be mapped
    background_paths = get_paths_to_images(args.background, img_type="background")
    if not background_paths:
        print("No background images found in the provided folder")
        return

    # Initialize augmenter alongside all available augmentations
    deformation = Deformation(deformation_limit=args.deformation_limit)
    rotation = Rotation(rotation_limit=args.rotation_limit)
    size = Resize(resize_range)
    augmenter = Augmenter(
        augmentation=[deformation, rotation, size],
        transparency_range=transparency_range,
        noise_thresh=args.noise_thresh,
        blur_thresh=args.blur_thresh,
        rotation_thresh=args.rotation_thresh,
        deformation_thresh=args.deform_thresh,
        transparency_thresh=args.transparency_thresh
    )
    # Augment images
    augment_images(
        rgba_image_paths=confirmed_rgba,
        background_image_paths=background_paths,
        augmenter=augmenter,
        save_path=args.save_path,
        total_number=args.total_images
    )
    print("Augmentation completed")


if __name__ == "__main__":
    main()
