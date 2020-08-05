from typing import Set, List
import argparse
import itertools
import numpy as np
import cv2
import os


ALLOWED_EXT = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]
AVAILABLE_ALGORITHMS = ["dhash", "humming"]


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to a folder with images")
    parser.add_argument("--remove", default=0, type=int, help="Remove duplicates or not. Expected 0 or 1")
    parser.add_argument("--algorithm", default="dhash", type=str, help="Available algorithms: dhash, humming")
    parser.add_argument("--humming_thresh", type=float, default=10, help="Humming distance threshold")
    return parser.parse_args()


def calculate_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """
    Creates a numerical representation of an input image by calculating relative horizontal gradient between
    adjacent pixels and then converting it into a hash.
    Images with the same numerical representation are considered duplicates
    :param image:
    :param hash_size:
    :return:
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert to gray so we do not need to compare pixel values across all 3 channels
    resized_image = cv2.resize(gray_image, dsize=(hash_size + 1, hash_size))
    difference_image = resized_image[:, 1:] > resized_image[:, :-1]

    return sum([2 ** i for i, v in enumerate(difference_image.flatten()) if v])


def collect_image_paths(folder: str, paths: list) -> list:
    assert os.path.isdir(folder), "Argument folder is not a folder but a file"
    for file in os.listdir(folder):
        path_to_file = os.path.join(folder, file)
        if any(file.endswith(ext) for ext in ALLOWED_EXT):
            paths.append(path_to_file)
        elif os.path.isdir(path_to_file):
            collect_image_paths(path_to_file, paths)

    return paths


def calculate_hamming_distance(hash_1: int, hash_2: int) -> bin:
    return bin(int(hash_1) ^ int(hash_2)).count("1")


def convert_hash(hash_) -> int:
    return int(np.array(hash_, dtype="float64"))


def visualise_similar_images(paths: Set[str], thumbnail_size: int = 400) -> None:
    if len(paths) > 4:
        thumbnail_size = 200

    print("\nPATHS OF SIMILAR/IDENTICAL IMAGES:")
    duplicate_images = None
    for path in paths:
        print(path)
        try:
            image = cv2.imread(path)
        except Exception as e:
            print(f"Failed to open {path} while handling duplicates. Error: {e}. Skipped")
            continue
        image = cv2.resize(image, (thumbnail_size, thumbnail_size))
        # Concat images for visualisation purposes
        if duplicate_images is None:
            duplicate_images = image
        else:
            duplicate_images = np.hstack([duplicate_images, image])
    cv2.imshow("Duplicated images", duplicate_images)
    cv2.waitKey(0)

    return


def generate_hashes_for_images(paths_to_images: List[str], algorithm: str = "dhash") -> dict:
    hashes = dict()
    print("Calculating hashes...")
    for i, path_to_image in enumerate(paths_to_images):
        print(f"Processing: {i} / {len(paths_to_images)}")
        try:
            image = cv2.imread(path_to_image)
        except Exception as e:
            print(f"Failed to open: {path_to_image}. Error: {e}. Image skipped")
            continue

        if algorithm == "dhash":
            hash_ = calculate_dhash(image)
        else:
            hash_ = convert_hash(calculate_dhash(image))

        # Get a list of all image paths for the calculated hash if any. Else, returns an empty list
        paths = hashes.get(hash_, list())
        paths.append(path_to_image)
        hashes[hash_] = paths

    return hashes


def run_dhash_algorithm(args) -> None:
    paths_to_images = list()
    paths_to_images = collect_image_paths(args.folder, paths_to_images)
    hashes = generate_hashes_for_images(paths_to_images, "dhash")
    # Handle duplicates
    for hash_, paths in hashes.items():
        if not len(paths) > 1:
            continue
        if not args.remove:
            visualise_similar_images(paths)
        else:
            for path in paths[1:]:
                os.remove(path)

    return


def run_humming_algorithm(args) -> None:
    paths_to_images = list()
    paths_to_images = collect_image_paths(args.folder, paths_to_images)
    paths_of_similar_images = set()
    hashes = generate_hashes_for_images(paths_to_images, "humming")

    #TODO: Review how you store duplicates. How they are stored together! Wrong
    #      You need to assemble similar images together, so you can delete and leave only 1
    #      Using dict with hashes of similar imgs as keys might work

    for hash_, paths in hashes.items():
        if len(paths) > 1:
            paths_of_similar_images.update(set(paths))

    #Check for similar images: humming distance between hashes within the threshold
    for k1, k2 in itertools.combinations(hashes, 2):
        if calculate_hamming_distance(k1, k2) <= args.humming_thresh:
            paths_of_similar_images.update(set(hashes[k1]))
            paths_of_similar_images.update(set(hashes[k2]))

    if args.remove:
        for path in paths_of_similar_images[1:]:
            os.remove(path)
    else:
        if len(paths_of_similar_images) > 1:
            visualise_similar_images(paths_of_similar_images)
        else:
            print("\nNo similar images to visualise")

    return


def main() -> None:
    args = read_args()
    assert os.path.isdir(args.folder), "Wrong input. Folder expected."
    assert args.humming_thresh >= 0, "Wrong value of hummning distance provided"
    assert args.algorithm.lower().strip() in AVAILABLE_ALGORITHMS, \
                                        "Wrong algorithm requested. Available: dhash, humming"
    if args.algorithm.lower().strip() == "dhash":
        run_dhash_algorithm(args)
    elif args.algorithm.lower().strip() == "humming":
        run_humming_algorithm(args)

    return

if __name__ == "__main__":
    main()
