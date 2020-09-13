import os

from typing import Dict, List
import numpy as np
import cv2
import matplotlib.pyplot as plt


ALLOWED_EXTS = [".png", ".jpg", ".jpeg"]
WEIGHTS_EXTS = [".weights", ".pt", ".pth"]


def collect_test_data(dir_: str) -> Dict[str, list]:
    data = dict()
    for folder in os.listdir(dir_):
        if folder not in data.keys():
            data[folder] = list()
        folder_path = os.path.join(dir_, folder)
        for filename in os.listdir(folder_path):
            if os.path.splitext(filename)[-1].lower() not in ALLOWED_EXTS:
                continue
            data[folder].append(os.path.join(folder_path, filename))

    return data


def collect_model_dependencies(dir_: str) -> dict:
    dependencies = {
        "weights": [],
        "cfg": None,
        "txt": None
    }
    for filename in os.listdir(dir_):
        file_ext = os.path.splitext(filename)[-1].lower()
        if file_ext in WEIGHTS_EXTS:
            dependencies["weights"].append(os.path.join(dir_, filename))
        elif file_ext == ".cfg":
            dependencies["cfg"] = os.path.join(dir_, filename)
        elif file_ext == ".txt":
            dependencies["txt"] = os.path.join(dir_, filename)
        else:
            continue

    return dependencies


def read_classes(txt_path: str) -> List[str]:
    class_names = list()
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)

    return class_names


def draw_bbs(image: np.ndarray, boxes: List[list]) -> None:
    for box in boxes:
        left, top, right, bot, *_ = box
        cv2.rectangle(image, (left, top), (right, bot), (0, 255, 0), 4)

    return


def create_dest_dirs(
        save_dir: str,
        weights: List[str],
        classes: List[str]
) -> None:
    for weight in weights:
        weight_path = os.path.join(save_dir, weight)
        if not os.path.exists(weight_path):
            os.mkdir(weight_path)
        for class_ in classes:
            class_path = os.path.join(weight_path, class_)
            if not os.path.exists(class_path):
                os.mkdir(class_path)

    return


def calculate_naive_acc(results: list) -> List[tuple]:
    output = list()
    for d in results:
        weight  = d["weights"]
        performance = d["perf"]
        acc = 0.0
        count = 0
        fp_total, fn_total, imgs_total = 0, 0, 0
        for item in performance:
            nb_imgs = item["nb_imgs"]
            # Some class have no test images available
            if nb_imgs == 0:
                continue
            tp = item["metrics"]["TP"]
            fp = item["metrics"]["FP"]
            fn = item["metrics"]["FN"]

            class_acc = round(float(tp / nb_imgs), 4)
            acc += class_acc
            count += 1
            fp_total += fp
            fn_total += fn
            imgs_total += nb_imgs
        acc = round(acc / count, 4)
        output.append((weight, acc, fp_total, fn_total, imgs_total))
    return output


def visualise_results(results: list) -> None:
    results.sort(key=lambda e: int(e[0].split("_")[-1]))
    left = list(range(1, len(results) + 1))
    weights = [e[0].split("_")[-1] + f"_t:{e[-1]}_fp:{e[2]}_fn:{e[3]}" for e in results]
    acc = [int(e[1] * 100) for e in results]
    plt.bar(left, acc, tick_label=weights, width=0.8, color=["red", "green"])
    plt.xlabel("Weights")
    plt.ylabel("Naive accuracy")
    plt.show()


if __name__ == "__main__":
    entry = [
        ("yolov4-tiny-obj_113000", 0.18, 100, 453),
        ("yolov4-tiny-obj_12000", 0.28, 100, 453),
        ("yolov4-tiny-obj_2000", 0.38, 100, 453),
        ("yolov4-tiny-obj_4000", 0.68, 100, 453),
        ("yolov4-tiny-obj_8900", 0.88, 100, 453)
    ]
    visualise_results(entry)
