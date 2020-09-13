import os
import argparse
import multiprocessing
import sys

import cv2
from copy import deepcopy

from logo_detector.yolov4_tiny import YOLOv4Tiny
from logo_detector.utils import test_utils as utils


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dependencies", required=True, type=str,
                        help="Path to model dependencies: cfgs, txts, weights")
    parser.add_argument("-t", "--test_data", type=str,
                        help="Path to test data")
    parser.add_argument("-s", "--save_path", type=str,
                        help="Path to where save processed results")
    arguments = parser.parse_args()
    return vars(arguments)


def test_on_cpu(
        model,
        test_data: dict,
        model_dep: dict,
        save_path: str,
        to_save: bool
) -> list:
    """
    For testing small models on multiple cores (test multiple weights at once)
    """
    proc = os.getpid()
    print(f"==> Process {proc} started")
    # Initialize the model
    txt = model_dep["txt"]
    cfg = model_dep["cfg"]
    weights = model_dep["weights"]

    try:
        model = model(
            device="cpu",
            weights=weights,
            cfg=cfg,
            txt=txt
        )
    except Exception as e:
        print(f"Failed to initialize model. Error: {e}")
        raise e

    print("\nTesting weights:", weights)
    performance = {
        "class": None,
        "nb_imgs": 0,
        "metrics": {
            "TP": 0,
            "FP": 0,
            "FN": 0
    }
    }
    results = list()
    for classname, img_paths in test_data.items():
        print("Testing class:", classname)
        result = deepcopy(performance)
        result["class"] = classname
        count = 0
        for img_path in img_paths:
            image = cv2.imread(img_path)
            if image is None:
                print(f"==> Process {proc} failed to open test image"
                      f" {img_path}")
                continue
            try:
                preds = model.predict(images=[image])
                preds = preds[0]  # Outputs for a batch, but 1 img given
            except Exception as e:
                print(f"==> Process {proc} failed to run the net. Error: {e}")
                raise e
            utils.draw_bbs(image, preds)

            # Metrics
            if preds:
                for pred in preds:
                    if pred[-1] == classname:
                        result["metrics"]["TP"] += 1
                    elif pred[-1] != classname:
                        result["metrics"]["FP"] += 1
            else:
                result["metrics"]["FN"] += 1

            if to_save and save_path:
                img_name = os.path.basename(img_path)
                try:
                    cv2.imwrite(
                        filename=os.path.join(save_path, classname, img_name),
                        img=image
                    )
                except Exception as e:
                    print(f"==> Process {proc} failed to save image. "
                          f"Error: {e}")
                    raise e
            count += 1
        result["nb_imgs"] = count
        results.append(result)

    return results


def test_on_gpu():
    """
    For testing of larger models where gpu and batch processing is beneficial
    """
    pass


def main():
    args = parse_arguments()

    to_save = False
    if args["save_path"]:
        to_save = True
        if not os.path.exists(args["save_path"]):
            os.mkdir(args["save_path"])

    test_data = utils.collect_test_data(args["test_data"])
    print(f"==> Test data collected. Model will be tested against "
          f"{len(test_data)} classes")

    model_dependencies = utils.collect_model_dependencies(args["dependencies"])
    print(f"==> Collected model dependencies. "
          f"{len(model_dependencies['weights'])} weights will be tested")

    f = lambda x: os.path.splitext(os.path.basename(x))[0]
    if to_save:
        weights = [f(w) for w in model_dependencies["weights"]]
        classes = [k for k in test_data.keys()]
        utils.create_dest_dirs(args["save_path"], weights, classes)

    # to_distribute = list()
    # arguments = {
    #     "model": YOLOv4Tiny,
    #     "test_data": test_data,
    #     "save_path": None,
    #     "model_dep": {},
    #     "to_save": False
    # }
    # for weight in model_dependencies["weights"]:
    #     args_to_proc = arguments.copy()
    #     if to_save:
    #         args_to_proc["to_save"] = to_save
    #         args_to_proc["save_path"] = os.path.join(args["save_path"],
    #                                                  f(weight))
    #     args_to_proc["model_dep"] = {
    #         "weights": weight,
    #         "txt": model_dependencies["txt"],
    #         "cfg": model_dependencies["cfg"]
    #     }
    #     to_distribute.append(args_to_proc)

    result_schema = {
        "weights": None,
        "perf": None
    }
    results = list()
    for weight in model_dependencies["weights"]:
        result = result_schema.copy()
        result["weights"] = f(weight)
        dep = {
            "weights": weight,
            "txt": model_dependencies["txt"],
            "cfg": model_dependencies["cfg"]
        }
        res = test_on_cpu(
            model=YOLOv4Tiny,
            test_data=test_data,
            model_dep=dep,
            save_path=os.path.join(args["save_path"], f(weight)),
            to_save=to_save
        )
        result["perf"] = res
        results.append(result)

    accuracies = utils.calculate_naive_acc(results)
    print("Accuracies:")
    for pair in accuracies:
        print(pair)
    utils.visualise_results(accuracies)


if __name__ == "__main__":
    main()
