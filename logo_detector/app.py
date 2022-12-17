from flask import Flask, request, jsonify
from logo_detector.yolov4_tiny import YOLOv4Tiny
from PIL import Image
import numpy as np
import tarfile
import tempfile
import os
import cv2
from typing import List, Dict


app = Flask(__name__)
app.config["ALLOWED_EXTS"] = [".png", ".jpg", ".jpeg"]
app.config["MODEL"] = YOLOv4Tiny()


def create_app() -> Flask:
    return app


def generate_metadata(image: np.ndarray, predictions: List[list]) -> Dict[str, list]:
    """ Determines box's size, orientation and confidence """
    image_size = image.shape[0] * image.shape[1]
    output = dict()
    for pred in predictions:
        # pred: [left, top, right, bot, obj_score, conf, class]
        if pred[-1] not in output.keys():
            output[pred[-1]] = list()
        confidence = pred[-2]
        box_size = round(((pred[2] - pred[0]) * (pred[3] - pred[1])) / image_size, 3)
        assert 0 < box_size < 1, "Expected box size is (0, 1)"
        entry = dict()
        entry["conf"] = str(confidence)
        entry["size"] = str(box_size)
        entry["pos"] = f"{pred[0]} {pred[1]} {pred[2]} {pred[3]}"
        output[pred[-1]].append(entry)

    return output


@app.route("/api/v1/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        print("\nNo files sent in request")
        return jsonify({"status": "error", "msg": "no files in request"}), 400

    file = request.files["file"]
    if file.filename == '':
        print("No file name in request")
        return (jsonify({"status": "error", "msg": "no file name in request"}), 400)

    result = list()

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(mode="r:gz", fileobj=file) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, tmp_dir)

        # TODO: Sasha uses: file.filename.split(".")[0] to get extracted file
        folder_path = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
        image_names = os.listdir(folder_path)

        for i, image_name in enumerate(image_names):
            if image_name.startswith("."):
                continue
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to open image: {image_name}")
                continue

            # All image preprocessing / result postprocessing is implemented inside the model class
            # Model's been written to process a batch of images, so it requires a list of images as input
            boxes = app.config["MODEL"].predict(images=[image])
            # Model returns a list of predictions for the batch: [ [[],], [[],], [[],] ]. In our case batch = 1
            #                                                      img1   img2   img3
            metadata = generate_metadata(image, boxes[0])
            row = {
                "file": image_name,
                "classes": list({e[-1] for e in boxes[0]}),
                "metadata": metadata
            }
            result.append(row)

    return jsonify(result), 200


@app.route("/classify_test", methods=["POST"])
def classify_test():
    image = Image.open(request.files["image"].stream)
    image = np.array(image)
    detection = app.config["MODEL"].predict([image])[0]
    detected_classes = " ".join([e[-1] for e in detection])
    response = {"result": detected_classes}
    return jsonify(response)


if __name__ == "__main__":
    app.run()
