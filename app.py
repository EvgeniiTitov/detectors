from flask import Flask, request, jsonify
from yolov4_tiny import YOLOv4Tiny
from PIL import Image
import numpy as np


app = Flask(__name__)
app.config["ALLOWED_EXTS"] = [".png", ".jpg", ".jpeg"]
app.config["UPLOAD_FOLDER"] = r""
app.config["MODEL"] = YOLOv4Tiny()


@app.route("/classify", methods=["POST"])
def classify():
    image = Image.open(request.files["image"].stream)
    image = np.array(image)
    detection = app.config["MODEL"].predict([image])[0]
    detected_classes = " ".join([e[-1] for e in detection])
    response = {"result": detected_classes}
    return jsonify(response)


if __name__ == "__main__":
    app.run()
