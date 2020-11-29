import requests
import io
import os
import cv2


def ping():
    URL = r"http://localhost:8091/api/v1/ping"
    r = requests.post(url=URL)
    print(r.text)


def detect_large_channels_model():
    URL = r"http://localhost:8091/api/v1/labels/large_channels"
    archive = r"C:\Users\Evgenii\sv-tv-scopes-detector\tests\scopesdetector" \
              r"\service\prediction\large_channels.tar.gz"
    data = dict()
    with open(archive, "rb") as f:
        content = f.read()
        data["file"] = io.BytesIO(content)
    print("\nMaking request to the large channels model")
    r = requests.post(URL, files=data)
    json_response = r.json()
    json_response.sort(key=lambda e: int(e["file"][:-4]))
    for e in json_response:
        print(e)


def detect_channels_model():
    URL = r"http://localhost:8091/api/v1/labels/channels"
    archive = r"C:\Users\Evgenii\sv-tv-scopes-detector\tests\scopesdetector" \
              r"\service\prediction\channels.tar.gz"
    data = dict()
    with open(archive, "rb") as f:
        content = f.read()
        data["file"] = io.BytesIO(content)
    print("\nMaking request to the channels model")
    r = requests.post(URL, files=data)
    json_response = r.json()
    json_response.sort(key=lambda e: int(e["file"][:-4]))
    for e in json_response:
        print(e)


def detect_on_archives_large_model():
    URL = r"http://localhost:8091/api/v1/labels/retail"
    archive = r"C:\Users\Evgenii\sv-tv-scopes-detector\tests\scopesdetector" \
              r"\service\prediction\retail_large.gz"
    data = dict()
    with open(archive, "rb") as f:
        content = f.read()
        data["file"] = io.BytesIO(content)

    print("\nMaking request to the large model")
    r = requests.post(URL, files=data)
    # for e in r.json():
    #     print()
    #     print(e)

    # I am a lazy fuck
    json_response = r.json()
    json_response.sort(key=lambda e: int(e["file"][:-4]))
    # image_dir = r"D:\Desktop\test_data_large_model"
    # image_names = os.listdir(image_dir)
    # image_names.sort(key=lambda e: int(e[:-4]))
    # for pred, e in zip(json_response, image_names):
    #     image_path = os.path.join(image_dir, e)
    #     image = cv2.imread(image_path)
    #     if image is None:
    #         continue
    #     cv2.putText(image, f"{pred['file']}_{pred['classes']}", (50, 50),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #     cv2.imshow(f"{e}", image)
    #     cv2.waitKey(0)
    for e in json_response:
        print()
        print(e)


def detect_on_archives_small_model():
    URL = r"http://localhost:8091/api/v1/labels"
    path_to_archives = [
        r"C:\Users\Evgenii\sv-tv-scopes-detector\tests\scopesdetector\service\prediction\retail_small_1.tar.gz",
        r"C:\Users\Evgenii\sv-tv-scopes-detector\tests\scopesdetector\service\prediction\retail_small_4.tar.gz",
        r"C:\Users\Evgenii\sv-tv-scopes-detector\tests\scopesdetector\service\prediction\retail_small_5.tar.gz"
    ]
    for path_to_archive in path_to_archives:
        data = {}
        with open(path_to_archive, "rb") as f:
            content = f.read()
            data["file"] = io.BytesIO(content)

        print("\nMaking a request")
        r = requests.post(
            url=URL,
            files=data
        )

        print("Status code:", r.status_code)
        # for e in r.text:
        #     print(e)
        for e in r.json():
            print(e)


def classify():
    URL = r"http://localhost:8091/api/v1/classify/255/255"
    path_to_archive = r"C:\Users\Evgenii\sv-tv-scopes-detector\tests\scopesdetector\service\prediction\ads.tar.gz"
    data = {}
    with open(path_to_archive, "rb") as f:
        content = f.read()
        data["file"] = io.BytesIO(content)

    print("\nMaking a request to Sasha's endpoint")
    r = requests.post(
        url=URL,
        files=data
    )

    print("Status code:", r.status_code)
    print(r.text)


if __name__ == "__main__":
    ping()
    detect_on_archives_small_model()
    detect_on_archives_large_model()
    detect_channels_model()
    detect_large_channels_model()
    #classify()
    ping()