import requests
import io


def ping():
    URL = r"http://localhost:8091/api/v1/ping"
    r = requests.post(url=URL)
    print(r.text)


def detect_on_archives():
    #URL = "http://127.0.0.1:5000//api/v1/detect"
    URL = r"http://localhost:8091/api/v1/detect"
    path_to_archives = [
        r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\singleview_project\test\test_cases\archive_1.tar.gz",
        r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\singleview_project\test\test_cases\archive_2.tar.gz",
        r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\singleview_project\test\test_cases\archive_3.tar.gz",
        r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\singleview_project\test\test_cases\archive_4.tar.gz",
        r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\singleview_project\test\test_cases\archive_5.tar.gz"
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
    detect_on_archives()
    #classify()
    ping()