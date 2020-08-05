import requests


URL = "http://127.0.0.1:5000/classify"
#path_to_archive = r"D:\Desktop\system_output\single_view_tests\archive_test.tar.gz"
path_to_archive = r"D:\Desktop\system_output\single_view_tests\archive_test\4.png"
archive = open(path_to_archive, "rb")

r = requests.post(url=URL, files= {"image": archive})

print("Request sent")
print(r.json())