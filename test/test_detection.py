from logo_detector.app import create_app
from flask import url_for
import os
import unittest
import io
import json


class TestModel(unittest.TestCase):
    app = None
    cli = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.testing = True
        cls.cli = cls.app.test_client()

    def test_detection(self):
        test_cases = [
            # (filename, [expected])
            (
                "archive_1.tar.gz",
                [
                    {'classes': ['countdown'], 'file': '00005.jpg'},
                    {'classes': ['countdown'], 'file': '00020.jpg'},
                    {'classes': ['countdown'], 'file': '00045.jpg'},
                    {'classes': ['countdown'], 'file': '00058.jpg'},
                    {'classes': ['countdown'], 'file': '00066.jpg'},
                    {'classes': ['new_world'], 'file': '00693.jpg'},
                    {'classes': ['new_world'], 'file': '00701.jpg'},
                    {'classes': ['new_world'], 'file': '00708.jpg'},
                    {'classes': ['new_world'], 'file': '00714.jpg'},
                    {'classes': ['new_world'], 'file': '00717.jpg'},
                    {'classes': ['paknsave'], 'file': '01422.jpg'},
                    {'classes': ['paknsave'], 'file': '01424.jpg'},
                    {'classes': ['paknsave'], 'file': '01539.jpg'},
                    {'classes': ['paknsave'], 'file': '01688.jpg'},
                    {'classes': ['paknsave'], 'file': '01712.jpg'}
                ]
            ),
            (
                "archive_2.tar.gz",
                [
                    {'classes': ['countdown'], 'file': '00657.jpg'},
                    {'classes': ['countdown'], 'file': '00668.jpg'},
                    {'classes': ['countdown'], 'file': '00670.jpg'},
                    {'classes': ['countdown'], 'file': '00671.jpg'},
                    {'classes': ['countdown'], 'file': '00675.jpg'},
                    {'classes': ['new_world'], 'file': '01387.jpg'},
                    {'classes': ['new_world'], 'file': '01390.jpg'},
                    {'classes': ['new_world'], 'file': '01392.jpg'},
                    {'classes': ['new_world'], 'file': '01398.jpg'},
                    {'classes': ['new_world'], 'file': '01402.jpg'},
                    {'classes': ['paknsave'], 'file': '01808.jpg'},
                    {'classes': ['paknsave'], 'file': '01876.jpg'},
                    {'classes': ['paknsave'], 'file': '01952.jpg'},
                    {'classes': ['paknsave'], 'file': '02006.jpg'},
                    {'classes': ['paknsave'], 'file': '02147.jpg'}
                ]
            ),
            (
                "archive_3.tar.gz",
                [
                    {'classes': ['paknsave'], 'file': '04724.jpg'},
                    {'classes': ['countdown'], 'file': '04730.jpg'},
                    {'classes': ['new_world'], 'file': '04737.jpg'},
                    {'classes': ['new_world'], 'file': '04745.jpg'},
                    {'classes': ['paknsave'], 'file': '04751.jpg'},
                    {'classes': ['countdown'], 'file': '04756.jpg'},
                    {'classes': ['countdown'], 'file': '04768.jpg'},
                    {'classes': ['new_world'], 'file': '04770.jpg'},
                    {'classes': ['new_world'], 'file': '04787.jpg'},
                    {'classes': ['paknsave'], 'file': '04800.jpg'},
                    {'classes': ['new_world'], 'file': '04802.jpg'},
                    {'classes': ['countdown'], 'file': '04807.jpg'},
                    {'classes': ['countdown'], 'file': '04825.jpg'},
                    {'classes': ['paknsave'], 'file': '04828.jpg'},
                    {'classes': ['paknsave'], 'file': '04955.jpg'}
                ]
            ),
            (
                "archive_4.tar.gz",
                [
                    {'classes': ['countdown'], 'file': '00629.jpg'},
                    {'classes': ['new_world'], 'file': '00853.jpg'},
                    {'classes': ['paknsave', 'new_world', 'countdown'], 'file': '01278.jpg'},
                    {'classes': ['paknsave', 'new_world', 'countdown'], 'file': '01376.jpg'}
                ]
            )
        ]
        cwd = os.path.dirname(__file__)  # ...\test
        for filename, expected in test_cases:
            path_to_test_file = os.path.join(cwd, "test_cases", filename)

            data = dict()
            with open(path_to_test_file, "rb") as f:
                content = f.read()
                data["file"] = io.BytesIO(content)

            with self.app.test_request_context():
                response = self.cli.post(
                    url_for("detect"),
                    data=data,
                    follow_redirects=True,
                    content_type='multipart/form-data'
                )
                self.assertEqual(200, response.status_code)
                resp_data = json.load(response.data)
                print("Reponse data:", resp_data)


if __name__ == "__main__":
    unittest.main()
