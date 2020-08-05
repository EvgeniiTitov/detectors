import os
import json


class JSONOutputGenerator:
    schema = {
        "file": None,
        "classes": list()
    }

    def __init__(self, save_path: str):
        self.save_path = save_path

    def generate_output_json(self, detections: dict) -> list:
        payload = list()
        for k, v in detections.items():
            schema_instance = self.schema.copy()
            schema_instance["file"] = k
            schema_instance["classes"] = [e[-1] for e in v]
            payload.append(schema_instance)

        return payload

    def dump_json(self, payload: list, name: str) -> bool:
        try:
            with open(os.path.join(self.save_path, name + ".json"), "w") as f:
                json.dump(payload, f)
        except Exception as e:
            print(f"Failed to save results to JSON. Error: {e}")
            return False

        return True
