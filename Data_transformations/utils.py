import json


def read_json_file(json_file_path: str):
    with open(json_file_path, "r") as file:
        return json.load(file)


def write_json_file(json_file_path: str, content: str):
    with open(json_file_path, "w") as file:
        json.dump(content, file)
