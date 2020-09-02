import os

path_to_images = r"D:\Desktop\darknet-master\build\darknet\x64\data\obj"
txt_with_image_names = r"D:\Desktop\SIngleView\training_results\weights_2_v4_tiny\valid.txt"
save_path = r"D:\Desktop\SIngleView\training_results\weights_2_v4_tiny\validation_images"

def get_image_names(path_to_txt: str) -> list:
    names = list()
    with open(path_to_txt, "r") as f:
        for line in f:
            names.append(line.split("/")[-1].rstrip("\n"))

    return names


names = get_image_names(txt_with_image_names)
for filename in os.listdir(path_to_images):
    if filename.endswith(".txt"):
        continue
    if not filename in names:
        continue
    path_to_image = os.path.join(path_to_images, filename)
    new_name = os.path.join(save_path, filename)
    os.rename(path_to_image, new_name)

print("Done")