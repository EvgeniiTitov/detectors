import tarfile
import os
import argparse
import cv2


ALLOWED_EXTS = [".png", ".jpg", ".jpeg"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True,
                        help="Path to a folder containing png images to turn into the archive")
    parser.add_argument("-s", "--save_path", default=r"D:\Desktop\system_output",
                        help="Path where the generated archive will be saved to")
    arguments = parser.parse_args()

    return arguments


def convert_folder_content_to_png(source_folder: str) -> None:
    for i, filename in enumerate(os.listdir(source_folder)):
        if os.path.splitext(filename)[-1].lower() == ".png":
            continue
        image = cv2.imread(os.path.join(source_folder, filename))
        os.remove(os.path.join(source_folder, filename))
        cv2.imwrite(os.path.join(source_folder, f"{i}.png"), image)


        print("{} converted to png".format(filename))

    return


def generate_archive(folder_with_images: str, destination: str) -> None:
    """
    Generates tar gz archive emulating input the model will receive in prod
    :param folder_with_images: path to a folder containing images
    :param destination: path where generated archive will be saved
    :return:
    """
    tgz_path = folder_with_images + ".tar.gz"
    with tarfile.open(tgz_path, 'w:gz') as tar:
        tar.add(folder_with_images, arcname=os.path.basename(folder_with_images))

    return


def main():
    args = parse_arguments()
    # if not os.path.exists(args.save_path):
    #     os.mkdir(args.save_path)
    convert_folder_content_to_png(args.folder)
    generate_archive(args.folder, args.save_path)
    print("Archive created at:", args.save_path)


if __name__ == "__main__":
    main()
