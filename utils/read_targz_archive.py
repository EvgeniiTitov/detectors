import tarfile
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--archive_path", required=True,
                        help="Path to a archive")
    parser.add_argument("-s", "--save_path", default=r"D:\Desktop\system_output",
                        help="Path where archive contents will be saved")
    arguments = parser.parse_args()

    return arguments


def extract_archive_content(path_to_archive: str, save_path: str) -> None:
    read_tarfile = tarfile.open(path_to_archive)
    if tarfile.is_tarfile(path_to_archive):
        read_tarfile.extractall(save_path)
        print("Archive content's been saved to:", save_path)
    else:
        print("Provided archive is not archive")

    return


def main():
    args = parse_arguments()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    extract_archive_content(args.archive_path, args.save_path)


if __name__ == "__main__":
    main()
