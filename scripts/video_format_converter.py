import argparse
import os
import ffmpeg


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str, help="path to video to convert")
    parser.add_argument("--save_path", default=r"D:\Desktop\SIngleView", help="Path where converted video will be saved")
    arguments = parser.parse_args()

    return arguments


def main():
    args = parse_arguments()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    stream = ffmpeg.input(filename=args.video)
    stream = ffmpeg.output(stream, os.path.join(args.save_path, video_name + "_converted.mp4"))
    ffmpeg.run(stream)

    print("Video successfully converted")


if __name__ == "__main__":
    main()
