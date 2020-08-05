# Press ESC to continue to the next video in --folder mode.
import cv2
import argparse
import os
import sys


parser = argparse.ArgumentParser(description='Frame Cropper in OPENCV')
parser.add_argument('--folder', help='Path to a folder containing videos.')
parser.add_argument('--video', help='Path to a video file.')
parser.add_argument('--frame', default=25, help='Save a frame once in N frames')
parser.add_argument('--save_path', help='Path to the folder where to save frames cropped')
arguments = parser.parse_args()


def crop_frames(cap, save_path, frame_N, output_name):
    '''
    :param cap: video object
    :param save_path: where to save cropped out frames
    :param frame_N: save a frame once in N frames
    :return:
    '''
    frame_counter = 0
    while cv2.waitKey(1) < 0:
        has_frame, frame = cap.read()
        if not has_frame:
            print("Video", output_name, "has been processed.")
            break

        cv2.imshow("", frame)

        if frame_counter < 1:  # Skip first section of the video
            frame_counter += 1
            print(frame_counter)
            continue

        if frame_counter % frame_N == 0:
            try:
                cv2.imwrite(os.path.join(save_path, output_name + '_' + str(frame_counter) + '.jpg'), frame)
            except Exception as e:
                print(f"Failed while saving a frame. Error: {e}")
                continue

        print(frame_counter)
        frame_counter += 1

def main():
    if not arguments.save_path:
        print("You have to specify the path to a folder where cropped frames will be saved.")
        sys.exit()

    save_path = arguments.save_path  # Path to save frames cropped
    once_in_N_frames = arguments.frame  # Save a frame once in N frames

    if arguments.video:
        video_path = arguments.video

        if not any(video_path.endswith(ext) for ext in ['.mp4', '.MP4', ".avi", ".AVI", ".flv"]):
            raise IOError("The provided extension is not supported")

        output_name = os.path.basename(video_path)[:-4]
        try:
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            print("Failed while creating the cap object")
            raise e

        if not cap.isOpened():
            print("Failed to open the cap for:", output_name)
            sys.exit()

        crop_frames(cap, save_path, int(once_in_N_frames), output_name)

    elif arguments.folder:
        #To process all videos in a folder
        if not os.path.isdir(arguments.folder):
            raise IOError("The provided folder is not a folder")

        for video in os.listdir(arguments.folder):

            if not any(video.endswith(ext) for ext in ['.mp4', '.MP4', ".avi", ".AVI", ".flv"]):  # Discard everything except what we are after
                continue

            video_path = os.path.join(arguments.folder, video)
            output_name = video[:-4]

            try:
                cap = cv2.VideoCapture(video_path)
            except Exception as e:
                print("Failed to create the cap object for:", output_name, "Skipped")
                continue

            if not cap.isOpened():
                print("Failed to open the cap for:", output_name)
                continue

            crop_frames(cap, save_path, int(once_in_N_frames), output_name)

    else:
        print("Incorrect input. Giving up")


if __name__ == '__main__':
    main()
