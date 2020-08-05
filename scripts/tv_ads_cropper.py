import argparse
import os
import cv2
import csv
from typing import Tuple, List


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to a folder with .flv videos to process")
    parser.add_argument("save_path", help="Path where results will be saved")
    parser.add_argument("csv",
                        help="Path to CSV file containing information about video sections to crop out")
    return parser.parse_args()


def read_csv(path_to_csv: str) -> Tuple[list, bool]:
    content = list()
    try:
        with open(path_to_csv, newline="") as csv_file:
            content_reader = csv.reader(csv_file)
            # Skip header
            for i, row in enumerate(content_reader):
                if i == 0:
                    continue
                content.append(row)
    except Exception as e:
        print(f"Failed while reading CSV. Error: {e}")
        return content, False

    return content, True


def get_ad_times(csv: List[list], video_name: str) -> list:
    ad_times = list()
    for entry in csv:
        company_name, video_n, start, finish = entry
        company_name = "NEW_WORLD" if company_name == "NEW WORLD" else company_name
        video_n = video_n.split('.')[0]
        if video_name == video_n:
            assert int(start) < int(finish), "Something's wrong with the ad timings: start > finish"
            ad_times.append([company_name, video_n, int(start), int(finish)])

    return ad_times


def process_video(
        path_to_video: str,
        ad_times: List[list],
        save_path:str
) -> bool:
    """

    :param path_to_video:
    :param ad_times:
    :return:
    """
    try:
        cap = cv2.VideoCapture(path_to_video)
    except Exception as e:
        print(f"Failed to open file {path_to_video}. Error: {e}")
        return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    frame_counter = 0
    seconds_counter = 0
    video_writer = None

    # Find earliest and latest ad appearances
    first_ad_start_second = min(e[2] for e in ad_times)
    last_ad_finish_second = max(e[3] for e in ad_times)

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame_counter += 1
        if frame_counter == fps:
            seconds_counter += 1
            frame_counter = 0
            print("Second:", seconds_counter)

        if seconds_counter < first_ad_start_second:
            continue
        elif seconds_counter > last_ad_finish_second:
            break

        # That's a very poor and slow idea yet gets the job done - fix
        for i in range(len(ad_times)):
            # If found ad's start
            if seconds_counter == ad_times[i][2] and video_writer == None:
                ad_name = ad_times[i][0]
                save_name = os.path.join(save_path, f"{os.path.basename(path_to_video)[:-4]}_{ad_name.upper()}_{str(i)}.avi")
                try:
                    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (frame_width, frame_height), True)
                except Exception as e:
                    print(f"Failed while creating a video writer for ad: {ad_name}. Error: {e}")
                    raise e
            # If found ad's finish
            elif seconds_counter == ad_times[i][3] and video_writer:
                video_writer = None

        if video_writer:
            cv2.imshow("", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            video_writer.write(frame)

    cap.release()
    cv2.destroyAllWindows()

    return True


def main() -> None:
    # Parse arguments
    args = read_args()
    assert os.path.exists(args.folder), "The provided folder doesn't exist"
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    assert os.path.exists(args.csv), "You haven't provided the CSV file"
    csv_content, success = read_csv(args.csv)
    if not success:
        return

    # Process videos
    for file in os.listdir(args.folder):
        if not any(file.endswith(ext) for ext in [".flv", ".mp4", ".MP4"]):
            continue
        print("Processing:", file)

        # Check CSV if there're any info about ads we're interested in for this particular video
        file_id = file[20:-4]
        ad_times = get_ad_times(csv_content, file_id)
        if not ad_times:
            print("\nNo ads found for:", file, "Skipping.")
            continue

        success = process_video(
            path_to_video=os.path.join(args.folder, file),
            ad_times=ad_times,
            save_path=args.save_path
        )
        if not success:
            print("Failed while processing", file)

    print("Processing completed")


if __name__ == "__main__":
    main()
